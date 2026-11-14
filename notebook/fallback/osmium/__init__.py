"""Pure-Python stand-in for the parts of `pyosmium` used by Route Planning's map_util.read_map.

Why this exists
---------------
`osmium` ships a compiled extension (`_osmium.pyd`). On Windows machines with Smart App
Control in enforcement mode that unsigned DLL is refused ("An Application Control policy has
blocked this file"), so the Stanford map cannot be loaded. The notebook helpers put this
package on PYTHONPATH **only when the real osmium fails to import**; with a working osmium
(and on the course autograder) it is never used.

Scope
-----
Implements exactly what `map_util.read_map` needs:
    osmium.SimpleHandler.apply_file(path, locations=True)  -> calls self.node(n) / self.way(w)
    n.id, n.tags (iterable of objects with .k/.v)
    w.tags.get(key, default), w.nodes (objects with .ref and .location.lat/.lon)
    `from osmium import osm` with osm.Node / osm.Way (only used as type hints)

It decodes the OSM PBF format directly: a sequence of length-prefixed BlobHeader/Blob
protobuf messages whose zlib-compressed payloads are PrimitiveBlocks containing (dense)
nodes and ways (https://wiki.openstreetmap.org/wiki/PBF_Format). Coordinates are rounded to
osmium's internal 1e-7 degree precision, so distances match pyosmium bit for bit.
"""

from __future__ import annotations

import struct
import zlib
from types import SimpleNamespace

from . import osm  # noqa: F401  (re-exported for `from osmium import osm`)

__all__ = ["SimpleHandler", "osm"]

_COORD_PRECISION = 10_000_000  # osmium stores coordinates as int32 multiples of 1e-7 degrees


# ----------------------------------------------------------------------------------------
# Minimal protobuf wire-format decoding
# ----------------------------------------------------------------------------------------
def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _zigzag(n: int) -> int:
    """Decode a protobuf sint32/sint64 (zigzag-encoded) value."""
    return (n >> 1) ^ -(n & 1)


def _int64(n: int) -> int:
    """Decode a protobuf int64 varint (negative numbers use two's complement on 64 bits)."""
    return n - (1 << 64) if n >= (1 << 63) else n


def _fields(buf: bytes):
    """Yield (field_number, wire_type, value) for every field of a protobuf message.
    Length-delimited values are returned as bytes, varints as int."""
    pos, end = 0, len(buf)
    while pos < end:
        key, pos = _varint(buf, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _varint(buf, pos)
        elif wire == 2:
            length, pos = _varint(buf, pos)
            value, pos = buf[pos:pos + length], pos + length
        elif wire == 1:
            value, pos = buf[pos:pos + 8], pos + 8
        elif wire == 5:
            value, pos = buf[pos:pos + 4], pos + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield field, wire, value


def _packed_varints(buf: bytes) -> list[int]:
    out, pos = [], 0
    while pos < len(buf):
        value, pos = _varint(buf, pos)
        out.append(value)
    return out


def _packed_sint_deltas(buf: bytes) -> list[int]:
    """Packed sint64 values that are delta-coded (ids, lats, lons, way refs)."""
    out, running = [], 0
    for raw in _packed_varints(buf):
        running += _zigzag(raw)
        out.append(running)
    return out


# ----------------------------------------------------------------------------------------
# Objects handed to the user's handler (mirroring the pyosmium attribute names used)
# ----------------------------------------------------------------------------------------
class _TagList(dict):
    """dict of tags; iterating yields objects with .k/.v like osmium.osm.TagList."""

    def __iter__(self):
        return (SimpleNamespace(k=k, v=v) for k, v in self.items())


class _Location:
    __slots__ = ("lat", "lon")

    def __init__(self, lat_e7: int, lon_e7: int):
        self.lat = lat_e7 / _COORD_PRECISION
        self.lon = lon_e7 / _COORD_PRECISION


class _NodeRef:
    __slots__ = ("ref", "location")

    def __init__(self, ref: int, location: _Location | None):
        self.ref = ref
        self.location = location


# ----------------------------------------------------------------------------------------
# PBF file reading
# ----------------------------------------------------------------------------------------
def _blobs(path: str):
    """Yield (blob_type, decompressed_payload) for every block in a .pbf file."""
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    while pos < len(data):
        (header_len,) = struct.unpack(">I", data[pos:pos + 4])
        pos += 4
        blob_type, blob_size = None, 0
        for field, _, value in _fields(data[pos:pos + header_len]):
            if field == 1:
                blob_type = value.decode()
            elif field == 3:
                blob_size = value
        pos += header_len
        blob, pos = data[pos:pos + blob_size], pos + blob_size
        payload = None
        for field, _, value in _fields(blob):
            if field == 1:            # raw (uncompressed)
                payload = value
            elif field == 3:          # zlib_data
                payload = zlib.decompress(value)
        if payload is None:
            raise ValueError("unsupported PBF blob compression (only raw/zlib are handled)")
        yield blob_type, payload


def _decode_block(payload: bytes):
    """Decode one PrimitiveBlock into lists of nodes and ways (plain tuples)."""
    strings, groups = [], []
    granularity, lat_offset, lon_offset = 100, 0, 0
    for field, _, value in _fields(payload):
        if field == 1:                                    # StringTable
            strings = [s.decode("utf-8") for f, _, s in _fields(value) if f == 1]
        elif field == 2:
            groups.append(value)
        elif field == 17:
            granularity = value
        elif field == 19:
            lat_offset = _int64(value)
        elif field == 20:
            lon_offset = _int64(value)

    def to_e7(raw: int, offset: int) -> int:
        # nanodegrees = offset + granularity * raw; libosmium converts to its 1e-7 degree
        # integers with C++ integer division (truncation towards zero), reproduced here.
        nano = offset + granularity * raw
        return nano // 100 if nano >= 0 else -((-nano) // 100)

    nodes, ways = [], []
    for group in groups:
        for field, _, value in _fields(group):
            if field == 1:                                # Node (non-dense)
                nid, keys, vals, lat, lon = 0, [], [], 0, 0
                for f, _, v in _fields(value):
                    if f == 1:
                        nid = _zigzag(v)
                    elif f == 2:
                        keys = _packed_varints(v)
                    elif f == 3:
                        vals = _packed_varints(v)
                    elif f == 8:
                        lat = _zigzag(v)
                    elif f == 9:
                        lon = _zigzag(v)
                tags = _TagList((strings[k], strings[x]) for k, x in zip(keys, vals))
                nodes.append((nid, to_e7(lat, lat_offset), to_e7(lon, lon_offset), tags))
            elif field == 2:                              # DenseNodes
                ids, lats, lons, kv = [], [], [], []
                for f, _, v in _fields(value):
                    if f == 1:
                        ids = _packed_sint_deltas(v)
                    elif f == 8:
                        lats = _packed_sint_deltas(v)
                    elif f == 9:
                        lons = _packed_sint_deltas(v)
                    elif f == 10:
                        kv = _packed_varints(v)
                it = iter(kv)
                for nid, lat, lon in zip(ids, lats, lons):
                    tags = _TagList()
                    if kv:                                # keys_vals: k v k v ... 0 per node
                        for k in it:
                            if k == 0:
                                break
                            tags[strings[k]] = strings[next(it)]
                    nodes.append((nid, to_e7(lat, lat_offset), to_e7(lon, lon_offset), tags))
            elif field == 3:                              # Way
                wid, keys, vals, refs = 0, [], [], []
                for f, _, v in _fields(value):
                    if f == 1:
                        wid = _int64(v)
                    elif f == 2:
                        keys = _packed_varints(v)
                    elif f == 3:
                        vals = _packed_varints(v)
                    elif f == 8:
                        refs = _packed_sint_deltas(v)
                tags = _TagList((strings[k], strings[x]) for k, x in zip(keys, vals))
                ways.append((wid, tags, refs))
            # field 4 (relations) and 5 (changesets) are not needed by map_util
    return nodes, ways


class SimpleHandler:
    """Subset of osmium.SimpleHandler: override node()/way() and call apply_file()."""

    def node(self, n):  # pragma: no cover - meant to be overridden
        pass

    def way(self, w):  # pragma: no cover - meant to be overridden
        pass

    def apply_file(self, filename: str, locations: bool = False, idx: str | None = None) -> None:
        all_nodes, all_ways = [], []
        for blob_type, payload in _blobs(str(filename)):
            if blob_type == "OSMData":
                nodes, ways = _decode_block(payload)
                all_nodes.extend(nodes)
                all_ways.extend(ways)

        # osmium visits all nodes first, then all ways (the order of a sorted PBF file).
        location_index = {}
        for nid, lat_e7, lon_e7, tags in all_nodes:
            location_index[nid] = (lat_e7, lon_e7)
            self.node(SimpleNamespace(id=nid, tags=tags, location=_Location(lat_e7, lon_e7)))
        for wid, tags, refs in all_ways:
            node_refs = [
                _NodeRef(ref, _Location(*location_index[ref]) if locations and ref in location_index else None)
                for ref in refs
            ]
            self.way(SimpleNamespace(id=wid, tags=tags, nodes=node_refs))
