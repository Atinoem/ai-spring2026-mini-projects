"""Build the final Moodle deliverables.

Each team uploads ONE notebook and ONE PDF. This script copies them to

    dist/CS221_Mini_Projects.ipynb   (run the whole notebook and save it first)
    dist/CS221_Mini_Projects.pdf     (build with `make report` first)

and also builds a fallback zip, because the brief asks teams that use helper
code to zip everything into one file that includes the team details:

    <name>/notebook/CS221_Mini_Projects.ipynb   the notebook (team details in its first cells)
    <name>/notebook/cs221_helpers.py            helper module (must sit next to the notebook)
    <name>/CS221_Mini_Projects.pdf              report/report.pdf (team details on the title page)
    <name>/assignments/...                      starter code + our submission.py files
    <name>/scripts/, notebook/fallback/, report/figures/   what the notebook calls / reads / writes
    <name>/README.md, requirements.txt, environment.yml

Files ignored by .gitignore (grader dumps, weights, caches) are left out.

Usage (from the repo root):
    python scripts/package_submission.py            # or: make package
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints", ".DS_Store", "__MACOSX"}


def iter_files(base: Path):
    """Files under `base` that git tracks or would track (i.e. not .gitignored)."""
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--",
                              base.relative_to(ROOT).as_posix()],
                             cwd=ROOT, capture_output=True, check=True).stdout
        paths = [ROOT / p for p in out.decode("utf-8").split("\0") if p]
    except (OSError, subprocess.CalledProcessError):  # not a git checkout: take everything
        paths = list(base.rglob("*"))
    for path in sorted(paths):
        if path.is_file() and not SKIP_PARTS.intersection(path.parts):
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="Group13_CS221_MiniProjects", help="zip / top-level folder name")
    parser.add_argument("--allow-missing-pdf", action="store_true", help="package even if the PDF is not built")
    args = parser.parse_args()

    pdf = ROOT / "report" / "report.pdf"
    if not pdf.exists() and not args.allow_missing_pdf:
        print(f"error: {pdf.relative_to(ROOT)} not found - build the report first (make report)",
              file=sys.stderr)
        return 1

    entries: list[tuple[Path, str]] = [
        (ROOT / "notebook" / "CS221_Mini_Projects.ipynb", "notebook/CS221_Mini_Projects.ipynb"),
        (ROOT / "notebook" / "cs221_helpers.py", "notebook/cs221_helpers.py"),
        (ROOT / "README.md", "README.md"),
        (ROOT / "requirements.txt", "requirements.txt"),
        (ROOT / "environment.yml", "environment.yml"),
    ]
    if pdf.exists():
        entries.append((pdf, "CS221_Mini_Projects.pdf"))
    # Everything the notebook needs to re-run: starter code + our submissions, the experiment scripts it
    # calls, the pure-Python osmium fallback, and the figures directory it writes into.
    for folder in ("assignments", "scripts", "notebook/fallback", "report/figures"):
        entries += [(p, p.relative_to(ROOT).as_posix()) for p in iter_files(ROOT / folder)]

    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    # The two files that are actually uploaded.
    shutil.copy2(ROOT / "notebook" / "CS221_Mini_Projects.ipynb", dist / "CS221_Mini_Projects.ipynb")
    if pdf.exists():
        shutil.copy2(pdf, dist / "CS221_Mini_Projects.pdf")
    print("upload: dist/CS221_Mini_Projects.ipynb" + ("  +  dist/CS221_Mini_Projects.pdf" if pdf.exists() else ""))

    out = dist / f"{args.name}.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arc in entries:
            zf.write(src, f"{args.name}/{arc}")
    print(f"fallback zip: {out.relative_to(ROOT)}  ({len(entries)} files, {out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
