from util import manhattan_distance
from game import Directions
import random
import util
from typing import Any, DefaultDict, List, Set, Tuple

from game import Agent
from pacman import GameState



class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def __init__(self):
        self.last_positions = []
        self.dc = None

    def get_action(self, game_state: GameState):
        """
        getAction chooses among the best options according to the evaluation function.

        getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East}
        ------------------------------------------------------------------------------
        Description of GameState and helper functions:

        A GameState specifies the full game state, including the food, capsules,
        agent configurations and score changes. In this function, the |game_state| argument
        is an object of GameState class. Following are a few of the helper methods that you
        can use to query a GameState object to gather information about the present state
        of Pac-Man, the ghosts and the maze.

        game_state.get_legal_actions(agentIndex):
            Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

        game_state.generate_successor(agentIndex, action):
            Returns the successor state after the specified agent takes the action.
            Pac-Man is always agent 0.

        game_state.get_pacman_state():
            Returns an AgentState object for pacman (in game.py)
            state.configuration.pos gives the current position
            state.direction gives the travel vector

        game_state.get_ghost_states():
            Returns list of AgentState objects for the ghosts

        game_state.get_num_agents():
            Returns the total number of agents in the game

        game_state.get_score():
            Returns the score corresponding to the current state of the game


        The GameState class is defined in pacman.py and you might want to look into that for
        other helper methods, though you don't need to.
        """
        # Collect legal moves and successor states
        legal_moves = game_state.get_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(
            game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(
            len(scores)) if scores[index] == best_score]
        # Pick randomly among the best
        chosen_index = random.choice(best_indices)


        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state: GameState, action: str) -> float:
        """
        The evaluation function takes in the current GameState (defined in pacman.py)
        and a proposed action and returns a rough estimate of the resulting successor
        GameState's value.

        The code below extracts some useful information from the state, like the
        remaining food (old_food) and Pacman position after moving (new_pos).
        new_scared_times holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = current_game_state.generate_pacman_successor(action)
        new_pos = successor_game_state.get_pacman_position()
        old_food = current_game_state.get_food()
        new_ghost_states = successor_game_state.get_ghost_states()
        new_scared_times = [
            ghost_state.scaredTimer for ghost_state in new_ghost_states]

        return successor_game_state.get_score()


def score_evaluation_function(current_game_state: GameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return current_game_state.get_score()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='score_evaluation_function', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluation_function = util.lookup(evalFn, globals())
        self.depth = int(depth)

######################################################################################
# Problem 1b: implementing minimax


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (problem 1)
    """

    def get_action(self, game_state: GameState) -> str:
        """
          Returns the minimax action from the current game_state using self.depth
          and self.evaluation_function. Terminal states can be found by one of the following:
          pacman won, pacman lost or there are no legal moves.

          Don't forget to limit the search depth using self.depth. Also, avoid modifying
          self.depth directly (e.g., when implementing depth-limited search) since it
          is a member variable that should stay fixed throughout runtime.

          Here are some method calls that might be useful when implementing minimax.

          game_state.get_legal_actions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          game_state.generate_successor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          game_state.get_num_agents():
            Returns the total number of agents in the game

          game_state.get_score():
            Returns the score corresponding to the current state of the game

          game_state.is_win():
            Returns True if it's a winning state

          game_state.is_lose():
            Returns True if it's a losing state

          self.depth:
            The depth to which search should continue

          Hint: To implement minimax recursively, consider creating a helper function that
          takes the current game state, current depth, and current agent index as parameters.
        """

        # BEGIN_YOUR_CODE (our solution is 22 lines of code, but don't worry if you deviate from this)
        num_agents = game_state.get_num_agents()

        def successor_turn(agent: int, depth: int) -> Tuple[int, int]:
            """Whose turn is next, and the depth left: one unit of depth is Pac-Man plus every
            ghost moving once, so depth only decreases when play returns to Pac-Man."""
            next_agent = (agent + 1) % num_agents
            return next_agent, (depth - 1 if next_agent == self.index else depth)

        def value(state: GameState, depth: int, agent: int) -> float:
            """V_minmax(s, d): Pac-Man (agent 0) maximises, every ghost minimises."""
            actions = state.get_legal_actions(agent)
            if state.is_win() or state.is_lose() or not actions:
                return state.get_score()                      # Utility(s) at end states
            if depth == 0:
                return self.evaluation_function(state)        # Eval(s) at the depth limit
            next_agent, next_depth = successor_turn(agent, depth)
            child_values = [value(state.generate_successor(agent, action), next_depth, next_agent)
                            for action in actions]
            return max(child_values) if agent == self.index else min(child_values)

        # Root (a max node): choose Pac-Man's action whose successor has the highest value.
        next_agent, next_depth = successor_turn(self.index, self.depth)
        return max(game_state.get_legal_actions(self.index),
                   key=lambda action: value(game_state.generate_successor(self.index, action),
                                            next_depth, next_agent))
        # END_YOUR_CODE

######################################################################################
# Problem 2a: implementing alpha-beta


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (problem 2)
      You may reference the pseudocode for Alpha-Beta pruning here:
      en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning#Pseudocode
    """

    def get_action(self, game_state: GameState) -> str:
        """
          Returns the minimax action using self.depth and self.evaluation_function
        """

        # BEGIN_YOUR_CODE (our solution is 43 lines of code, but don't worry if you deviate from this)
        num_agents = game_state.get_num_agents()

        def successor_turn(agent: int, depth: int) -> Tuple[int, int]:
            next_agent = (agent + 1) % num_agents
            return next_agent, (depth - 1 if next_agent == self.index else depth)

        def value(state: GameState, depth: int, agent: int, alpha: float, beta: float) -> float:
            """Minimax value with alpha-beta pruning.

            alpha = best value Pac-Man (max) can already guarantee on the path to the root,
            beta  = best (lowest) value the ghosts (min) can already guarantee.
            With several ghosts, consecutive min layers share the same beta: each ghost's
            layer prunes as soon as its value drops to <= alpha, exactly like a single min node.
            """
            actions = state.get_legal_actions(agent)
            if state.is_win() or state.is_lose() or not actions:
                return state.get_score()
            if depth == 0:
                return self.evaluation_function(state)
            next_agent, next_depth = successor_turn(agent, depth)

            if agent == self.index:                      # max node
                best = float('-inf')
                for action in actions:
                    best = max(best, value(state.generate_successor(agent, action),
                                           next_depth, next_agent, alpha, beta))
                    if best >= beta:                     # a min ancestor will never allow this
                        return best
                    alpha = max(alpha, best)
                return best
            else:                                        # min node (one ghost)
                best = float('inf')
                for action in actions:
                    best = min(best, value(state.generate_successor(agent, action),
                                           next_depth, next_agent, alpha, beta))
                    if best <= alpha:                    # Pac-Man already has something better
                        return best
                    beta = min(beta, best)
                return best

        # Root (max node): evaluate children left to right, tightening alpha as we go.
        next_agent, next_depth = successor_turn(self.index, self.depth)
        alpha, best_action = float('-inf'), None
        for action in game_state.get_legal_actions(self.index):
            child_value = value(game_state.generate_successor(self.index, action),
                                next_depth, next_agent, alpha, float('inf'))
            if child_value > alpha or best_action is None:
                alpha, best_action = child_value, action
        return best_action
        # END_YOUR_CODE

######################################################################################
# Problem 3b: implementing expectimax


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (problem 3)
    """

    def get_action(self, game_state: GameState) -> str:
     """
       Returns the expectimax action using self.depth and self.evaluation_function

       All ghosts should be modeled as choosing uniformly at random from their
       legal moves.
     """

     # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
     num_agents = game_state.get_num_agents()

     def successor_turn(agent: int, depth: int) -> Tuple[int, int]:
         next_agent = (agent + 1) % num_agents
         return next_agent, (depth - 1 if next_agent == self.index else depth)

     def value(state: GameState, depth: int, agent: int) -> float:
         """V_exptmax(s, d): Pac-Man maximises; each ghost picks a legal move uniformly at
         random, so a ghost node's value is the *average* of its children's values."""
         actions = state.get_legal_actions(agent)
         if state.is_win() or state.is_lose() or not actions:
             return state.get_score()
         if depth == 0:
             return self.evaluation_function(state)
         next_agent, next_depth = successor_turn(agent, depth)
         child_values = [value(state.generate_successor(agent, action), next_depth, next_agent)
                         for action in actions]
         if agent == self.index:
             return max(child_values)
         return sum(child_values) / len(child_values)

     next_agent, next_depth = successor_turn(self.index, self.depth)
     return max(game_state.get_legal_actions(self.index),
                key=lambda action: value(game_state.generate_successor(self.index, action),
                                         next_depth, next_agent))
     # END_YOUR_CODE

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function


def better_evaluation_function(current_game_state: GameState) -> float:
    """
      Your extreme, unstoppable evaluation function (problem 4). Note that you can't fix a seed in this function.
    """

    # BEGIN_YOUR_CODE (our solution is 16 lines of code, but don't worry if you deviate from this)
    # Start from the game score (it already rewards eating food, ghosts and winning) and add
    # terms that make *progress* visible to a depth-2 search, which otherwise sees no
    # difference between most moves:
    #   - true maze distance to the nearest food (reciprocal, so being close matters most),
    #   - remaining food and capsules (fewer is better),
    #   - scared ghosts within reach are worth chasing (+200 points each when eaten),
    #   - active ghosts are dangerous only when they are adjacent.
    if current_game_state.is_win() or current_game_state.is_lose():
        return current_game_state.get_score()

    pacman_pos = current_game_state.get_pacman_position()
    walls = current_game_state.get_walls()
    food = current_game_state.get_food()

    # Breadth-first search from Pac-Man gives real maze distances (walls included).
    dist = {pacman_pos: 0}
    frontier = [pacman_pos]
    while frontier:
        next_frontier = []
        for (x, y) in frontier:
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if not walls[nx][ny] and (nx, ny) not in dist:
                    dist[(nx, ny)] = dist[(x, y)] + 1
                    next_frontier.append((nx, ny))
        frontier = next_frontier

    def maze_distance(pos) -> int:
        return dist.get((int(pos[0]), int(pos[1])), 1000)

    food_list = food.as_list()
    nearest_food = min(maze_distance(f) for f in food_list) if food_list else 0
    capsules = current_game_state.get_capsules()

    value = current_game_state.get_score()
    value += 10.0 / (nearest_food + 1)
    value -= 4.0 * len(food_list)
    value -= 20.0 * len(capsules)

    for ghost in current_game_state.get_ghost_states():
        d = maze_distance(ghost.get_position())
        if ghost.scaredTimer > d:          # reachable before it recovers: hunt it
            value += 200.0 / (d + 1)
        elif d <= 1:                       # an active ghost next to us: very bad
            value -= 500.0
        else:
            value -= 2.0 / d
    return value
    # END_YOUR_CODE


# Abbreviation
better = better_evaluation_function
