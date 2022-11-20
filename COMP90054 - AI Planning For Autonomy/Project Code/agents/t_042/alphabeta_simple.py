# based on example_bfs.py

import cProfile
from functools import lru_cache
import time
import random
from copy import deepcopy
from pprint import pprint

# from Yinsh.yinsh_model import YinshGameRule
# from Yinsh.yinsh_model import *

from agents.t_042.better_gamerule import * 
from agents.t_042.better_gamerule import YinshGameRule 
import numpy as np

THINKTIME = 0.95
ADJACENT = ((0,1),(0, -1),
            (1,0),(-1 ,0),
            (1,1),(-1,-1))
CORNERS = ((6,0),(9,0),(10,1),(10,4),
           (9,6),(6,9),(4,10),(1,10),
           (0,9),(0,6),(4,1),(1,4))
C_ACCESS = CORNERS + ((5,4),(5,6),(4,5),(6,5),(4,6),(6,4))

inf = float('inf')


@lru_cache(maxsize=None)
def connected_point(pos1, pos2):
    x, y = pos1
    w, z = pos2
    return (x == w or # on h
           y == z or # on v
           x-y == w-z # on d
           )

ALL_POSITIONS = [(i,j) for i in range(11) for j in range(11) if (i,j) not in ILLEGAL_POS]

@lru_cache(maxsize=None)
def connected_points(pos):
    return [other for other in ALL_POSITIONS if connected_point(pos, other)]

def memoize(f):
    memo = {}
    def helper(self, *args):
        if args not in memo:
            memo[args] = f(self, *args)
        return memo[args]
    return helper

class myAgent():
    def __init__(self, _id):
        self.id = _id  # Agent needs to remember its own id.
        ### Player 0 is White, Player 1 is Black;  White goes first.
        # Agent stores an instance of GameRule, from which to obtain functions.
        self.max_depth = 2 # ply
        self.oppo_id = int(not self.id)
        if _id == 0:
            self.marker = 2
            self.oppo_marker = 4
        else:
            self.marker = 4
            self.oppo_marker = 2

        self.thinktime = THINKTIME * 5
        self.game_rule = YinshGameRule(2)


    def turn_to_player(self,turn: 1|-1) :
        return self.id if turn == 1 else self.oppo_id

    def GetSuccessors(self, state: YinshState, turn: 1|-1):
        """Get successor states from given state."""
        player = self.turn_to_player(turn)
        for action in self.game_rule.getLegalActions(state, player):
            successor = state.__deepcopy__(None)
            yield self.game_rule.generateSuccessor(successor, action, player), action

    def IsGameOver(self, state: YinshState) -> bool:
        """Have we game over'd?"""
        return state.counters_left == 0 or state.agents[0].score == 3 or state.agents[1].score == 3

    def RingsHeuristic(self, state: YinshState) -> float:
        """The number of rings won between white and black"""
        return state.agents[self.id].score - state.agents[self.oppo_id].score

    def PositionsConnectedHeuristic(self, state: YinshState) -> float:
        """The number of positions connected to the each ring."""
        def positionsConnected(player):
            return sum([len(connected_points(pos)) for pos in state.ring_pos[player]])
        return positionsConnected(self.id) - positionsConnected(self.oppo_id)

    def num_counters(self, state: YinshState, player):
        return np.count_nonzero(state.board == player)

    def CountersHeuristic(self, state: YinshState) -> float:
        """The number of counters on the board -- one with more counters is better?"""
        return self.num_counters(state, self.marker) - self.num_counters(state, self.oppo_marker)
    
    def Heuristic(self, state: YinshState):
        """Calculate heursitic value of given state from white's (player 1) perspective.
        This is ok since Yinsh is a zero-sum game."""
        if self.IsGameOver(state):
            return 1000000 * self.RingsHeuristic(state)
        return 100000 * self.RingsHeuristic(state) \
            + 10 * self.CountersHeuristic(state) \
            + 1 * self.PositionsConnectedHeuristic(state)
            # + 20*self.ChainHeuristic(state)\
 
    def Negascout(self, state: YinshState, turn: 1 | -1, depth=0, alpha=-inf, beta=inf):
        """ Negascout is a variant of Negamax.
        It is a variant of Negamax where the alpha-beta pruning is replaced by a more efficient version.
        https://en.wikipedia.org/wiki/Principal_variation_search
        """
        # state: The current state (initially is one successor of input game state)
        # Turn: who's turn, initially is the maximizing player (1)
        self.explored += 1
        if self.timeup() or depth == 0 or self.IsGameOver(state):
            return self.Heuristic(state) * turn
        first = True
        # First is the arbitrary first successor 
        for successor, action in self.GetSuccessors(state, turn):
            if first:
                first = False
                score = -self.Negascout(successor, -turn, depth - 1, -beta, -alpha) # principal variation
            else:
                score = -self.Negascout(successor, -turn, depth - 1, -alpha - 1, -alpha) # search with null window
                if alpha < score < beta:
                    score = -self.Negascout(successor, -turn, depth - 1, -beta, -score)  # full window search
            alpha = max(alpha, score)
            if self.stop or alpha >= beta:
                break # beta cutoff
        return alpha

    def timeup(self):
        if self.stop or time.time() - self.start_time > self.thinktime:
            self.stop = True
        return self.stop

    def NegamaxAlphaBetaSearch(self, state: YinshState, turn: 1 | -1, depth, alpha=-inf, beta=inf):
        """ As Yinsh is a zero-sum game, we may adopt the Negamax framework.
        here, max(a,b) = -min(-a,-b), and we proceed with usual alpha-beta pruning
        """
        self.explored += 1
        if self.timeup():
            return -inf
        if depth == 0 or self.IsGameOver(state):
            return self.Heuristic(state) * turn
        value = -inf
        for successor, action in self.GetSuccessors(state, turn):
            value = max(value, -self.NegamaxAlphaBetaSearch(successor, -turn, depth - 1, -beta, -alpha))
            alpha = max(alpha, value)
            if self.stop or alpha >= beta:
                break # cutoff
        if self.timeup():
            print(f"TIMEUP val={value} turn={turn} depth={depth} alpha={alpha} beta={beta}")
        return value
  
    # Take a list of actions and an initial state, and perform alpha-beta search to find the best action.
    # Return the first action that leads to reward, if any was found.
    def SelectAction(self, actions, rootstate):
        self.start_time = time.time()
        self.stop = False
        self.explored = 0

        # convert original gamerule's state to our state representation
        rootstate = YinshState.from_other(rootstate)

        # prevent timeout on win
        if rootstate.agents[self.id].score == 2:
            win_actions = [action for action in actions if action["type"] == 'place, move, remove']
            if win_actions:
                return win_actions[0]

        def loop(depths):
             # Loop through all the actions.
            best_action, best_value = None, -inf
            # Max_depth: from 1 to 9 inclusive
            for max_depth in depths:
                assert max_depth > 0
                 # value, action = self.NegamaxAlphaBetaSearch(rootstate, 1, max_depth)
                for action in actions:
                    succ = deepcopy(rootstate)
                    # Generate successor state given this action and calculate its value
                    state: YinshState = self.game_rule.generateSuccessor(succ, action, self.id)
                    # Update here to try different nega
                    # Negascout: take the successor state, player view (id=0 => 1, id=1 => -1), and max_depth
                    value = -self.NegamaxAlphaBetaSearch(state, -1, max_depth)
                    # New best action? Update the best action and value.
                    if value > best_value:
                        best_value = value
                        best_action = action
                    # If the time is up, return the best action.
                    if self.timeup():
                        return best_action, best_value, max_depth

            return best_action, best_value, max_depth

        # with cProfile.Profile() as pr:
        try:
            best_action, best_value, max_depth = loop([1])#range(1, 10, 2))
        except Exception as e:
            import traceback
            print("Caught exception, choosing a random action...")
            traceback.print_exc()
            # choose a random action
            return random.choice(actions)

        if self.timeup():
            print(f"TIMEUP best_value={best_value} best_action={best_action} max_depth={max_depth}")
            return best_action

        # pr.print_stats(sort='cumtime')
        prev_value = self.Heuristic(rootstate)
        print("alphabeta: current value:", prev_value, "final value:", best_value, "explored", self.explored, 
                "stopped at depth", max_depth, "time", time.time() - self.start_time,
                "action", best_action)
        self.thinktime = THINKTIME

        return best_action
        # next_state =  self.game_rule.generateSuccessor(deepcopy(rootstate), best_action, self.id)
        # if not self.IsGameOver(next_state):
        #     next_moves = [(action, self.Heuristic(state)) for state, action in self.GetSuccessors(next_state, -1)]
        #     next_moves.sort(key = lambda x: x[1])
        #     if next_moves[0][1] != best_value:
        #         print("Warning! Best action value is incorrect!")
        #         print("next move scores:", [s for _, s in next_moves])
        #         # pprint(next_moves)
        #     # else:

        #     # If no reward was found in the time limit, return a random action.
        #     return best_action
       
    
    def ChainHeuristic(self, s):
        return self.ChainHeuristic1(s, 0) - self.ChainHeuristic1(s, 1)

    def ChainHeuristic1(self, s: GameState, opponent_perspective: int = 0):
        '''
        Heuristic that returns score based on the feasible, unblocked sequences of markers.

        params
        ---
        - s: Game state
        - opponent_perspective: Whose view are we looking at? Our view (0) or opponent's view (1)?
        - pts: A dictionary mapping chains to corresponding point
        '''
        pts = {0:0, 1:0, 2:1, 3:3, 4:5, 5:50}

        # What color are we looking at?
        view = self.id ^ opponent_perspective
        tot_mark = 0
        ring = str(RING_1 if view else RING_0)
        counter = str(CNTR_1 if view else CNTR_0)
        opponent_counter = str(CNTR_0 if view else CNTR_1)
        # Get all markers first
        markers = lookup_pos(CNTR_1 if view else CNTR_0,s.board)
        lines_of_interest = set()
        # Get all lines with markers
        for m in markers:
            for line in ['v','h','d']:
                lines_of_interest.add(tuple(self.game_rule.positionsOnLine(m,line)))
        # For each line that has markers, see if a feasible chain exists
        # R: ring       M: my marker     M_opponent: opponent marker
        for p in lines_of_interest:
            p_str  = ('').join([str(s.board[i]) for i in p])
            # Chains of 5 mixed R/M
            for st in re.findall(f'[{ring}{counter}]+',p_str):
                # How many rings needed to move to get 5-marker streak?
                #if r''.match(st): pass
                if len(st)>4: tot_mark+=pts[4-st.count(ring)+opponent_perspective]
            # Incomplete but feasible EMPTY-R-nM (not summing up to 5 yet)
            for st in re.findall(f'{str(EMPTY)*3}{ring}{counter*1}'
                              + f'|{str(EMPTY)*2}{ring}{counter*2}'
                              + f'|{str(EMPTY)*1}{ring}{counter*3}',p_str):
                tot_mark+=pts[st.count(counter)+opponent_perspective]
            # Incomplete but feasible nM-R-EMPTY (not summing up to 5 yet)
            for st in re.findall(f'{counter*1}{ring}{str(EMPTY)*3}'
                              + f'|{counter*2}{ring}{str(EMPTY)*2}'
                              + f'|{counter*3}{ring}{str(EMPTY)*1}',p_str):
                tot_mark+=pts[st.count(counter)+opponent_perspective]
            # R-nM_opponent-EMPTY
            for st in re.findall(f'{ring}{opponent_counter*1}{str(EMPTY)*3}'
                              + f'|{ring}{opponent_counter*2}{str(EMPTY)*2}'
                              + f'|{ring}{opponent_counter*3}{str(EMPTY)*1}'
                              # Needs at least a space to land after jumping over
                              + f'|{ring}{opponent_counter*4}{counter}*{str(EMPTY)*1}',p_str):
                tot_mark+=pts[st.count(opponent_counter)+opponent_perspective]
            # EMPTY-nM_opponent-R
            for st in re.findall(f'{str(EMPTY)*3}{opponent_counter*1}{ring}'
                              + f'|{str(EMPTY)*2}{opponent_counter*2}{ring}'
                              + f'|{str(EMPTY)*1}{opponent_counter*3}{ring}'
                              # Needs at least a space to land after jumping over
                              + f'|{str(EMPTY)*1}{counter}*{opponent_counter*4}{ring}',p_str):
                tot_mark+=pts[st.count(opponent_counter)+opponent_perspective]
            # # nM-EMPTY-nM_opponent-R
            # for st in re.findall(f'[{counter}]+{EMPTY}[{opponent_counter}]+{ring}'
            #                   + f'|{ring}[{opponent_counter}]+{EMPTY}[{counter}]+',p_str):
            #     if st.count(counter) + st.count(opponent_counter) > 2:
            #         tot_mark += pts[4+opponent_perspective]
            # EMPTY-nM_opponent-R-nM
            for st in re.findall(f'{EMPTY}[{opponent_counter}]+{ring}[{counter}]+'
                              + f'|[{counter}]+{ring}[{opponent_counter}]+{EMPTY}',p_str):
                nc = min(st.count(counter) + st.count(opponent_counter),4)
                if nc > 2:
                    tot_mark += pts[nc+opponent_perspective]
            #TODO: R-nM-4M_opponent-EMPTY
            # nM-flippable M_opponent-nM
            # for st in re.finditer(f'{counter}*{opponent_counter}{counter}*',p_str):
            #     start = st.start()
            #     end = st.end()
            #     # No less than 4 counters in the chain, and regardless how many, only 5 valuable
            #     nc = min(end-start,5)
            #     if nc > 4:
            #         oc_loc = p[start+st.group().index(opponent_counter)]
            #         # If it's flippable, it's a potential
            #         line = ('v','d') if p[0][0]==p[-1][0] else ('h','d') if p[0][1]==p[-1][1] else ('v','h')
            #         if not self.Cornered(s,oc_loc,line):
            #             tot_mark += pts[nc-1+opponent_perspective]
        return tot_mark

    def Cornered(self, s: GameState, loc: tuple, dims = ['v','h','d']):
        '''
        Check if a position is unflippable if a marker lies there'''
        for dim in dims:
            line=self.game_rule.positionsOnLine(loc,dim)
            idx = line.index(loc)
            p_str  = ('').join([str(s.board[i]) for i in line])
            p_str = p_str[:idx] + 'x' + p_str[idx + 1:]
            if len(re.findall(f'{EMPTY}[{CNTR_0}{CNTR_1}{RING_0}{RING_1}]*x[{CNTR_0}{CNTR_1}{RING_0}{RING_1}]*{EMPTY}',p_str)):
                return False
        return True


# Return the list of all locations of a specific type of grid on the board
lookup_pos = lambda n,b : list(zip(*np.where(b==n)))

# Returns true if 2 locations lie on the same line
linear = lru_cache(maxsize=None)(lambda l1,l2: l1[0] == l2[0] or l1[1] == l2[1] or sum(l1)==sum(l2))

# Check if location of interest is not in the board or illegal
out_of_board = lambda x,y: x<0 or x>10 or y<0 or y>10
wall = lru_cache(maxsize=None)(lambda l: out_of_board(*l) or l in ILLEGAL_POS)