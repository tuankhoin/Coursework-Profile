# based on example_bfs.py

import cProfile
from functools import lru_cache
import time
import random
from copy import deepcopy

# from Yinsh.yinsh_model import YinshGameRule
# from Yinsh.yinsh_model import *

from agents.t_042.better_gamerule import * 
from agents.t_042.better_gamerule import YinshGameRule 
import numpy as np

THINKTIME = 0.9
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

        self.game_rule = YinshGameRule(2)


    def turn_to_player(self,turn: 1|-1) :
        return self.id if turn == 1 else self.oppo_id

    def GetActions(self, state: YinshState, turn: 1 | -1):
        """Get legal actions from given state."""
        return self.game_rule.getLegalActions(state, self.turn_to_player(turn))

    def GetSuccessors(self, state: YinshState, turn: 1|-1):
        """Get successor states from given state."""
        player = self.turn_to_player(turn)
        for action in self.game_rule.getLegalActions(state, player):
            successor = deepcopy(state)
            yield self.game_rule.generateSuccessor(successor, action, player)

    def IsGameOver(self, state: YinshState) -> bool:
        """Have we game over'd?"""
        for agent in state.agents:
            if agent.score == 3:
                return True
        return False

    def RingsHeuristic(self, state: YinshState) -> float:
        """The number of rings won between white and black"""
        return state.agents[self.id].score - state.agents[self.oppo_id].score

    @lru_cache(maxsize=2000)
    def RingMovesHeuristic(self, state: YinshState) -> float:
        """The number of places we can move a ring to."""
        def num_ring_moves(player):
            return sum([len(self.game_rule.movementsAlongLine(pos, line)) 
                    for pos in state.ring_pos[player] for line in ['h','v','d']])
        return 0
        return num_ring_moves(self.id) - num_ring_moves(self.oppo_id)
        # return len(self.GetActions(state, 1)) - len(self.GetActions(state, -1))

    def PositionsConnectedHeuristic(self, state: YinshState) -> float:
        """The number of positions connected to the board."""
        def positionsConnected(player):
            return len(set([p1 for pos in state.ring_pos[player] for p1 in connected_points(pos)]))
        return positionsConnected(self.id) - positionsConnected(self.oppo_id)

    def num_counters(self, state: YinshState, player):
        return np.count_nonzero(state.board == player)

    def CountersHeuristic(self, state: YinshState) -> float:
        """The number of counters on the board -- one with more counters is better?"""
        return self.num_counters(state, self.marker) - self.num_counters(state, self.oppo_marker)
        
    def Heuristic(self, state):
        if state.counters_left == 0:
            return 100 * self.RingsHeuristic(state)
        return 10000*self.RingsHeuristic(state)\
                + 20*self.ChainHeuristic(state)\
                + 10*self.CornerHeuristic(state) \
                + 1 * self.PositionsConnectedHeuristic(state) \
                + 10 * self.CountersHeuristic(state)

    # def Heuristic(self, state: YinshState):
    #     """Calculate heursitic value of given state from white's (player 1) perspective.
    #     This is ok since Yinsh is a zero-sum game."""
    #     if state.counters_left == 0:
    #         return 1000000 * self.RingsHeuristic(state) # someone wins
    #     return 100000 * self.RingsHeuristic(state) \
    #         + 10 * self.CountersHeuristic(state) \
    #         + 1 * self.RingMovesHeuristic(state) \
    #         + 20 * self.ChainHeuristic(state) \
    #         + 2 * self.CornerHeuristic(state) \
    #         + 1 * self.PositionsConnectedHeuristic(state)

    def NegamaxSearch(self, state: YinshState, turn: 1 | -1, depth: int):
        """ Negamax search; which is minimax where max(a,b) = -min(-a,-b), since Yinsh is a zero-sum game.
        Depth starts at max_depth, and is decremented by 1 each time a node is expanded.
        Turn = 1 means we are the maximizing player, and turn = -1 means we are the minimizing player.
        """
        self.explored += 1
        # If the depth is 0 or the game is over, return the heuristic value of the state.
        if depth == 0 or self.IsGameOver(state):
            return self.Heuristic(state) * turn
        value = -inf
        for successor in self.GetSuccessors(state, turn):
            value = max(value, -self.NegamaxSearch(successor, -turn, depth - 1))
        return value
 
    def Negascout(self, state: YinshState, turn: 1 | -1, depth=0, alpha=-inf, beta=inf):
        """ Negascout is a variant of Negamax.
        It is a variant of Negamax where the alpha-beta pruning is replaced by a more efficient version.
        https://en.wikipedia.org/wiki/Principal_variation_search
        """
        # state: The current state (initially is one successor of input game state)
        # Turn: who's turn, initially is the maximizing player (1)
        self.explored += 1
        if self.stop or time.time() - self.start_time > THINKTIME:
            self.stop = True
            # Stop, if this is our own, return -inf; and if it is the opponent's, return inf (being pessimistic)
            return -turn * inf
        # If the depth is 0 or the game is over, return the heuristic value of the state.
        if depth == 0 or self.IsGameOver(state):
            return self.Heuristic(state) * turn
        first = True
        # First is the arbitrary first successor 
        for successor in self.GetSuccessors(state, turn):
            if first:
                score = -self.Negascout(successor, -turn, depth - 1, -beta, -alpha) # principal variation
                first = False
            else:
                score = -self.Negascout(successor, -turn, depth - 1, -alpha - 1, -alpha) # search with null window
                if alpha < score < beta:
                    score = -self.Negascout(successor, -turn, depth - 1, -beta, -score)  # full window search
            alpha = max(alpha, score)
            if self.stop or alpha >= beta:
                break # beta cutoff
        # print(f"depth={depth} a={alpha} b={beta} turn={turn} value={value}")
        return alpha


    def NegamaxAlphaBetaSearch(self, state: YinshState, turn: 1 | -1, depth, alpha=-inf, beta=inf):
        """ As Yinsh is a zero-sum game, we may adopt the Negamax framework.
        here, max(a,b) = -min(-a,-b), and we proceed with usual alpha-beta pruning
        """
        self.explored += 1
        # If the depth is 0 or the game is over, return the heuristic value of the state.
        if self.stop or time.time() - self.start_time > THINKTIME:
            self.stop = True
            # Being pessimistic, we return the worst possible value.
            return -turn * inf
        # Reach the bottom of the tree, return the heuristic value of the state.
        # Max (our): turn = 1 , Min (opponent): turn = -1
        if depth == 0 or self.IsGameOver(state):
            return self.Heuristic(state) * turn
        value = -inf
        for successor in self.GetSuccessors(state, turn):
            value = max(value, -self.NegamaxAlphaBetaSearch(successor, -turn, depth - 1, -beta, -alpha))
            alpha = max(alpha, value)
            if self.stop or alpha >= beta:
                break # cutoff
        # print(f"depth={depth} a={alpha} b={beta} turn={turn} value={value}")
        return value

    def GreedySearch(self, state: YinshState, turn: 1 | -1, depth: int):
        """ Just calculate the heuristic of the next value without doing search"""
        return self.Heuristic(state)

    def SelectRingPlacementAction(self, actions, rootstate):
        ## Use Khoi's master starting strategy to place rings
        def adjacentPlacement():
            '''Look at 6 adjacent places surrounding the new ring, 
            then place it where it has most mobility'''
            prev_ring = rootstate.agents[int(not self.id)].last_action['place pos']
            where = np.argmax([self.Mobility(rootstate,(prev_ring[0]+c[0],prev_ring[1]+c[1])) 
                                                for c in ADJACENT])
            act = ADJACENT[where]
            return {'type':'place ring','place pos':(prev_ring[0]+act[0],prev_ring[1]+act[1])}

        # Strategy will change depends on the assigned color
        if not self.id:
        # White: put 1st 3 rings to flexible positions, then control black
            # Empty board: Place ring next to middle. It open ways to corners
            if rootstate.rings_to_place == 10: return {'type': 'place ring','place pos': (5,4)}
            # Then put a close by one in the corner or can access corners
            if rootstate.rings_to_place in (6,8):
                rhombus = [(5,6),(6,5)] + ([(6,9),(4,1)] if rootstate.rings_to_place==6 else [])
                where = np.argmax([self.Mobility(rootstate,c) for c in rhombus])
                return {'type':'place ring','place pos':rhombus[where]}
            # Defence at end: line ppponent ring intersections if they are close, else next to most recent ring
            if rootstate.rings_to_place in (2,4): 
                places = self.RingPlacement(rootstate,*rootstate.ring_pos[int(not self.id)][-1:-3:-1])
                if not len(places): return adjacentPlacement()
                where = np.argmax([self.Mobility(rootstate,c) for c in places])
                return {'type':'place ring','place pos':places[where]}
        else:
        # Black: 1st 3 rings try to block white paths, last 2 try to create clear space
            # First ring try to put next to opponent's first
            if rootstate.rings_to_place == 9:
                return adjacentPlacement()
            # Next: line intersection if they are close, else next to most recent ring
            if rootstate.rings_to_place in (5,7):
                places = self.RingPlacement(rootstate,*rootstate.ring_pos[int(not self.id)][-1:-3:-1])
                if not len(places): return adjacentPlacement()
                where = np.argmax([self.Mobility(rootstate,c) for c in places])
                return {'type':'place ring','place pos':places[where]}
            # Get a good spot for itself: put to a flexible that can access corners
            if rootstate.rings_to_place in (1,3):
                where = np.argmax([self.Mobility(rootstate,c) for c in C_ACCESS])
                return {'type':'place ring','place pos':C_ACCESS[where]}
        return random.choice(actions)

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
            for st in re.finditer(f'{counter}*{opponent_counter}{counter}*',p_str):
                start = st.start()
                end = st.end()
                # No less than 4 counters in the chain, and regardless how many, only 5 valuable
                nc = min(end-start,5)
                if nc > 4:
                    oc_loc = p[start+st.group().index(opponent_counter)]
                    # If it's flippable, it's a potential
                    line = ('v','d') if p[0][0]==p[-1][0] else ('h','d') if p[0][1]==p[-1][1] else ('v','h')
                    if not self.Cornered(s,oc_loc,line):
                        tot_mark += pts[nc-1+opponent_perspective]
        return tot_mark

    # Take a list of actions and an initial state, and perform alpha-beta search to find the best action.
    # Return the first action that leads to reward, if any was found.
    def SelectAction(self, actions, rootstate):
        if rootstate.rings_to_place > 0: 
            return self.SelectRingPlacementAction(actions, rootstate)
        
        self.start_time = time.time()
        self.stop = False
        self.explored = 0

        rootstate = YinshState.from_other(rootstate)

        # print("Player:", player, "id:", self.id)

        #print("hello from alphabeta")
        print("alphabeta has", len(actions), "actions to consider")

        def loop(depths):
             # Loop through all the actions.
            best_action, best_value = None, -inf
            # Max_depth: from 1 to 9 inclusive
            for max_depth in depths:
                for action in actions:
                    succ = deepcopy(rootstate)
                    # Generate successor state given this action and calculate its value
                    state: YinshState = self.game_rule.generateSuccessor(succ, action, self.id)
                    # Update here to try different nega
                    # Negascout: take the successor state, player view (id=0 => 1, id=1 => -1), and max_depth
                    #value = self.Negascout(state, player, max_depth)
                    value = self.Negascout(state, 1, max_depth)
                    # New best action? Update the best action and value.
                    if value > best_value:
                        best_value = value
                        best_action = action
                    # If the time is up, return the best action.
                    if self.stop or time.time() - self.start_time > THINKTIME:
                        return best_action, best_value, max_depth
            return best_action, best_value, max_depth


        with cProfile.Profile() as pr:
            best_action, best_value, max_depth = loop(range(1, 10))

        pr.print_stats(sort='cumtime')
        print("final value:", best_value, "explored", self.explored, 
        "stopped at depth", max_depth, "time", time.time() - self.start_time)

        # If no reward was found in the time limit, return a random action.
        return best_action
        

    def Mobility(self, s: GameState, loc: tuple, check_empty: bool = True):
        '''
        Check the mobility of a position. That is, sum of how far one can move in 3 axes
        '''
        if wall(loc): return -2
        if check_empty and s.board[loc] != EMPTY: return -1
        m_score = 0
        for dim in ['v','h','d']:
            line=self.game_rule.positionsOnLine(loc,dim)
            idx = line.index(loc)
            p_str  = ('').join([str(s.board[i]) for i in line])
            p_str = p_str[:idx] + 'x' + p_str[idx + 1:]
            for st in re.findall(f'[{EMPTY}x]*',p_str):
                if st.count('x'): 
                    m_score += len(st)
                    continue
        return m_score
    
    def RingPlacement(self, s: GameState, l1: tuple, l2: tuple, far_thresh = 6):
        '''
        Given 2 opponent ring locations, find the best place to place your ring.
        This can either be:
        - The intersections (if rings are near)
        - In-between (if rings on same line)
        - Nothing (if rings too far away, might better off do them separately)
        '''
        # On same line: set the ring between them
        if linear(l1,l2) and (abs(l1[0]-l2[0]),abs(l1[1]-l2[1])) not in ADJACENT:
            potentials = self.game_rule.positionsPassed(l1,l2)
            return potentials if any([s.board[l]==EMPTY for l in potentials]) else []
        ics = []
        # On different lines and nearby: put at their intersections
        for o1 in ['v','h','d']:
            for o2 in ['v','h','d']:
                line1 = self.game_rule.positionsOnLine(l1,o1)
                line2 = self.game_rule.positionsOnLine(l2,o2)
                try:
                    i = (set(line1) & set(line2)).pop()
                except:
                    continue
                # Get Manhattan distance between 2 rings
                md = len(self.game_rule.positionsPassed(l1,i)) + len(self.game_rule.positionsPassed(i,l2))
                # If they're too far away or ring's already there, putting at intersection would be meaningless
                if md<far_thresh and s.board[i] == EMPTY: ics.append(i)
        return ics

    def CornerHeuristic(self, s: GameState):
        """
        This heuristic is the difference between unflippable markers on 2 sides
        """
        h = 0
        cnt = CNTR_1 if self.id else CNTR_0
        for loc in lookup_pos(CNTR_0,s.board) + lookup_pos(CNTR_1,s.board):
            if self.Cornered(s,loc):
                h += 1 if s.board[loc] == cnt else -1
        return h

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