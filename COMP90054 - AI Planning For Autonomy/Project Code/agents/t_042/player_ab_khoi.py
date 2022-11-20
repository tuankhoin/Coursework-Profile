from typing import Iterable
from template import Agent
import random

import time, random, math
import numpy as np
#from Yinsh.yinsh_model import * 
from agents.t_042.better_gamerule import *
from Yinsh.yinsh_utils import *
from copy import deepcopy

THINKTIME = 9#0.95
ADJACENT = ((0,1),(0, -1),
            (1,0),(-1 ,0),
            (1,1),(-1,-1))
CORNERS = ((6,0),(9,0),(10,1),(10,4),
           (9,6),(6,9),(4,10),(1,10),
           (0,9),(0,6),(4,1),(1,4))
C_ACCESS = CORNERS + ((5,4),(5,6),(4,5),(6,5),(4,6),(6,4))

class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.game_rule = YinshGameRule(2) 
        self.rival = int(not _id)

    # Generates actions from this state.
    def GreedyReward(self, state, action):
        return self.Reward(self.game_rule.generateSuccessor(state,action,self.id))
    def GetActions(self, state: GameState):
        '''
        Get sequence of feasible actions from given state.
        '''
        return self.game_rule.getLegalActions(state, self.id, sort=False).sort(key=myAgent.GreedyReward)
    
    def SelectAction(self, actions: Iterable, s: GameState):
        '''
        Return the best determined action, being one of the following types:
        ```
        {'type': 'place ring',          'place pos': (y,x)}
        {'type': 'place and move',      'place pos': (y,x), 'move pos': (y,x)}
        {'type': 'place, move, remove', 'place pos': (y,x), 'move pos': (y,x), 'remove pos':(y,x), 'sequences': []}
        {'type': 'pass'}
        ```

        Params
        ---
        - actions: List of available actions.
        - s: The current game state.
        '''
        start = time.time()
        if s.rings_to_place > 0: return self.RingSearch(s,actions)
        # Greedy search
        #if any([a['type'] == 'place, move, remove' for a in actions]): return self.Greedy(s,actions,start)
        return self.ABP(s,actions,2,start)
        #return self.Greedy(s,actions,start)

    def RingSearch(self, s: GameState, actions=[]):
        '''Perform search on where to put the ring when game starts'''
        def adjacentPlacement():
            '''Look at 6 adjacent places surrounding the new ring, 
            then place it where it has most mobility'''
            prev_ring = s.agents[int(not self.id)].last_action['place pos']
            where = np.argmax([self.Mobility(s,(prev_ring[0]+c[0],prev_ring[1]+c[1])) 
                                                for c in ADJACENT])
            act = ADJACENT[where]
            return {'type':'place ring','place pos':(prev_ring[0]+act[0],prev_ring[1]+act[1])}
        # Strategy will change depends on the assigned color
        if not self.id:
        # White: put 1st 3 rings to flexible positions, then control black
            # Empty board: Place ring next to middle. It open ways to corners
            if s.rings_to_place == 10: return {'type': 'place ring','place pos': (5,4)}
            # Then put a close by one in the corner or can access corners
            if s.rings_to_place in (6,8):
                rhombus = [(5,6),(6,5)] + ([(6,9),(4,1)] if s.rings_to_place==6 else [])
                where = np.argmax([self.Mobility(s,c) for c in rhombus])
                return {'type':'place ring','place pos':rhombus[where]}
            # Defence at end: line ppponent ring intersections if they are close, else next to most recent ring
            if s.rings_to_place in (2,4): 
                places = self.RingPlacement(s,*s.ring_pos[int(not self.id)][-1:-3:-1])
                if not len(places): return adjacentPlacement()
                where = np.argmax([self.Mobility(s,c) for c in places])
                return {'type':'place ring','place pos':places[where]}
        else:
        # Black: 1st 3 rings try to block white paths, last 2 try to create clear space
            # First ring try to put next to opponent's first
            if s.rings_to_place == 9:
                return adjacentPlacement()
            # Next: line intersection if they are close, else next to most recent ring
            if s.rings_to_place in (5,7):
                places = self.RingPlacement(s,*s.ring_pos[int(not self.id)][-1:-3:-1])
                if not len(places): return adjacentPlacement()
                where = np.argmax([self.Mobility(s,c) for c in places])
                return {'type':'place ring','place pos':places[where]}
            # Get a good spot for itself: put to a flexible that can access corners
            if s.rings_to_place in (1,3):
                where = np.argmax([self.Mobility(s,c) for c in C_ACCESS])
                return {'type':'place ring','place pos':C_ACCESS[where]}
        # Fallback just in case
        return random.choice(actions)

    def GetSuccessors(self, state: YinshState, turn: 1|0, a = None):
        """Get successor states from given state."""
        actions = a if a else self.GetActions(state)
        for action in actions:
            successor = deepcopy(state)
            yield self.game_rule.generateSuccessor(successor, action, turn)
    def ABP(self, s: GameState, 
                  actions: Iterable,
                  depth, 
                  start=time.time(),
                  tlimit = THINKTIME):
        best_action = actions[0]
        best_h = -999999
        for a in actions:
            ns = deepcopy(s)
            self.game_rule.generateSuccessor(ns,a,self.id)
            value = math.inf
            for oa in self.game_rule.getLegalActions(ns, int(not self.id)):
                nns = deepcopy(ns)
                self.game_rule.generateSuccessor(nns,oa,int(not self.id))
                h = -self.Reward(nns,1)
                value = min(value, h)
                if value < best_h or time.time() - start > tlimit: break 
            #print(value,a)
            if value>best_h:
                best_h = value
                best_action = a
            # In case time not allowing, get the current best one ready
            if time.time() - start > tlimit: return best_action
        return best_action

    def Greedy(self, s: GameState, actions: Iterable, start=time.time()):
        '''Perform Greedy Search'''
        best_action = actions[0]
        best_h = -999999
        for a in actions:
            ns = deepcopy(s)
            self.game_rule.generateSuccessor(ns,a,self.id)
            h = self.Reward(ns)
            if h>best_h:
                best_h = h
                best_action = a
            # In case time not allowing, get the current best one ready
            if time.time() - start > THINKTIME:
                return best_action
        return best_action

    def Reward(self, ns: GameState, enemy: int = 0):
        '''Calculate the reward of a given new state'''
        return (1000*self.RingHeuristic(ns) + 1.2*self.CornerHeuristic(ns)) * (-1 if enemy else 1)\
              + 2*(self.ChainHeuristic(ns,0,enemy) - self.ChainHeuristic(ns,1,enemy))

    def RingHeuristic(self,s: GameState):
        '''
        Heuristic that compares number of rings won between self and opponent.
        '''
        return -99999 if s.rings_won[int(not self.id)]==3 \
                      else s.rings_won[self.id] - s.rings_won[int(not self.id)]

    def ChainHeuristic(self,s: GameState,
                            opponent_perspective: int = 0,
                            enemy: int = 0,
                            pts: dict = {0:0, 1:0, 2:1, 3:3, 4:5, 5:10}):
        '''
        Heuristic that returns score based on the feasible, unblocked sequences of markers.

        params
        ---
        - s: Game state
        - opponent_perspective: Whose view are we looking at? Our view (0) or opponent's view (1)?
        - enemy: Which side are we running the calculation on behalf?
        - pts: A dictionary mapping chains to corresponding point
        '''
        # What color are we looking at?
        view = (self.rival if enemy else self.id) ^ opponent_perspective
        tot_mark = 0
        ring = str(RING_1 if view else RING_0)
        opponent_ring = str(RING_0 if view else RING_1)
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
            for st in re.findall(f'{ring}{counter*4}'
                                +f'|{counter*1}{ring}{counter*3}'
                                +f'|{counter*2}{ring}{counter*2}'
                                +f'|{counter*3}{ring}{counter*1}'
                                +f'|{counter*4}{ring}'
                                +f'|{ring}{opponent_counter*4}[{counter}{opponent_counter}]*{str(EMPTY)}'
                                +f'|{counter*1}{ring}{opponent_counter*3}[{counter}{opponent_counter}]*{str(EMPTY)}'
                                +f'|{counter*2}{ring}{opponent_counter*2}[{counter}{opponent_counter}]*{str(EMPTY)}'
                                +f'|{counter*3}{ring}{opponent_counter*1}[{counter}{opponent_counter}]*{str(EMPTY)}'
                                +f'|{str(EMPTY)}[{counter}{opponent_counter}]*{opponent_counter*3}{ring}{counter*1}'
                                +f'|{str(EMPTY)}[{counter}{opponent_counter}]*{opponent_counter*2}{ring}{counter*2}'
                                +f'|{str(EMPTY)}[{counter}{opponent_counter}]*{opponent_counter*1}{ring}{counter*3}'
                                +f'|{str(EMPTY)}[{counter}{opponent_counter}]*{opponent_counter*4}{ring}',p_str):
                tot_mark+=pts[4+opponent_perspective]
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
            # Incomplete but feasible R-nM_opponent-EMPTY
            for st in re.findall(f'{ring}{opponent_counter*1}{str(EMPTY)*3}'
                              + f'|{ring}{opponent_counter*2}{str(EMPTY)*2}'
                              + f'|{ring}{opponent_counter*3}{str(EMPTY)*1}',p_str):
                tot_mark+=pts[st.count(opponent_counter)+opponent_perspective]
            # Incomplete but feasible EMPTY-nM_opponent-R
            for st in re.findall(f'{str(EMPTY)*3}{opponent_counter*1}{ring}'
                              + f'|{str(EMPTY)*2}{opponent_counter*2}{ring}'
                              + f'|{str(EMPTY)*1}{opponent_counter*3}{ring}',p_str):
                tot_mark+=pts[st.count(opponent_counter)+opponent_perspective]
            # nM-flippable M_opponent-nM (5 markers)
            for st in re.finditer(f'{counter*0}{opponent_counter}{counter*4}'
                                +f'|{counter*1}{opponent_counter}{counter*3}'
                                +f'|{counter*2}{opponent_counter}{counter*2}'
                                +f'|{counter*3}{opponent_counter}{counter*1}'
                                +f'|{counter*4}{opponent_counter}{counter*0}',p_str):
                start = st.start()
                oc_loc = p[start+st.group().index(opponent_counter)]
                # If it's flippable, it's a potential
                line = ('v','d') if p[0][0]==p[-1][0] else ('h','d') if p[0][1]==p[-1][1] else ('v','h')
                if self.Flippable(s,opponent_ring,oc_loc,line):
                    tot_mark += pts[4+opponent_perspective]
        return tot_mark

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

    def Cornered(self, s: GameState, loc: tuple, dims: Iterable = ['v','h','d']):
        '''
        Check if a position is always unflippable if a marker lies there'''
        for dim in dims:
            line=self.game_rule.positionsOnLine(loc,dim)
            idx = line.index(loc)
            p_str  = ('').join([str(s.board[i]) for i in line])
            p_str = p_str[:idx] + 'x' + p_str[idx + 1:]
            if len(re.findall(f'{EMPTY}[{CNTR_0}{CNTR_1}{RING_0}{RING_1}]*x[{CNTR_0}{CNTR_1}{RING_0}{RING_1}]*{EMPTY}',p_str)):
                return False
        return True
    
    def Flippable(self, s: GameState, ring: str, loc: tuple, dims: Iterable = ['v','h','d']):
        '''
        Check if a position is flippable next turn if a marker lies there'''
        for dim in dims:
            line=self.game_rule.positionsOnLine(loc,dim)
            idx = line.index(loc)
            p_str  = ('').join([str(s.board[i]) for i in line])
            p_str = p_str[:idx] + 'x' + p_str[idx + 1:]
            if len(re.findall(f'{ring}{EMPTY}*[{CNTR_0}{CNTR_1}]*x[{CNTR_0}{CNTR_1}]*{EMPTY}'
                            +f'|{EMPTY}[{CNTR_0}{CNTR_1}]*x[{CNTR_0}{CNTR_1}]*{EMPTY}*{ring}',p_str)):
                return True
        return False

    def Movable(self, s: GameState, loc: tuple, dims: Iterable = ['v','h','d']):
        '''
        Check if a ring position is movable next turn'''
        for dim in dims:
            line=self.game_rule.positionsOnLine(loc,dim)
            idx = line.index(loc)
            p_str  = ('').join([str(s.board[i]) for i in line])
            p_str = p_str[:idx] + 'x' + p_str[idx + 1:]
            if len(re.findall(f'{EMPTY}[{CNTR_0}{CNTR_1}]*x'
                            +f'|x[{CNTR_0}{CNTR_1}]*{EMPTY}',p_str)):
                return True
        return False

# Return the list of all locations of a specific type of grid on the board
lookup_pos = lambda n,b : list(zip(*np.where(b==n)))

# Returns true if 2 locations lie on the same line
linear = lambda l1,l2: l1[0] == l2[0] or l1[1] == l2[1] or sum(l1)==sum(l2)

# Check if location of interest is not in the board or illegal
out_of_board = lambda x,y: x<0 or x>10 or y<0 or y>10
wall = lambda l: out_of_board(*l) or l in ILLEGAL_POS