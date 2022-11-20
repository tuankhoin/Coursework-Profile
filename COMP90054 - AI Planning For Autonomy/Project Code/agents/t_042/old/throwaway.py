"""I feel bad to throw away something I spents hours writing, so I put them here as a memory"""

from typing import Iterable
from template import Agent
import random

import time, random, math
import numpy as np
from Yinsh.yinsh_model import * 
from Yinsh.yinsh_utils import *
from copy import deepcopy
from collections import defaultdict, deque

THINKTIME = 0.9
ADJACENT = ((0,1),(0, -1),
            (1,0),(-1 ,0),
            (1,1),(-1,-1))
CORNERS = ((6,0),(9,0),(10,1),(10,4),
           (9,6),(6,9),(4,10),(1,10),
           (0,9),(0,6),(4,1),(1,4))

class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.game_rule = YinshGameRule(2) 

    # Generates actions from this state.
    def GetActions(self, state: GameState):
        '''
        Get sequence of feasible actions from given state.
        '''
        return self.game_rule.getLegalActions(state, self.id)
    
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
        best_action = actions[0]
        best_h = -999999
        if s.rings_to_place > 0: 
            # Strategy will change depends on the assigned color
            if not self.id:
            # White: put 1st 3 rings to flexible positions, then control black
                # Empty board: Place ring next to middle. It open ways to corners
                if s.rings_to_place == 10: return {'type': 'place ring','place pos': (5,4)}
                # Then put an adjacent one that maintains 6 DoF (opposite rhombus heads)
                if s.rings_to_place in (6,8):
                    rhombus = [(5,6),(6,5)] + ([(6,9),(4,1)] if s.rings_to_place==6 else [])
                    where = np.argmax([self.Mobility(s,c) for c in rhombus])
                    return {'type':'place ring','place pos':rhombus[where]}
                if s.rings_to_place == 4: pass # put somewhere near a opponent ring concentrate
                if s.rings_to_place == 2: pass
            else:
            # Black: 1st 3 rings try to block white paths, last 2 try to create clear space
                def adjacentPlacement():
                    '''Look at 6 adjacent places surrounding the new ring, 
                    then place it where it has most mobility'''
                    prev_ring = s.agents[int(not self.id)].last_action['place pos']
                    where = np.argmax([self.Mobility(s,(prev_ring[0]+c[0],prev_ring[1]+c[1])) 
                                                        for c in ADJACENT])
                    act = ADJACENT[where]
                    return {'type':'place ring','place pos':(prev_ring[0]+act[0],prev_ring[1]+act[1])}
                # First ring try to put next to opponent's
                if s.rings_to_place == 9:
                    return adjacentPlacement()
                # Next: line intersection if they are close, else next to most recent ring
                if s.rings_to_place in (5,7):
                    places = self.RingPlacement(*s.ring_pos[int(not self.id)][-1:-3:-1])
                    if not len(places): return adjacentPlacement()
                    where = np.argmax([self.Mobility(s,c) for c in places])
                    return {'type':'place ring','place pos':places[where]}
                # Get a good spot: put to a flexible corner
                if s.rings_to_place == 3:
                    where = np.argmax([self.Mobility(s,c) for c in CORNERS])
                    return {'type':'place ring','place pos':CORNERS[where]}
                if s.rings_to_place == 1: pass # Divide 6 corner regions, choose empty place
            return random.choice(actions)
        for a in actions:
            ns = deepcopy(s)
            self.game_rule.generateSuccessor(ns,a,self.id)
            h = 100*self.RingHeuristic(ns)\
                + 5*(self.ChainHeuristic(ns,0) - self.ChainHeuristic(ns,1))
            if h>best_h:
                best_h = h
                best_action = a
            # In case time not allowing, get the current best one ready
            if time.time() - start > THINKTIME:
                return best_action
        return best_action

    def RingHeuristic(self,s: GameState):
        '''
        Heuristic that compares number of rings won between self and opponent.
        '''
        return -99999 if s.rings_won[int(not self.id)]==3 \
                      else s.rings_won[self.id] - s.rings_won[int(not self.id)]

    def ChainHeuristic(self,s: GameState,
                            opponent_perspective: int = 0,
                            pts: dict = {0:0, 1:0, 2:1, 3:3, 4:5, 5:6}):
        '''
        Heuristic that returns score based on the feasible, unblocked sequences of markers.

        params
        ---
        - s: Game state
        - opponent_perspective: Whose view are we looking at? Our view (0) or opponent's view (1)?
        - pts: A dictionary mapping chains to corresponding point
        '''
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
            for st in re.findall(f'[{ring}{counter}]*',p_str):
                # How many rings needed to move to get 5-marker streak?
                if len(st)>4: tot_mark+=pts[5-st.count(ring)]
            # Incomplete but feasible EMPTY-R-nM (not summing up to 5 yet)
            for st in re.findall(f'{str(EMPTY)*3}{ring}{counter*1}'
                              + f'|{str(EMPTY)*2}{ring}{counter*2}'
                              + f'|{str(EMPTY)*1}{ring}{counter*3}',p_str):
                tot_mark+=pts[st.count(counter)]
            # Incomplete but feasible nM-R-EMPTY (not summing up to 5 yet)
            for st in re.findall(f'{counter*1}{ring}{str(EMPTY)*3}'
                              + f'|{counter*2}{ring}{str(EMPTY)*2}'
                              + f'|{counter*3}{ring}{str(EMPTY)*1}',p_str):
                tot_mark+=pts[st.count(counter)]
            # R-nM_opponent-EMPTY
            for st in re.findall(f'{ring}{opponent_counter*1}{str(EMPTY)*3}'
                              + f'|{ring}{opponent_counter*2}{str(EMPTY)*2}'
                              + f'|{ring}{opponent_counter*3}{str(EMPTY)*1}'
                              # Needs at least a space to land after jumping over
                              + f'|{ring}{opponent_counter*4}{str(EMPTY)*1}',p_str):
                tot_mark+=pts[st.count(counter)]
            # EMPTY-nM_opponent-R
            for st in re.findall(f'{str(EMPTY)*3}{opponent_counter*1}{ring}'
                              + f'|{str(EMPTY)*2}{opponent_counter*2}{ring}'
                              + f'|{str(EMPTY)*1}{opponent_counter*3}{ring}'
                              # Needs at least a space to land after jumping over
                              + f'|{str(EMPTY)*1}{opponent_counter*4}{ring}',p_str):
                tot_mark+=pts[st.count(counter)]
        return tot_mark

    def Mobility(self, s: GameState, loc: tuple, check_empty: bool = True):
        '''
        Check the mobility of a position. That is, sum of how far one can move in 3 axes
        '''
        m_score = 0
        if check_empty and s.board[loc] != EMPTY: return -1
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

    def RingPlacement(self, l1:tuple, l2:tuple, far_thresh = 6):
        '''
        Given 2 opponent ring locations, find the best place to place your ring.
        This can either be:
        - The intersections (if rings are near)
        - In-between (if rings on same line)
        - Nothing (if rings too far away, might better off do them separately)
        '''
        # On same line: set the ring between them
        if linear(l1,l2) and (abs(l1[0]-l2[0]),abs(l1[1]-l2[1])) not in ADJACENT:
            return self.game_rule.positionsPassed(l1,l2)
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
                # If they're too far away, putting at intersection would be meaningless
                if md<far_thresh: ics.append(i)
        return ics

    def CornerHeuristic(self, s: GameState):
        checked = {}
        corners = []
        h = 0
        cnt = CNTR_1 if self.id else CNTR_0
        for loc in lookup_pos(CNTR_0,s.board) + lookup_pos(CNTR_1,s.board):
            if self.Cornered(s,loc,checked):
                h += 1 if s.board[loc] == cnt else -1
                corners.append(loc)
        return h,corners

    def Cornered(self, s: GameState, 
                       loc: tuple, 
                       already_checked: dict = defaultdict(lambda:'checking'),
                       waitlist: defaultdict = defaultdict(lambda:[])):
        if loc in already_checked.keys(): return already_checked[loc]
        if loc in CORNERS: 
            already_checked[loc] = 'corner'
            return True
        clear = lambda x,y: False if out_of_board(x,y) else s.board[x,y]==EMPTY
        side_check = [clear(loc[0]+1,loc[1]) and clear(loc[0]-1,loc[1]),
                      clear(loc[0],loc[1]+1) and clear(loc[0],loc[1]-1),
                      clear(loc[0]+1,loc[1]-1) and clear(loc[0]-1,loc[1]+1)]
        if any(side_check): 
            already_checked[loc] = 'no'
            return False
        waitlist[loc] = 'checking'
        block = lambda x,y: wall((x,y)) or (s.board[x,y] in (CNTR_0,CNTR_1,RING_0,RING_1) and self.Cornered(s,(x,y),already_checked))
        if all([block(loc[0]+1,loc[1]) or block(loc[0]-1,loc[1]),
                block(loc[0],loc[1]+1) or block(loc[0],loc[1]-1),
                block(loc[0]+1,loc[1]-1) or block(loc[0]-1,loc[1]+1)]):
            already_checked[loc] = 'corner'
            return True
        else:
            already_checked[loc] = 'no'
            return False

    def Cornered2(self, s: GameState, loc: tuple):
        for dim in ['v','h','d']:
            line=self.game_rule.positionsOnLine(loc,dim)
            idx = line.index(loc)
            p_str  = ('').join([str(s.board[i]) for i in line])
            p_str = p_str[:idx] + 'x' + p_str[idx + 1:]
            if len(re.findall(f'{EMPTY}[{CNTR_0}{CNTR_1}{RING_0}{RING_1}]*x[{CNTR_0}{CNTR_1}{RING_0}{RING_1}]*{EMPTY}')):
                return False
        return True

# Return the list of all locations of a specific type of grid on the board
lookup_pos = lambda n,b : list(zip(*np.where(b==n)))

# Returns true if 2 locations lie on the same line
linear = lambda l1,l2: l1[0] == l2[0] or l1[1] == l2[1] or sum(l1)==sum(l2)

out_of_board = lambda x,y: x<0 or x>10 or y<0 or y>10
wall = lambda l: out_of_board(*l) or l in ILLEGAL_POS