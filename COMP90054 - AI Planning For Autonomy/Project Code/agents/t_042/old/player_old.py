from typing import Iterable
from template import Agent
import random

import time, random, math
import numpy as np
from Yinsh.yinsh_model import * 
from Yinsh.yinsh_utils import *
from copy import deepcopy
from collections import deque

THINKTIME = 0.9

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
        if s.rings_to_place > 0: return random.choice(actions)
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

    def Mobility(self, s: GameState, 
                       loc: tuple, 
                       opponent_perspective: int = 0):
        '''
        Check the mobility of a position.
        '''
        # What color are we looking at?
        view = self.id ^ opponent_perspective
        m_score = 0
        #for r in s.ring_pos[view]:
        for dim in ['v','h','d']:
            line=self.game_rule.positionsOnLine(loc,dim)
            idx = line.index(loc)
            r_idx = []
            for r in s.ring_pos[0] + s.ring_pos[1]:
                try:
                    ri = line.index(r)
                except:
                    continue
                r_idx.append(ri)
            if len(r_idx) == 1: m_score += r_idx[0] if r_idx[0]<idx else len(line)-r_idx[0]
            elif len(r_idx) == 0: m_score += len(line)
            else:
                pass
        pass

# Return the list of all locations of a specific type of grid on the board
lookup_pos = lambda n,b : list(zip(*np.where(b==n)))