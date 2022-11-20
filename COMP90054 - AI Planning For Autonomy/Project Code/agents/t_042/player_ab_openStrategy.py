# based on example_bfs.py

#import cProfile

from functools import lru_cache
import time
import random
from copy import deepcopy
import numpy as np


################### BETTER GAME RULE #####################
# INFORMATION ------------------------------------------------------------------------------------------------------- #

# Author:  Steven Spratley, extending code by Guang Ho and Michelle Blom
# Date:    07/03/22
# Purpose: Implements "Yinsh" for the COMP90054 competitive game environment

# IMPORTS ------------------------------------------------------------------------------------------------------------#

import re, numpy, time, random
from template import GameState, GameRule

#On the gameboard, Agent 0's rings and counters are denoted by 1s and 2s. 
#Agent 1's rings and counters are 3s and 4s. Illegal positions are 5.
EMPTY    = 0
RING_0   = 1
CNTR_0   = 2
RING_1   = 3
CNTR_1   = 4
ILLEGAL  = 5
NAMES    = {0:"Teal", 1:"Magenta"}

#List of illegal positions.
ILLEGAL_POS = [(0,0),  (0,1),  (0,2),  (0,3),  (0,4),  (0,5),  (0, 10),
               (1,0),  (1,1),  (1,2),  (1,3),
               (2,0),  (2,1),  (2,2),
               (3,0),  (3,1),
               (4,0),
               (5,0),
               (5,10),
               (6,10),
               (7,9),  (7,10),
               (8,8),  (8,9),  (8,10),
               (9,7),  (9,8),  (9,9),  (9,10),
               (10,0), (10,5), (10,6), (10,7), (10,8), (10,9), (10,10)]

# CLASS DEF ----------------------------------------------------------------------------------------------------------#

# Bundle together an agent's activity in the game for use in updating a policy.
class AgentTrace:
    def __init__(self, pid):
        self.id = pid
        self.action_reward = [] # Turn-by-turn history consisting of (action,reward) tuples.
    
    def __deepcopy__(self, memo):
        new_trace = AgentTrace(self.id)
        new_trace.action_reward = self.action_reward.copy()
        return new_trace
    
def ActionToString(agent_id, action):
    if action["type"] == "place ring":
        return f"{NAMES[agent_id]} placed a ring at {action['place pos']}."
    elif action["type"] == "place and move":
        if "sequences" not in action:
            return f"{NAMES[agent_id]} placed a counter at {action['place pos']} and moved to {action['move pos']}."
        else:
            return f"{NAMES[agent_id]} placed at {action['place pos']} and moved to {action['move pos']}, creating a sequence for their opponent."
    elif action["type"] == "place, move, remove":
        return f"{NAMES[agent_id]} placed at {action['place pos']}, moved to {action['move pos']}, formed a sequence and removed ring {action['remove pos']}."
    elif action["type"] == "pass":
        return f"{NAMES[agent_id]} has no counters to play, and passes."
    else:
        return "Unrecognised action."

def AgentToString(agent_id, ps):
    desc = "Agent #{} has scored {} rings thus far.\n".format(agent_id, ps.score)
    return desc

def BoardToString(game_state):
    desc = ""
    return desc

class AgentState:
    def __init__(self, _id):
        self.id     = _id
        self.score  = 0
        self.passed = False
        self.agent_trace = AgentTrace(_id)
        self.last_action = None

    @classmethod
    def from_other(cls, other):
        agent_state = AgentState(other.id)
        agent_state.score = other.score
        agent_state.passed = other.passed
        agent_trace = AgentTrace(other.agent_trace.id)
        agent_trace.action_reward = other.agent_trace.action_reward.copy()
        agent_state.agent_trace = agent_trace
        agent_state.last_action = other.last_action
        return agent_state

    def __deepcopy__(self, memo):
        agent_state = AgentState(self.id)
        agent_state.score = self.score
        agent_state.passed = self.passed
        agent_state.agent_trace = self.agent_trace.__deepcopy__(memo)
        agent_state.last_action = self.last_action
        return agent_state

#Represents game as agents playing on a board.
class YinshState(GameState):           
    def __init__(self, num_agents):
        #Board is stored as a numpy array of ints (the meanings of which are explained in Yinsh.yinsh_utils).
        self.board = numpy.zeros((11,11), dtype=numpy.int8)
        for pos in ILLEGAL_POS:
            self.board[pos] = ILLEGAL
        #Ring_pos stores ring positions for each agent.
        self.ring_pos  = [[], []]
        self.counters_left = 51
        self.rings_to_place = 10
        self.rings_won = [0, 0]
        self.agents = [AgentState(i) for i in range(num_agents)]
        self.agent_to_move = 0

    def __deepcopy__(self, memo):
        num_agents = len(self.agents)
        new_state = YinshState(num_agents)
        new_state.board = self.board.copy()
        new_state.ring_pos = [self.ring_pos[0].copy(), self.ring_pos[1].copy()]
        new_state.counters_left = self.counters_left
        new_state.rings_to_place = self.rings_to_place
        new_state.agents = [self.agents[i].__deepcopy__(memo) for i in range(num_agents)]
        new_state.agent_to_move = self.agent_to_move
        return new_state

    @classmethod
    def from_other(cls, other):
        num_agents = len(other.agents)
        new_state = YinshState(num_agents)
        new_state.board = other.board.copy()
        new_state.ring_pos = [other.ring_pos[0].copy(), other.ring_pos[1].copy()]
        new_state.counters_left = other.counters_left
        new_state.rings_to_place = other.rings_to_place
        new_state.agents = [AgentState.from_other(other.agents[i]) for i in range(num_agents)]
        new_state.agent_to_move = other.agent_to_move
        return new_state

#Regex: Pass through 0 or more empty spaces, followed by 0 or more counters, then one empty space.
movementsAlongLineRegex = re.compile(f"{EMPTY}*[{CNTR_0}{CNTR_1}]*{EMPTY}"+'{1}') 

#Implements game logic.
class YinshGameRule(GameRule):
    def __init__(self,num_of_agent):
        super().__init__(num_of_agent)
        self.private_information = None #Yinsh is a perfect-information game.
        
    #Returns the list of board positions found by drawing a line through a given played position.
    @lru_cache(None)
    def positionsOnLine(self, pos, line):
        x, y = pos
        if line=='h':
            return [(x, i) for i in range(11) if (x, i) not in ILLEGAL_POS]
        elif line=='v':
            return [(i, y) for i in range(11) if (i, y) not in ILLEGAL_POS]
        else: # line == 'd':
            return [(x+i, y-i) for i in range(-10, 11) 
                    if (0 <= x+i <= 10 and 0 <= y-i <= 10) 
                    if (x+i, y-i) not in ILLEGAL_POS]

    #Returns the list of board positions between two play positions.
    @lru_cache(None)
    def positionsPassed(self, start_pos, end_pos):
        line = 'h' if start_pos[0]==end_pos[0] else 'v' if start_pos[1]==end_pos[1] else 'd'
        play_positions = sorted([start_pos, end_pos])
        line_positions = sorted(self.positionsOnLine(start_pos, line))
        idx1,idx2 = line_positions.index(play_positions[0]),line_positions.index(play_positions[1])
        return line_positions[idx1:idx2+1]

    #Flips a counter (if present).
    def flip(self, board, pos):
        board[pos] = EMPTY if board[pos]==EMPTY else CNTR_0 if board[pos]==CNTR_1 else CNTR_1
        
    #Checks the board, given recent changes, for possible new sequences. 
    #Returns a list of sequences, indexed by agent ID, with a maximum 1 sequence per agent.
    # @lru_cache(100)
    def sequenceCheck(self, board, changes):
        sequences = [None, None]
        lines = ['v','h','d']
        def checkLineLC(sequences, line, pos):
            posits = self.positionsOnLine(pos, line) # reminder: cached
            bvs = [board[pos] for pos in posits]
            conseq5 = [(i,a,b,c,d,e) for i,(a,b,c,d,e) in enumerate(zip(bvs,bvs[1:],bvs[2:],bvs[3:],bvs[4:]))
                if 0!=a==b==c==d==e]
            conseq5_0 = [s[0] for s in conseq5 if s[1]==CNTR_0]
            conseq5_1 = [s[0] for s in conseq5 if s[1]==CNTR_1]
            if len(conseq5_0)>0:
                sequences[0] = posits[conseq5_0[0] : conseq5_0[0]+5]
            if len(conseq5_1)>0:
                sequences[1] = posits[conseq5_1[0] : conseq5_1[0]+5]

        def checkLineRE(sequences, line, pos):
            # RE is faster.
            posits = self.positionsOnLine(pos, line)
            cntrs  = ''.join([str(board[pos]) for pos in posits])
            seq0_start,seq1_start = cntrs.find(str(CNTR_0)*5),cntrs.find(str(CNTR_1)*5)
            if not seq0_start==-1:
                sequences[0] = posits[seq0_start : seq0_start+5]
            if not seq1_start==-1:
                sequences[1] = posits[seq1_start : seq1_start+5]

        checkLine = checkLineRE

        #If multiple counters flipped, start by sequence-checking the line that connects them.
        def more_than_one_change(sequences):
            ys,xs  = list(zip(*changes))
            line   = 'v' if len(set(xs))==1 else 'h' if len(set(ys))==1 else 'd'
            lines.remove(line) #Remove this line from lines list to avoid checking it twice.
            checkLine(sequences, line, changes[0])
        
        def do_changes():
            for pos in changes:
                for line in lines:
                    checkLine(sequences, line, pos)

        if len(changes)>1:
            more_than_one_change(sequences)
        do_changes()
        
        return sequences, None
    
    #Returns a list of legal positions to move a ring (at pos) along a given line.
    #First, get all positions along this line (not including ring's current position). Then, for each position, get a 
    #string representing the rings and counters passed. If that string is valid, as checked by a regular expression
    #(i.e. no rings jumped, and only one group of counters jumped), add position to list to return.
    def movementsAlongLine(self, game_state, pos, line):
        valid_movements = []
        positions = self.positionsOnLine(pos, line) 
        
        for p in positions:
            if p == pos:
                continue
            passed = self.positionsPassed(pos, p)
            passed = passed if passed[0]==pos else list(reversed(passed))
            p_str  = ('').join([str(game_state.board[i]) for i in passed[1:]])
            if re.fullmatch(movementsAlongLineRegex, p_str):
                valid_movements.append(p)
        return valid_movements
    
    def ringMoves(self, board, pos):
        pass



    def initialGameState(self):
        return YinshState(self.num_of_agent)
    
    #Takes a game state and and action, and returns the successor state.
    def generateSuccessor(self, state, action, agent_id):
        agent,board = state.agents[agent_id],state.board
        opponent_id = 1 if agent_id==0 else 0
        agent.last_action = action #Record last action such that other agents can make use of this information.
        score = 0
        changes = []
        
        #Check for pass first.
        if action["type"] == "pass":
            agent.passed = True
            return state
        
        #Place and remove symbols on the board representing rings and counters, as necessary.
        ring,cntr = (RING_0,CNTR_0) if agent_id==0 else (RING_1,CNTR_1)
        if action["type"] == "place ring":
            board[action["place pos"]] = ring
            state.ring_pos[agent_id].append(action["place pos"])
            state.rings_to_place -= 1
        elif action["type"] == "place and move":
            board[action["place pos"]]  = cntr
            board[action["move pos"]]   = ring
            state.ring_pos[agent_id].append(action["move pos"])
            state.ring_pos[agent_id].remove(action["place pos"])
            state.counters_left -= 1
        else:
            board[action["place pos"]]  = cntr
            board[action["move pos"]]   = ring
            board[action["remove pos"]] = EMPTY
            for pos in action["sequences"][agent_id]:
                board[pos] = EMPTY
            state.ring_pos[agent_id].append(action["move pos"])
            state.ring_pos[agent_id].remove(action["place pos"])
            state.ring_pos[agent_id].remove(action["remove pos"])
            score += 1
            state.rings_won[agent_id] += 1
            state.counters_left -= 1
            
        #If a ring was moved, flip counters in its path.
        if "move pos" in action:
             [self.flip(board, pos) for pos in self.positionsPassed(action["place pos"],  action["move pos"])[1:-1]]
        
        #If this action sets up a sequence for the opponent, remove it, along with a random ring of theirs.
        opp_id = 0 if agent_id==1 else 1
        if "sequences" in action and action["sequences"][opp_id]:
            for pos in action["sequences"][opp_id]:
                state.board[pos] = EMPTY
            ring_pos = random.choice(state.ring_pos[opp_id])
            state.ring_pos[opp_id].remove(ring_pos)
            state.rings_won[opp_id] += 1
            state.agents[opp_id].score += 1
            board[ring_pos] = EMPTY
            
        #Log this turn's action and any resultant score. Return updated gamestate.
        agent.agent_trace.action_reward.append((action,score))
        agent.score += score
        return state
    
    #Game ends if any agent possesses 3 rings. As a rare edge case, poor playing agents might encounter a game where 
    #none are able to proceed. Game also ends in this case.
    def gameEnds(self):
        deadlock = 0
        for agent in self.current_game_state.agents:
            deadlock += 1 if agent.passed else 0
            if agent.score == 3:
                return True
        return deadlock==len(self.current_game_state.agents)

    #Return final score for this agent.
    def calScore(self, game_state, agent_id):
        return game_state.agents[agent_id].score

    #Return a list of all legal actions available to this agent in this gamestate.
    def getLegalActions(self, game_state, agent_id):
        actions,agent = [],game_state.agents[agent_id]
        
        #A given turn consists of the following:
        #  1. Place a counter of an agent's colour inside one of their rings.
        #  2. Move that ring along a line to an empty space. 
        #     - Any number of empty spaces may be jumped over.
        #     - Any number of counters may be jumped over. Those jumped will be flipped to the opposite colour.
        #     - Once a section of counters is jumped, the ring must take the next available empty space.
        #     - Rings may not be jumped.
        #  3. If a sequence is formed in this process, the game will remove it, along with the agent's chosen ring.
        #     - If multiple sequences are formed, the first one found will be removed (the agent does not decide).
        #     - If an opponent's sequence is formed, it will be removed and credited to their score on their turn.
        
        #Since the gamestate does not change during an agent's turn, all turn parts are able to be planned for at once.
        #Actions will always take the form of one of the following four templates:
        # {'type': 'place ring',          'place pos': (y,x)}
        # {'type': 'place and move',      'place pos': (y,x), 'move pos': (y,x)}
        # {'type': 'place, move, remove', 'place pos': (y,x), 'move pos': (y,x), 'remove pos':(y,x), 'sequences': []}
        # {'type': 'pass'} You cannot voluntarily pass, but there may be scenarios where the agent cannot legally move.
        #Note that the 'sequences' field is a list of sequences, indexed by agent ID. This is to cover cases where an
        #agent creates a sequence for the opponent, or even creates sequences for both players simultaneously.
        #In the case that the agent completes an opponent's sequence, but does not also create their own, the
        #'sequences' field will appear in a 'place and move' action.
        
        if game_state.rings_to_place: #If there are rings still needing to be placed,
            for y in range(11): #For all positions on the board,
                for x in range(11):
                    if game_state.board[(y,x)]==EMPTY: #If this position is empty,
                        actions.append({'type': 'place ring', 'place pos':(y,x)}) #Generate a 'place ring' action.
        
        elif not game_state.counters_left: #Agent has to pass if there are no counters left to play.
            return [{'type':'pass'}]
        
        #For all of the agent's rings, search outwards and ennumerate all possible actions.
        #Check each action for completed sequences. If a sequence can be made, create the action type accordingly.
        else:
            for ring_pos in list(game_state.ring_pos[agent_id]):
                for line in ['v', 'h', 'd']:
                    for pos in self.movementsAlongLine(game_state, ring_pos, line):
                        #Make temporary changes to the board state, for the purpose of sequence checking.
                        changes = []
                        cntr,ring = (CNTR_0,RING_0) if agent_id==0 else (CNTR_1,RING_1)
                        game_state.board[ring_pos] = cntr
                        game_state.board[pos] = ring
                        game_state.ring_pos[agent_id].remove(ring_pos)
                        game_state.ring_pos[agent_id].append(pos)
                        
                        #Append a new action to the actions list, given whether or not a sequence was made.
                        start_pos,end_pos = tuple(sorted([ring_pos, pos]))
                        changes.append(ring_pos)
                        for p in self.positionsPassed(start_pos, end_pos)[1:-1]:
                            self.flip(game_state.board, p)
                            changes.append(p)
                        sequences,details = self.sequenceCheck(game_state.board, changes)
                        if sequences[agent_id]:
                            for r_pos in game_state.ring_pos[agent_id]:
                                actions.append({'type':'place, move, remove', 'place pos':ring_pos, 'move pos':pos, \
                                                'remove pos':r_pos, 'sequences': sequences})
                        elif sequences[0 if agent_id==1 else 1]:
                            for r_pos in game_state.ring_pos[agent_id]:
                                actions.append({'type':'place and move', 'place pos':ring_pos, 'move pos':pos, \
                                                'sequences': sequences})                            
                        else:
                            actions.append({'type': 'place and move', 'place pos':ring_pos, 'move pos':pos})
                        
                        #Remove temporary changes to the board state.
                        [self.flip(game_state.board, p) for p in self.positionsPassed(start_pos, end_pos)[1:-1]]
                        game_state.board[ring_pos] = ring
                        game_state.board[pos] = EMPTY
                        game_state.ring_pos[agent_id].remove(pos)
                        game_state.ring_pos[agent_id].append(ring_pos)
                      
        return actions



###############################################################################################




THINKTIME = 0.85
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
        EMPTY    = 0
        if _id == 0:
            self.marker = 2
            self.oppo_marker = 4
        else:
            self.marker = 4
            self.oppo_marker = 2
        self.game_rule = YinshGameRule(2)
        # More advanced agents might find it useful to not be bound by the functions in GameRule, instead executing
        # their own custom functions under GetActions and DoAction.
        
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
        

    def Heuristic(self, state: YinshState):
        """Calculate heursitic value of given state from white's (player 1) perspective.
        This is ok since Yinsh is a zero-sum game."""
        if state.counters_left == 0:
            return 1000000 * self.RingsHeuristic(state) # someone wins
        return 100000 * self.RingsHeuristic(state) \
            + 10 * self.CountersHeuristic(state) \
            + 1 * self.PositionsConnectedHeuristic(state)
   
    def NegamaxAlphaBetaSearch(self, state: YinshState, turn: 1 | -1, depth, alpha, beta):
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

    # Take a list of actions and an initial state, and perform alpha-beta search to find the best action.
    # Return the first action that leads to reward, if any was found.
    def SelectAction(self, actions, rootstate):
        self.start_time = time.time()
        ## Use Khoi's master starting strategy to place rings
        def adjacentPlacement():
            '''Look at 6 adjacent places surrounding the new ring, 
            then place it where it has most mobility'''
            prev_ring = rootstate.agents[int(not self.id)].last_action['place pos']
            where = np.argmax([self.Mobility(rootstate,(prev_ring[0]+c[0],prev_ring[1]+c[1])) 
                                                for c in ADJACENT])
            act = ADJACENT[where]
            return {'type':'place ring','place pos':(prev_ring[0]+act[0],prev_ring[1]+act[1])}

        if rootstate.rings_to_place > 0: 
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
                    if rootstate.board[5,4] == EMPTY: return {'type': 'place ring','place pos': (5,4)}
                    elif rootstate.board[5,6] == EMPTY: return {'type': 'place ring','place pos': (5,6)}
                    # return adjacentPlacement()
                # Then put a close by one in the corner or can access corners
                if rootstate.rings_to_place in (5,7):
                    candidates = []
                    if rootstate.rings_to_place == 7 and rootstate.board[5][6] != EMPTY and rootstate.board[6][5] != EMPTY:
                        candidates = [(5,6),(6,5),(6,9),(4,1)]
                    else:
                        candidates = [(5,6),(6,5)]
                    if rootstate.rings_to_place == 5 and rootstate.board[5][6] != EMPTY and rootstate.board[6][5] != EMPTY and rootstate.board[6][9] != EMPTY and rootstate.board[4][1] != EMPTY:
                        candidates = [(5,6),(6,5),(6,9),(4,1),(2,5),(9,6)]
                    else:
                        candidates = [(5,6),(6,5),(6,9),(4,1)]
                    where = np.argmax([self.Mobility(rootstate,c) for c in candidates])
                    return {'type':'place ring','place pos':candidates[where]}
                # Defence at end: line ppponent ring intersections if they are close, else next to most recent ring
                if rootstate.rings_to_place in (1,3): 
                    places = self.RingPlacement(rootstate,*rootstate.ring_pos[int(not self.id)][-1:-3:-1])
                    if not len(places): return adjacentPlacement()
                    where = np.argmax([self.Mobility(rootstate,c) for c in places])
                    return {'type':'place ring','place pos':places[where]}
                
            return random.choice(actions)
        
        self.stop = False
        self.explored = 0

        rootstate = YinshState.from_other(rootstate)
        # Initialize player view: max (1)
        player = 1# if self.id == 0 else -1
        #depths = range(1, 10)
        #global next_best_action
        #for max_depth in depths:
            #value = self.NegamaxAlphaBetaSearch(rootstate, player, max_depth)


        
        # print("Player:", player, "id:", self.id)

        #print("hello from alphabeta")
        # value = self.Heuristic(rootstate)
        #print("current state evaluation:",
            #value,
            #100000 * self.RingsHeuristic(rootstate),
            #10 * self.CountersHeuristic(rootstate),
            #1 * self.PositionsConnectedHeuristic(rootstate)
        #)

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
                    alpha=-inf
                    beta=inf
                    value = -self.NegamaxAlphaBetaSearch(state, -player, max_depth, -beta, -alpha)
                    # print(player, value)
                    # New best action? Update the best action and value.
                    if value > best_value:
                        best_value = value
                        best_action = action
                    # If the time is up, return the best action.
                    if self.stop or time.time() - self.start_time > THINKTIME:
                        return best_action, best_value, max_depth
            return best_action, best_value, max_depth


        # with cProfile.Profile() as pr:
        best_action, best_value, max_depth = loop(range(1, 10))

        #pr.print_stats(sort='cumtime')
        #print("final value:", best_value, "explored", self.explored, 
        #"stopped at depth", max_depth, "time", time.time() - self.start_time)

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


# Return the list of all locations of a specific type of grid on the board
lookup_pos = lambda n,b : list(zip(*np.where(b==n)))

# Returns true if 2 locations lie on the same line
linear = lambda l1,l2: l1[0] == l2[0] or l1[1] == l2[1] or sum(l1)==sum(l2)

# Check if location of interest is not in the board or illegal
out_of_board = lambda x,y: x<0 or x>10 or y<0 or y>10
wall = lambda l: out_of_board(*l) or l in ILLEGAL_POS