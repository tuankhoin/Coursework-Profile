# INFORMATION ------------------------------------------------------------------------------------------------------- #

# Author:  Steven Spratley, extending code by Guang Ho and Michelle Blom
# Date:    07/03/22
# Purpose: Implements "Yinsh" for the COMP90054 competitive game environment

# IMPORTS ------------------------------------------------------------------------------------------------------------#


from functools import lru_cache
import re, numpy, time, random
from   template import GameState, GameRule

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

def idx_2d_to_1d(idx):
    return idx[0] * 10 + idx[1]

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

CNTR_0_SEQ = str(CNTR_0)*5
CNTR_1_SEQ = str(CNTR_1)*5

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
        pos1, pos2 = sorted([start_pos, end_pos])
        line_positions = self.positionsOnLine(start_pos, line)
        idx1,idx2 = line_positions.index(pos1),line_positions.index(pos2)
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

        board_list = board.tolist()

        def checkLineLC(sequences, line, pos):
            posits = self.positionsOnLine(pos, line) # reminder: cached
            bvs = [board_list[i][j] for i,j in posits]
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
            cntrs  = ''.join([str(board_list[i][j]) for i,j in posits])
            seq0_start,seq1_start = cntrs.find(CNTR_0_SEQ),cntrs.find(CNTR_1_SEQ)
            if not seq0_start==-1:
                sequences[0] = posits[seq0_start : seq0_start+5]
            if not seq1_start==-1:
                sequences[1] = posits[seq1_start : seq1_start+5]

        checkLine = checkLineLC

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
    def getLegalActions(self, game_state, agent_id, sort=True):
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

        if sort: actions.sort(key=action_sort)    
        return actions


# sort by centrality of the play position
def action_sort(action):
    score = 0
    if action['type'] == 'place ring':
        score = 0
    elif action['type'] == 'place and move':
        score = 10000
    elif action['type'] == 'place, move, remove':
        score = 0  # we would like to move and remove if possible
    elif action['type'] == 'pass':
        score = 999999999

    pos = action['place pos']
    # find manhattan distance to (5,5) -> closer to the centre is better!
    score += abs(pos[0] - 5) + abs(pos[1] - 5)
    return score

# END FILE -----------------------------------------------------------------------------------------------------------#
