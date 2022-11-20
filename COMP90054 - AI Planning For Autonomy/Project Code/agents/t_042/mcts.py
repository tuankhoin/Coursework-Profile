from cmath import exp
from typing import Iterable, List, Literal
from template import Agent
import random
import itertools
import time, random, math
import numpy as np
from copy import deepcopy
from collections import defaultdict
import cProfile

from agents.t_042.better_gamerule import * 
from agents.t_042.better_gamerule import YinshGameRule 

THINKTIME = 0.95

class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.game_rule = YinshGameRule(2) 
    
    def SelectAction(self,actions,game_state):
        root_node = Node(self, game_state)
        mcts = MCTS(root_node)
        return mcts.search()

expanded_nodes = 0

class Node:
    # TODO: A json that stores previous reward result from simulations
    def __init__(self, a: myAgent, 
                       s: GameState,
                       parent=None, 
                       action=None):
        self.agent = a
        # Pointers pointing to the parent and child nodes
        self.parent = parent
        # The expanded children of this node: key: action from this node to the child, value: child node
        self.children = []
        # Times simulated through
        self.visited = 0

        global expanded_nodes
        expanded_nodes += 1

        # Depth
        self.depth = parent.depth+1 if parent is not None else 0
        # Current ring situation. Termination node is when another ring won
        self.original = parent.original if parent is not None else s.rings_won
        # Whose turn is it to go? (Flip parent's turn if parent is not None)
        # Otherwise, it's the first turn (ours)
        self.turn = int(not bool(parent.turn)) if parent is not None else a.id
        # The view, initially is 1 (root, our own), then flip each turn
        self.view = -1 * parent.view if parent is not None else 1
        self.state = s
        # Node state's feasible actions
        self.actions = a.game_rule.getLegalActions(self.state, self.turn)
        self.actions_unexplored: List = list(range(len(self.actions)))
        # 1 if we won a ring, -1 if opponent won a ring
        self.value = 0
        # The action that generated this node (intially none)
        self.action = action


    def select(self):
        # If the current node is not fully expanded, expand it
        if not self.isFullyExpanded():
            return self
        else:
            # Select the child using UCT
            # now, don't break tie randomly
            max_node = max(self.children, key=lambda n: n.UCT)
            return max_node

            # Tie break randomly
            max_value = max(self.children, key=lambda n: n.UCT).UCT
            max_nodes = [n for n in self.children if n.UCT == max_value]
            node = random.choice(max_nodes)
            return node.select()
        # Selecting a non-visited child node, or unexpanded node
        #while len(node.child):
            #children = node.child.values()
            # Select child to go by UCT value
            #max_value = max(children, key=lambda n: n.UCT).UCT
            #max_nodes = [n for n in children if n.UCT == max_value]
            # Tie break randomly
            #node = random.choice(max_nodes)
            # If we found the non-visited child, return it
            #if not node.visited: return node
        # If unexpanded, expand and return its result
        # return self #node.expand()

    def expand(self):
        if not self.is_terminal:
            # Randomly choose an unexpanded action to expand (TODO: use your greedy, but this could loss the advantage of MCTS which is fast in each run)
            # TODO: weight the actions by earliest in list
            action_idx = random.randint(0, len(self.actions_unexplored)-1)
            self.actions_unexplored.pop(action_idx)
            action = self.actions[action_idx]
            newChild = Node(self.agent,
                            self.agent.game_rule.generateSuccessor(deepcopy(self.state),action,self.turn),
                            self,
                            action)
            self.children.append(newChild)
            return newChild
            # Open up all its children and return one child
            #for a in self.actions:
                #self.children[a] = Node(self.agent,
                                        #self.agent.game_rule.generateSuccessor(deepcopy(self.state),a,self.turn),
                                        #self,
                                        #a)
            # Choose the action with best heuristic to expand
            # TODO: Rather than just one to expand, define the whole expansion order by heuristic
            # What about min side?
            #a = self.agent.Greedy(self.actions,self.state) * self.view
            # Return a child to expand
            #return self.children[a]
        
        # This is a terminal node, return itself
        return self

    def back_propagate(self, reward, child, discount=0.97):
        """ Backpropagate the reward back to the parent node """
        while child is not None:
            child.visited += 1
            child.value = child.value + ((reward - child.value)/child.visited) # * discount**self.depth
            child = child.parent
            # TODO: Store the rewards to a json file or something, and read/update it back in every time

    # def get_visits(self):
    #     return Node.visits[self.state]
    
    def isFullyExpanded(self):
        return len(self.actions_unexplored) == 0

    @property
    def UCT(self, explr: float=0.5):
        if not self.visited:
            return 0 if not explr else math.inf
        else:
            side = -1 if self.agent.id else 1
            return side*self.value/self.visited + explr*math.sqrt(2*math.log(self.parent.visited)/self.visited)

    @property
    def is_terminal(self):
        return self.state.rings_won[0] - self.original[0] > 0\
            or self.state.rings_won[1] - self.original[1] > 0


class MCTS:
    def __init__(self, root_node: Node, discount = 0.97):
        self.discount = discount
        self.root = root_node

    def search(self, timeout=0.9):
        """
        Execute the MCTS algorithm from the initial state given, with timeout in seconds
        """
        global expanded_nodes
        expanded_nodes = 0

        start_time = time.time()
        time_limit = start_time + timeout
        current_time = time.time()
        # TODO: After a ring is won, update node's original state to that to continue on searching

        with cProfile.Profile() as pr:

            while current_time < time_limit:
                # Find a state node to expand
                # Initially: a child with best greedy score
                selected_node = self.root.select()
                if not selected_node.is_terminal:
                    # Expand the selected child
                    child = selected_node.expand()
                    reward = self.simulate(child)
                    selected_node.back_propagate(reward, child, self.discount)
                current_time = time.time()
            # Pick best move - the most visited children
            max_value = max(self.root.children, key=lambda n: n.visited).visited
            max_nodes = [n for n in self.root.children if n.visited == max_value]
            # Tie break randomly
            bestchild = random.choice(max_nodes)
        
        print("picked a move in %.2f seconds" % (time.time() - start_time), "expanded", expanded_nodes, "nodes")
        
        pr.print_stats(sort='time')

        return bestchild.action

    def simulate(self, node: Node, verbose: int = 0):
        """ Simulate until a terminal state, which returns a corresponding reward
        
        Returns
        ---
        - 0: Draw
        - 1: Teal wins a ring
        - -1: Magenta wins a ring
        """
        state = deepcopy(node.state)
        turn = node.turn
        while (state.counters_left
               and not state.rings_won[0]-node.original[0] 
               and not state.rings_won[1]-node.original[1]):
            # Choose a random action to execute
            action = random.choice(node.agent.game_rule.getLegalActions(state,turn))
            if verbose: print(f'{"Magenta" if turn else "Teal"}: {action}')
            # action = node.agent.SelectAction(node.actions,state)

            # Execute the action
            node.agent.game_rule.generateSuccessor(state,action,turn)
            if verbose>1: print(state)

            # Switch side on next turn
            turn = 1 if not turn else 0
        # Return reward when terminal state is reached
        return 0 if not state.counters_left else \
               1 if state.rings_won[node.agent.id]-node.original[node.agent.id] \
               else -1


