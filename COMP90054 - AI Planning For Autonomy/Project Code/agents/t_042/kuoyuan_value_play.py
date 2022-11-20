from typing import Iterable
from template import Agent
import random

import time, random, math
import numpy as np
from Yinsh.yinsh_model import * 
from Yinsh.yinsh_utils import *
from copy import deepcopy
from collections import deque
import tensorflow as tf
from tensorflow import keras

THINKTIME = 0.9

class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.game_rule = YinshGameRule(2) 
        self.model = self.initialize_model()
        self.model.load_weights('main_model.h5')
        
    def game_end_update(self,state,i):
        pass

    def store_weights(self):
        pass
    
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
        best_next_v = -999999
        for a in actions:
            ns = deepcopy(s)
            self.game_rule.generateSuccessor(ns,a,self.id)
            next_v = self.model.predict(np.array([self.flatten_state(ns)]))[0][0]
            if next_v>best_next_v:
                best_next_v = next_v
                best_action = a
            # In case time not allowing, get the current best one ready
            #if time.time() - start > THINKTIME:
                #return best_action
        return best_action

    def flatten_state(self, state):
        flatten_state = []
        for i in range (0,state.board.shape[0]):
                for j in range (0,state.board.shape[1]):
                    # Not consider illegal position
                    if state.board[i][j] != 5: 
                        # Update value:
                        # EMPTY = 0
                        # own ring = 1 
                        # own ring = 2 
                        # opponent ring = -1 
                        # opponent ring = -2 
                        # based on Id: if id = 0, 1 = 1, 2 = 2, 3 = -1, 4 = -2
                        # if id = 1, 1 = -1, 2 = -2, 3 = 1, 4 = 2
                        if self.id == 0: # add original values
                            if state.board[i][j] == 1:
                                flatten_state.append(1)
                            elif state.board[i][j] == 2:
                                flatten_state.append(2)
                            elif state.board[i][j] == 3:
                                flatten_state.append(-1)
                            elif state.board[i][j] == 4:
                                flatten_state.append(-2)
                            else: # EMPTY
                                flatten_state.append(0)
                        else: # change as described above
                            if state.board[i][j] == 1:
                                flatten_state.append(-1)
                            elif state.board[i][j] == 2:
                                flatten_state.append(-2)
                            elif state.board[i][j] == 3:
                                flatten_state.append(1)               
                            elif state.board[i][j] == 4:
                                flatten_state.append(2)
                            else: # EMPTY
                                flatten_state.append(0)
        assert len(flatten_state) == 85
        return flatten_state

    def initialize_model(self):
        CELL_NUM = 85
        model_learning_rate = 0.001
        init = tf.keras.initializers.RandomNormal(mean = 0, stddev = 0.1)
        model = keras.Sequential()
        model.add(keras.layers.Dense(85, input_dim=CELL_NUM, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(42, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(1, activation='linear', kernel_initializer=init))
        model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.RMSprop(learning_rate=model_learning_rate), metrics=['accuracy'])
        return model