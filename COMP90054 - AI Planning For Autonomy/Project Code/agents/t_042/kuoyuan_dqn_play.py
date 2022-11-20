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
        self.model.load_weights('main_model_dqn_replay.h5')
        self.action_to_index_dict, _ = self.initialize_action_dict()
        
    def game_end_update(self,state,i):
        pass

    def store_weights(self):
        pass
    
    def SelectAction(self, actions, game_state):
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
        reshaped_state = np.expand_dims(self.reshape_state_for_net(game_state), axis=0)
        action_scores = self.model.predict(reshaped_state)[0]
        highest_score = 0
        for action in actions:
            if time.time() - start > THINKTIME:
                return best_action
            # We don't consider remove in action space, if we can remove a ring, remove it 
            if action['type']== "place, move, remove":
                return action
            if action["type"] == "place ring":
                # Using the CNN to estimate Q value for the action
                score = action_scores[self.action_to_index_dict[action["place pos"]]]
                if score > highest_score:
                    highest_score = score
                    best_action = action
            elif action["type"] == "place and move":
                score = action_scores[self.action_to_index_dict[(action["place pos"][0],action["place pos"][1],action["move pos"][0],action["move pos"][1])]]
                if score > highest_score:
                    highest_score = score
                    best_action = action
            elif action["type"] == "pass":
                return action
        return best_action

    def reshape_state_for_net(self, state):
        # Reshape the state to be [11, 11, 4] => 4 dimensions are our ring, our marker, opponent's ring, opponent's marker
        reshaped_state = np.zeros((11, 11, 4),dtype=np.int8)
        assert state.board.shape[0] == state.board.shape[1] == 11
        for i in range (0, 11):
            for j in range (0, 11):
                # If the pos is illegal, update all layers to be -1
                if state.board[i][j] == 5:
                    for z in range (0, 4):
                        reshaped_state[i][j][z] = -1
                elif state.board[i][j] == 1: # RING_0
                    # If the pos is our ring, update layer 0 to be 1
                    if self.id == 0:
                        reshaped_state[i][j][0] = 1
                    # If the pos is opponent's ring, update layer 2 to be 1
                    else:
                        reshaped_state[i][j][2] = 1
                elif state.board[i][j] == 2: # CNTR_0
                    # If the pos is our marker, update layer 1 to be 1
                    if self.id == 0:
                        reshaped_state[i][j][1] = 1
                    # If the pos is opponent's marker, update layer 3 to be 1
                    else:
                        reshaped_state[i][j][3] = 1
                elif state.board[i][j] == 3: # RING_1
                    # If the pos is our ring, update layer 0 to be 1
                    if self.id == 1:
                        reshaped_state[i][j][0] = 1
                    # If the pos is opponent's ring, update layer 2 to be 1
                    else:
                        reshaped_state[i][j][2] = 1
                elif state.board[i][j] == 4: # CNTR_1
                    # If the pos is our marker, update layer 1 to be 1
                    if self.id == 1:
                        reshaped_state[i][j][1] = 1
                    # If the pos is opponent's marker, update layer 3 to be 1
                    else:
                        reshaped_state[i][j][3] = 1
        return reshaped_state

    def initialize_model(self):
        MODEL_LEARNING_RATE = 0.01
        model = keras.Sequential()
        model.add(keras.Input(shape=(11, 11, 4)))
        model.add(keras.layers.Conv2D(filters=64,kernel_size = 3,activation = 'relu'))
        model.add(keras.layers.Conv2D(filters=128,kernel_size = 3,activation = 'relu'))
        model.add(keras.layers.Conv2D(filters=256,kernel_size = 3,activation = 'relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=1933,activation = 'softmax'))
        model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.RMSprop(learning_rate=MODEL_LEARNING_RATE), metrics=['accuracy'])
        return model
    
    def initialize_action_dict(self):
        board = np.zeros((11,11), dtype=np.int8)
        for pos in ILLEGAL_POS:
            board[pos] = ILLEGAL
        # A dictionary that map an action into an index in NN output
        action_to_index_dict = {}
        # A dictionary that map an index in NN output into an action
        index_to_action_dict = {}
        count = 0
        legal_pos = []
        for i in range(11):
            for j in range(11):
                if board[i,j] != ILLEGAL:
                    legal_pos.append((i,j))
                    action_to_index_dict[(i,j)] = count
                    index_to_action_dict[count] = (i,j)
                    count += 1
        assert len(legal_pos) == 85
        for pos in legal_pos:
            for line in ['v', 'h', 'd']:
                if line == 'h':
                    for i in range(11):
                        if board[pos[0], i] != ILLEGAL and i != pos[1]:
                            # From pos[0],pos[1] to pos[0],i
                            action_to_index_dict[(pos[0],pos[1],pos[0],i)] = count
                            index_to_action_dict[count] = (pos[0],pos[1],pos[0],i)
                            count += 1
                elif line == 'v':
                    for i in range(11):
                        if board[i, pos[1]] != ILLEGAL and i != pos[0]:
                            # From pos[0],pos[1] to pos[0],i
                            action_to_index_dict[(pos[0],pos[1],i,pos[1])] = count
                            index_to_action_dict[count] = (pos[0],pos[1],i,pos[1])
                            count += 1
                elif line == 'd':
                    for i in range(-10, 11):
                        if i == 0: continue
                        if (0 <= pos[0]+i <= 10 and 0 <= pos[1]-i <= 10 and board[pos[0]+i, pos[1]-i] != ILLEGAL):
                            action_to_index_dict[(pos[0],pos[1],pos[0]+i,pos[1]-i)] = count
                            index_to_action_dict[count] = (pos[0],pos[1],pos[0]+i,pos[1]-i)
                            count += 1
        return action_to_index_dict, index_to_action_dict