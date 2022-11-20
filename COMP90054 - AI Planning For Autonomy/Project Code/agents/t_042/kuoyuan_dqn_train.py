import os
import re
from statistics import mode
from turtle import shape
from template import Agent
import random
from Yinsh.yinsh_model import YinshGameRule 
from copy import deepcopy
from collections import deque
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import TensorBoard
import numpy as np
import time
from Yinsh.yinsh_utils import *
import copy
import pickle


### NOTE ####
# The training needs modified game.py to work.



SMALL_NUMBER = -999999
class myAgent(Agent):
    def __init__(self,_id):
         # Constants 
        self.LEARNING_RATE = 0.01
        self.DISCOUNT_FACTOR = 0.9
        # Use replay memory to store the past states and actions to speed up the learning process
        self.REPLAY_SIZE_MAX = 5000  # How many last steps to keep for model training
        self.MIN_REPLAY_SIZE =  400  # Minimum number of steps in a memory to start training
        self.MINIBATCH_SIZE = 50  # How many steps (samples) to use for training
        self.UPDATE_TARGET_EVERY = 10  # Update target network every n episodes
        self.EPSILON = 1
        self.EPSILON_DECAY = 0.999
        self.EPSILON_MIN = 0.01
        self.AGGREGATE_STATS_EVERY  = 10
        self.win_count = 0
        # Other variables
        self.id = _id
        self.game_rule = YinshGameRule(2)
        self.episode_reward = 0
        self.ep_rewards = []
        self.old_state = None
        self.old_action = None
        self.replay_memory = deque(maxlen=self.REPLAY_SIZE_MAX) #replay_memory
        # Load replay_memory
        try:
            with open('replay_memory_dqn.pkl', 'rb') as f:
                self.replay_memory = pickle.load(f)
        except:
            pass
        
        # Used to update tensorboard
        #self.step = 1
        self.model = self.initialize_model()
        # Target network
        self.target_model = self.initialize_model()
        # Load main model weights
        try:
            self.model.load_weights('main_model_dqn.h5')
        except:
            print("No model found")
        # Set the target network weights
        try:
            self.target_model.load_weights('target_model_dqn.h5')
        except:
            print("No target model found")
            # Set the target network weights to be equal to the main network weights
            self.target_model.set_weights(self.model.get_weights())
        
        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="dqn_logs/{}".format(int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
        # Load target update counter
        try:
            with open('target_update_counter_dqn.pkl', 'rb') as f:
                self.target_update_counter = pickle.load(f)
        except:
            pass
        # Load epsilon 
        try:
            with open('epsilon_dqn.pkl', 'rb') as f:
                self.EPSILON = pickle.load(f)
        except:
            pass
        self.action_to_index_dict, self.index_to_action_dict = self.initialize_action_dict()

    def game_end_update(self,final_state,episode):

        ################ FULL GAME VERSION ##############################
        
        # At the end of game, do another update:
        # Update the replay memory, train the main model, and update the target network weights
        reward = 0
        # If we remove 3 rings, we win
        if final_state.rings_won[self.id] == 3:
            reward = 10
            print("WIN with 3 rings removed")
        # If we remove less rings, we lose
        elif final_state.rings_won[self.id] < final_state.rings_won[abs(self.id-1)]:
            reward = -10
            print("LOSE")
        # If we remove the same number of rings, we tie
        elif final_state.rings_won[self.id] == final_state.rings_won[abs(self.id-1)]:
            reward = 2
            print("Tie")
        # we have less rings
        else:
            reward = 5
            print("Less rings")

        assert self.old_state is not None and self.old_action is not None
        self.replay_memory.append((self.old_state, self.old_action, reward, final_state))
        self.train(game_end = True)  
        # Increase target network update counter by 1 (finish one episode)
        self.target_update_counter += 1
        # If we have reached the target update counter, update the target network weights
        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
        self.episode_reward += reward
        # Append episode reward to a list and log stats (every given number of episodes)
        self.ep_rewards.append(self.episode_reward)
        if episode % self.AGGREGATE_STATS_EVERY != 0 or episode == 1:
            average_reward = sum(self.ep_rewards[-self.AGGREGATE_STATS_EVERY :])/len(self.ep_rewards[-self.AGGREGATE_STATS_EVERY :])
            min_reward = min(self.ep_rewards[-self.AGGREGATE_STATS_EVERY :])
            max_reward = max(self.ep_rewards[-self.AGGREGATE_STATS_EVERY :])
            self.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=self.EPSILON)
        self.episode_reward = 0
        #self.step = 1
        self.old_action = None
        self.old_state = None
        # Decay epsilon
        if self.EPSILON > self.EPSILON_MIN:
            self.EPSILON *= self.EPSILON_DECAY
            self.EPSILON = max(self.EPSILON_MIN, self.EPSILON)

        # print("rewards in memory:",[i[2] for i in self.replay_memory])

    def store_weights(self):
        #print("replay reward history:", [history[2] for history in self.replay_memory])
        #print("epsilon:", self.EPSILON)
        # Save main model weights
        self.model.save_weights('main_model_dqn.h5')
        # Save target model weights
        self.target_model.save_weights('target_model_dqn.h5')
        # Store replay memory
        with open('replay_memory_dqn.pkl', 'wb') as f:
            pickle.dump(self.replay_memory, f)
        # Store epsilon
        with open('epsilon_dqn.pkl', 'wb') as f:
            pickle.dump(self.EPSILON, f)
        # Load target update counter
        with open('target_update_counter_dqn.pkl', 'wb') as f:
            pickle.dump(self.target_update_counter, f)
        print("Win count:", self.win_count)


    def SelectAction(self,actions,game_state):
        if self.old_state is not None and self.old_action is not None:
            reward = self.calculate_reward(self.old_state, game_state)
            self.episode_reward += reward
            # Every step we update replay memory and train main network
            # Adds step's data to a memory replay array
            # (observation state (not flatted), action, reward, new observation state (not flatted) )
            assert self.old_action["type"] != "pass"
            self.replay_memory.append((self.old_state, self.old_action, reward, game_state))
            self.train()
        reshaped_state = np.expand_dims(self.reshape_state_for_net(game_state), axis=0)
        action_scores = self.model.predict(reshaped_state)[0]
        scores = [] # stores the state and the score for this state
        for action in actions:
            if action["type"] == "place ring":
                score = action_scores[self.action_to_index_dict[action["place pos"]]]
                scores.append((action,score))
            elif action["type"] == "place and move" or  action["type"] == "place, move, remove":
                score = action_scores[self.action_to_index_dict[(action["place pos"][0],action["place pos"][1],action["move pos"][0],action["move pos"][1])]]
                scores.append((action,score)) 
            elif action["type"] == "pass":
                return action
        if len(scores) != len(actions):
            print(scores)
            print(actions)
        # Use epsilon greedy to select the action
        this_action = self.epsilon_greedy(scores)
        # Update old_state and old_action
        self.old_state = copy.deepcopy(game_state)
        self.old_action = this_action
        #update the step (for tensorboard) by 1 
        #self.step += 1
        return this_action

        
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
        model = keras.Sequential()
        model.add(keras.Input(shape=(11, 11, 4)))
        model.add(keras.layers.Conv2D(filters=64,kernel_size = 3,activation = 'relu'))
        model.add(keras.layers.Conv2D(filters=128,kernel_size = 3,activation = 'relu'))
        model.add(keras.layers.Conv2D(filters=256,kernel_size = 3,activation = 'relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=1933,activation = 'softmax'))
        model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.LEARNING_RATE), metrics=['accuracy'])
        return model


    def epsilon_greedy(self,scores):
        action_list = [pair[0] for pair in scores]
        score_list = [pair[1] for pair in scores]
        if random.random() < self.EPSILON:
            # Choose a random action
            return random.choice(action_list)
        else:
            # Choose the action with the highest score
            return action_list[score_list.index(max(score_list))]

    # Trains main network every step during episode
    def train(self,game_end=False):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < self.MIN_REPLAY_SIZE:
            return
        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)
        # observation state (not flatted), action, reward, new observation state (not flatted) 
        X = []
        y = []
        for experience in minibatch:
            # Get stored values
            old_state = self.reshape_state_for_net(experience[0])
            old_state_prediction = self.model.predict(np.expand_dims(old_state,axis=0))[0]
            action_index = None
            action = experience[1]
            if action["type"] == "place ring":
                action_index = self.action_to_index_dict[action["place pos"]]
            elif action["type"] == "place and move" or  action["type"] == "place, move, remove":
                action_index = self.action_to_index_dict[(action["place pos"][0],action["place pos"][1],action["move pos"][0],action["move pos"][1])]

            reward = experience[2]
            new_state = np.expand_dims(self.reshape_state_for_net(experience[3]), axis=0) 
            # Get prediction of the new state
            new_state_prediction = self.target_model.predict(new_state)[0]
            # Get the target value
            target_value = None
            if game_end:
                target_value = reward
            else:
                target_value = reward + self.DISCOUNT_FACTOR * np.max(new_state_prediction)
            # Get the target value for the old state
            target_f = old_state_prediction
            target_f[action_index] = target_value
            # Train the network
            X.append(old_state)
            y.append(target_f)
        X = np.array(X)
        y = np.array(y)
        self.model.fit(X, y, epochs=1, verbose=0, callbacks=[self.tensorboard] if game_end else None)
        
    
    def calculate_reward(self, old_state, current_state):
        # Based on previous state and current state, calculate reward
        # If our ring is removed, r = 3
        # If opponent's ring is removed, r = -3
        # If we win, r = 10
        # If opponent wins, r = -10
        # Number of our rings removed
        # return self.potential_func(current_state) - self.potential_func(old_state)  
        ############# FULL GAME #############
        
        reward = 0
        updated = False
        ring_change = len(old_state.ring_pos[self.id]) - len(current_state.ring_pos[self.id])
        # Number of opponent's rings removed
        opp_ring_change = len(old_state.ring_pos[abs(self.id - 1)]) - len(current_state.ring_pos[abs(self.id - 1)])
        if ring_change > 0:
            reward += 3 * ring_change
            updated = True
        
        if opp_ring_change > 0:
            reward -= 3 * opp_ring_change
            updated = True
        
        if updated:
            return reward
        else:
            return self.potential_func(current_state) - self.potential_func(old_state)    
        
    def potential_func(self,state):
        # Calculate potential function for a given state using heuristic by Khoi
        if state.rings_to_place > 0: return 0
        return 10*self.RingHeuristic(state) + 0.5*(self.ChainHeuristic(state,0) - 0.1*self.ChainHeuristic(state,1))

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
    def RingHeuristic(self,s):
        '''
        Heuristic that compares number of rings won between self and opponent.
        '''
        return -10 if s.rings_won[int(not self.id)]==3 else s.rings_won[self.id] - s.rings_won[int(not self.id)]

    def ChainHeuristic(self,s,
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
        # Return the list of all locations of a specific type of grid on the board
        lookup_pos = lambda n,b : list(zip(*np.where(b==n)))
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



# Visualize the learning process
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir
    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        self.model = model
        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter
        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter
        self._should_write_train_graph = False

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()

        