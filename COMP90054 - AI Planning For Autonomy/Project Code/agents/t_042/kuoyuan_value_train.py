import os
import re
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
        self.MINIBATCH_SIZE = 30  # How many steps (samples) to use for training
        self.UPDATE_TARGET_EVERY = 10  # Update target network every n episodes
        self.EPSILON = 0.9
        self.EPSILON_DECAY = 0.999
        self.EPSILON_MIN = 0.01
        self.AGGREGATE_STATS_EVERY  = 10
        self.win_count = 0
        # Other variables
        self.id = _id
        self.game_rule = YinshGameRule(2)
        self.episode_reward = 0
        self.ep_rewards = [-200]
        self.old_state = None
        self.old_action = None
        self.replay_memory = deque(maxlen=self.REPLAY_SIZE_MAX) #replay_memory
        # Load replay_memory
        try:
            with open('replay_memory.pkl', 'rb') as f:
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
            self.model.load_weights('main_model.h5')
        except:
            print("No model found")
        # Set the target network weights
        try:
            self.target_model.load_weights('target_model.h5')
        except:
            print("No target model found")
            # Set the target network weights to be equal to the main network weights
            self.target_model.set_weights(self.model.get_weights())
        
        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}".format(int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
        # Load target update counter
        try:
            with open('target_update_counter.pkl', 'rb') as f:
                self.target_update_counter = pickle.load(f)
        except:
            pass
        # Load epsilon 
        try:
            with open('epsilon.pkl', 'rb') as f:
                self.EPSILON = pickle.load(f)
        except:
            pass
        
    def game_end_update(self,final_state,episode):

        ################ FULL GAME VERSION ##############################
        """
        # At the end of game, do another update:
        # Update the replay memory, train the main model, and update the target network weights
        reward = 0
        # If we remove 3 rings, we win
        if final_state.rings_won[self.id] == 3:
            reward = 10
            print("WIN with 3 rings removed")
        # If we have more rings, we lose
        elif final_state.rings_won[self.id] < final_state.rings_won[abs(self.id-1)]:
            reward = -10
            print("LOSE")
        elif final_state.rings_won[self.id] == final_state.rings_won[abs(self.id-1)]:
            reward = 0
            print("Tie")
        # we have less rings
        else:
            reward = 5
            print("Less rings")
        """
        reward = 0
        # If we remove the rings, reward = 3
        if final_state.rings_won[self.id] > final_state.rings_won[abs(self.id-1)]:
            reward = 10
            print("WIN")
            self.win_count += 1
        elif final_state.rings_won[self.id] < final_state.rings_won[abs(self.id-1)]:
            reward = -10
            print("LOSE")

        assert self.old_state is not None and self.old_action is not None
        self.replay_memory.append((self.old_state, self.old_action, reward, final_state))
        self.train(game_end = True)  
        # Increase target network update counter by 1 (finish one episode)
        self.target_update_counter += 1
        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
        self.episode_reward += reward
        # Append episode reward to a list and log stats (every given number of episodes)
        self.ep_rewards.append(self.episode_reward)
        if not episode % self.AGGREGATE_STATS_EVERY or episode == 1:
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
        self.model.save_weights('main_model.h5')
        # Save target model weights
        self.target_model.save_weights('target_model.h5')
        # Store replay memory
        with open('replay_memory.pkl', 'wb') as f:
            pickle.dump(self.replay_memory, f)
        # Store epsilon
        with open('epsilon.pkl', 'wb') as f:
            pickle.dump(self.EPSILON, f)
        # Load target update counter
        with open('target_update_counter.pkl', 'wb') as f:
            pickle.dump(self.target_update_counter, f)
        print("Win count:", self.win_count)

    def SelectAction(self,actions,game_state):
        if self.old_state is not None and self.old_action is not None:
            reward = self.calculate_reward(self.old_state, game_state)
            self.episode_reward += reward
            # Every step we update replay memory and train main network
            # Adds step's data to a memory replay array
            # (observation state (not flatted), action, reward, new observation state (not flatted) )
            self.replay_memory.append((self.old_state, self.old_action, reward, game_state))
            self.train()

        new_states = []
        for action in actions:
            new_states.append(self.game_rule.generateSuccessor(copy.deepcopy(game_state), action, self.id))
        scores = [] # stores the state and the score for this state
        for i,new_state in enumerate(new_states):
            scores.append((actions[i], self.model.predict(np.array([self.flatten_state(new_state)]))[0][0]))   
        # Use epsilon greedy to select the action
        this_action = self.epsilon_greedy(scores)
        # Update old_state and old_action
        self.old_state = copy.deepcopy(game_state)
        self.old_action = this_action
        #update the step (for tensorboard) by 1 
        #self.step += 1
        return this_action

    
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
        
        # Each element (a tuple) in replay memory
        # (observation state (not flatted), action, reward, new observation state (not flatted) )
        # Query NN model for current state values (i.e. V(s)) 
        current_states = np.array([self.flatten_state(transition[0]) for transition in minibatch])
        
        ##### NO NEED #####
        # State values for batch of current states
        # current_vs_list = self.model.predict(current_states).flatten()

        # Get future states from minibatch, then query NN model for their state values
        future_states = np.array([self.flatten_state(transition[3]) for transition in minibatch])
        future_vs_list = self.target_model.predict(future_states).flatten()
        # Get reward for each transition in minibatch
        rewards = [transition[2] for transition in minibatch]
        # print(rewards)
        X = current_states
        y = rewards + self.DISCOUNT_FACTOR * future_vs_list 
        self.model.fit(X, y, batch_size=self.MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if game_end else None)

        ######################### Previous work on Q learning that is not correct #########################
        #future_qs_list = []
        #for i,transition in enumerate(minibatch):
        #    future_state = transition[3]
        #    possible_actions = self.game_rule.getLegalActions(future_state, self.id)
        #    # Find max Q value for this future state
        #    max_future_q = SMALL_NUMBER
        #    for action in possible_actions:
        #        future_future_state = self.game_rule.generateSuccessor(copy.deepcopy(future_state), action, self.id)
        #        future_future_q = self.target_model.predict(np.array([self.flatten_state(future_future_state)]))[0][0]
        #        if future_future_q > max_future_q:
        #            max_future_q = future_future_q
        #    future_qs_list.append(max_future_q)
        #assert len(future_qs_list) == len(minibatch) == len(current_qs_list)

        #X = []
        #y = []

        # Now we need to enumerate our batches (current_state and new_current_state are not flatten)
        #for index, (current_state, action, reward, new_current_state) in enumerate(minibatch):

            # Q value updating
        #    new_q = current_qs_list[index] + self.LEARNING_RATE * (reward + self.DISCOUNT_FACTOR * future_qs_list[index] - current_qs_list[index])
            # And append to our training data
        #    X.append(self.flatten_state(new_current_state))
        #    y.append(new_q)
        ####################################################################################################
        # Fit on all samples as one batch, log only on terminal state
        
    
    def calculate_reward(self, old_state, current_state):
        # Based on previous state and current state, calculate reward
        # If our ring is removed, r = 3
        # If opponent's ring is removed, r = -3
        # If we win, r = 10
        # If opponent wins, r = -10
        # Number of our rings removed
        return self.potential_func(current_state) - self.potential_func(old_state)  
        ############# FULL GAME #############
        """
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
        """
    def potential_func(self,state):
        # Calculate potential function for a given state using heuristic by Khoi
        if state.rings_to_place > 0: return 0
        return 10*self.RingHeuristic(state) + 0.5*(self.ChainHeuristic(state,0) - 0.1*self.ChainHeuristic(state,1))

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

        