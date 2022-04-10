# -*- coding: utf-8 -*-

# Hi from Mac
# DDQN Code
#Imports and SmartHomeModel creation
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from tensorflow import keras
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.optimizers import Adam
import random
import SmartHomeModule
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
# Loadling the SmartHomeModule
shm = SmartHomeModule.SmartHomeSimulator("RF_1month_withdays.h5")
shm.HORIZON = 7*24*60
ALPHA = 0.0
TRAINING_EPISODES = 200
TEST_Episodes = 0
filename = "policy_from_simulation_NN_5_PriceTest_alpha_" + str(ALPHA) + "_episode_" + str(TRAINING_EPISODES)


#%%


TRAIN_END = 0


def discount_rate(): #Gamma
    return 0.95

def learning_rate(): #Alpha
    return 0.001

def batch_size():
    return 200

# A pricing schema used by the Pacific Gas & Electric Co. for its customers 
# in parts of California, 3 which accounts for
# 7 tiers ranging from $0.198 per kWh to $0.849 per kWh, reported in Table 5 

priceschema = [0.198, 0.198, 0.198, 0.198, 0.198, 0.198, 0.198, 0.198, # [0:00 - 7:59]
               0.225, 0.225, 0.225, 0.225, # [8:00 - 11:59]
               0.249, 0.249, # [12:00 - 13:59]
               0.849, 0.849, 0.849, 0.849, # [14:00 - 17:59]
               0.225, 0.225, 0.225, 0.225, # [18:00 - 21:59]
               0.198, 0.198] # [22:00-23:59]
maxprice = 0.849

#%%

class DoubleDeepQNetwork():
    def __init__(self, states, actions, alpha, gamma, epsilon,epsilon_min, epsilon_decay):
        self.nS = states
        self.nA = actions
        self.memory = deque([], maxlen=2500)
        self.alpha = alpha
        self.gamma = gamma
        #Explore/Exploit
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self.build_model()
        self.model_target = self.build_model() #Second (target) neural network
        self.update_target_from_model() #Update weights
        self.loss = []
        
    def build_model(self):
        model = keras.Sequential() #linear stack of layers https://keras.io/models/sequential/
        model.add(keras.layers.Dense(7, input_dim=self.nS, activation='relu')) #[Input] -> Layer 1
        #   Dense: Densely connected layer https://keras.io/layers/core/
        #   24: Number of neurons
        #   input_dim: Number of input variables
        #   activation: Rectified Linear Unit (relu) ranges >= 0
        model.add(keras.layers.Dense(10, activation='relu')) #Layer 2 -> 3
        model.add(keras.layers.Dense(7, activation='relu')) #Layer 2 -> 3
        model.add(keras.layers.Dense(self.nA, activation='linear')) #Layer 3 -> [output]
        #   Size has to match the output (different actions)
        #   Linear activation on the last layer
        model.compile(loss='mean_squared_error', #Loss function: Mean Squared Error
                      optimizer=keras.optimizers.Adam(lr=self.alpha)) #Optimaizer: Adam (Feel free to check other options)
        return model

    def update_target_from_model(self):
        #Update the target model from the base model
        self.model_target.set_weights( self.model.get_weights() )

    def action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.nA) #Explore
        action_vals = self.model.predict(state) #Exploit: Use the NN to predict the correct action from this state
        return np.argmax(action_vals[0])

    def test_action(self, state): #Exploit
        action_vals = self.model.predict(state)
        return np.argmax(action_vals[0])

    def store(self, state, action, reward, nstate, done):
        #Store the experience in memory
        self.memory.append( (state, action, reward, nstate, done) )

    def reward_fuction(self, state, action, reward, alpha, priceschema, maxprice):
        cost = 1.0
        if(action != 0):
            cost = ((maxprice-priceschema[int(state[1])])/maxprice)
        new_reward = alpha * reward + (1-alpha) * cost 
        return new_reward

    def experience_replay(self, batch_size):
        
        #Execute the experience replay
        minibatch = random.sample( self.memory, batch_size ) #Randomly sample from memory
        #Convert to numpy for speed by vectorization
        x = []
        y = []
        
        np_array = np.array(minibatch, dtype=object)
        st = np.zeros((0,self.nS)) #States
        nst = np.zeros( (0,self.nS) )#Next States
        #print("down")
        
        for i in range(len(np_array)): #Creating the state and next state np arrays
            st = np.append( st, np_array[i,0], axis=0)
            nst = np.append( nst, np_array[i,3], axis=0)
        #print("up")
        st_predict = self.model.predict(st) #Here is the speedup! I can predict on the ENTIRE batch
        nst_predict = self.model.predict(nst)
        
        nst_predict_target = self.model_target.predict(nst) #Predict from the TARGET
        
        index = 0
        
        for state, action, reward, nstate, done in minibatch:
            x.append(state)
            #Predict from state
            nst_action_predict_target = nst_predict_target[index]
            nst_action_predict_model = nst_predict[index]
            if done == True: #Terminal: Just assign reward much like {* (not done) - QB[state][action]}
                target = reward
            else:   #Non terminal
                target = reward + self.gamma * nst_action_predict_target[np.argmax(nst_action_predict_model)] #Using Q to get T is Double DQN
            target_f = st_predict[index]
            target_f[action] = target
            y.append(target_f)
            index += 1
        #Reshape for Keras Fit
        
        x_reshape = np.array(x).reshape(batch_size,self.nS)
        y_reshape = np.array(y)
        epoch_count = 1
        
        hist = self.model.fit(x_reshape, y_reshape, epochs=epoch_count, verbose=0)
        
        #Graph Losses
        for i in range(epoch_count):
            self.loss.append( hist.history['loss'][i] )
        #Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

nS = shm.state_space #This is only 6
nA = shm.action_space #Actions 2
dqn = DoubleDeepQNetwork(nS, nA, learning_rate(), discount_rate(), 1, 0.001, 0.995 )

batch_size = batch_size()



    
#%%
import datetime
#Training
rewards = [] #Store rewards for graphing
epsilons = [] # Store the Explore/Exploit


for e in range(TRAINING_EPISODES):
    start = datetime.datetime.now()
    np_state = shm.reset()
    state = np.reshape(np_state, [1, nS]) # Resize to store in memory to pass to .predict
    tot_rewards = 0
    for time in range(shm.HORIZON): #200 is when you "solve" the game. This can continue forever as far as I know
        #start1 = datetime.datetime.now()
        action = dqn.action(state)
        #end1 = datetime.datetime.now()
        #print("predict time by net {}".format(end1-start1))
        
        np_nstate, reward, done, _ = shm.step_simulation(action) # use the model to get to the next state & reward
        nstate = np.reshape(np_nstate, [1, nS])
        
        # apply the reward fuction here
        #print("state {} hour {} priceschema {}".format(state, int(np_state[1]), priceschema[int(np_state[1])]))
        reward = dqn.reward_fuction(np_state, action, reward, ALPHA, priceschema, maxprice)
        
        tot_rewards += reward
        
        
        
        dqn.store(state, action, reward, nstate, done) # Resize to store in memory to pass to .predict
        state = nstate
        #done: CartPole fell. 
        #time == 209: CartPole stayed upright
        if done or time == shm.HORIZON-1:
            rewards.append(tot_rewards)
            epsilons.append(dqn.epsilon)
            print("episode: {}/{}, score: {}, e: {}"
                  .format(e, TRAINING_EPISODES, tot_rewards, dqn.epsilon))
            break
    #Experience Replay
    if len(dqn.memory) > batch_size:
        #start2 = datetime.datetime.now()
        dqn.experience_replay(batch_size)
        #end2 = datetime.datetime.now()
        #print("exp replay training time {}".format(end2-start2))
    #Update the weights after each episode (You can configure this for x steps as well
    dqn.update_target_from_model()
    #If our current NN passes we are done
    #I am going to use the last 5 runs
    #if len(rewards) > 5 and np.average(rewards[-5:]) > 195:
        #Set the rest of the EPISODES for testing
    #    TEST_Episodes = EPISODES - e
    #    TRAIN_END = e
    #    break
    end = datetime.datetime.now()
    print("Episode Runtime {}".format(end-start))
    
#%%
#Plotting
rolling_average = np.convolve(rewards, np.ones(100)/100)
plt.plot(rewards, label='reward')
#plt.plot(rolling_average, color='black')
#plt.axhline(y=195, color='r', linestyle='-') #Solved Line
#Scale Epsilon (0.001 - 1.0) to match reward (0 - 200) range
eps_graph = [shm.HORIZON*x for x in epsilons]
plt.plot(eps_graph, color='g', linestyle='-', label='epsilon')
#Plot the line where TESTING begins
plt.axvline(x=TRAIN_END, color='y', linestyle='-')
plt.xlim( (0,TRAINING_EPISODES) )
plt.ylim( (0,shm.HORIZON+2) )
#plt.show()    
leg = plt.legend()
plt.savefig(filename +'.png')

#%%
#import pickle
# save the model to disk
#pickle.dump(dqn.model, open("policy_from_simulation_withdays_NN5_tran1.h5", 'wb'))

#%%
dqn.model.save(filename + ".pol")


#%%

shm.HORIZON = 1*24*60

nS = shm.state_space #This is only 6
nA = shm.action_space #Actions 2



#Testing
#print('Training complete. Testing started...')
#TEST Time
#   In this section we ALWAYS use exploit don't train any more

rewards = []
epsilons = []

print("Testing {}".format(filename))

for e_test in range(TEST_Episodes):
    np_state = shm.reset()
    state = np.reshape(np_state, [1, nS])
    tot_rewards = 0
    cost = 0
    
    for t_test in range(shm.HORIZON):
        action = np.argmax(dqn.predict(state)[0])
        
        nstate, reward, done, _ = shm.step_simulation(action)
        
        if(action != 0):
            cost += priceschema[int(np_state[1])]
            print("action {} min {} hour {} charge {} reward {} cost {} ts {}".format(action, nstate[0], nstate[1], nstate[3], reward, cost, shm.cur_timestamp))
        
        #if(nstate[0] == 0):
        #print("action {} min {} hour {} charge {} reward {} ts {}".format(action, nstate[0], nstate[1], nstate[3], reward, shm.cur_timestamp))
        nstate = np.reshape( nstate, [1, nS])
        tot_rewards += reward
        #DON'T STORE ANYTHING DURING TESTING
        state = nstate
        #done: CartPole fell. 
        #t_test == 209: CartPole stayed upright
        if done or t_test == shm.HORIZON - 1: 
            rewards.append(tot_rewards)
            epsilons.append(0) #We are doing full exploit
            print("episode: {}/{}, pref_score: {}, cost: {}"
                  .format(e_test, TEST_Episodes, tot_rewards, cost))
            break;
