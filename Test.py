# -*- coding: utf-8 -*-
import numpy as np
import SmartHomeModule
from tensorflow import keras
#import pickle
#%%
ALPHA = 0.7
TRAINING_EPISODES = 1000
TEST_Episodes = 1

model_name = "policy_from_simulation_NN_5_PriceTest_alpha_" + str(ALPHA) + "_episode_" + str(TRAINING_EPISODES) + ".pol"

#model_name = 'policy_from_simulation_NN_5_PriceTest_alpha_0.0.pol' #"test2.pol"
# load the model to disk
print("Loading model.. {}".format(model_name))
model = keras.models.load_model(model_name)

#model = pickle.load(open("test2.h5", 'rb'))

print("Model loaded successfully")

shm = SmartHomeModule.SmartHomeSimulator("RF_1month.h5")
shm.HORIZON = 1*24*60

nS = shm.state_space #This is only 6
nA = shm.action_space #Actions 2



#Testing
#print('Training complete. Testing started...')
#TEST Time
#   In this section we ALWAYS use exploit don't train any more

rewards = []
epsilons = []

priceschema = [0.198, 0.198, 0.198, 0.198, 0.198, 0.198, 0.198, 0.198, # [0:00 - 7:59]
               0.225, 0.225, 0.225, 0.225, # [8:00 - 11:59]
               0.249, 0.249, # [12:00 - 13:59]
               0.849, 0.849, 0.849, 0.849, # [14:00 - 17:59]
               0.225, 0.225, 0.225, 0.225, # [18:00 - 21:59]
               0.198, 0.198] # [22:00-23:59]
maxprice = 0.849

for e_test in range(TEST_Episodes):
    np_state = shm.reset()
    state = np.reshape(np_state, [1, nS])
    tot_rewards = 0
    cost = 0
    
    for t_test in range(shm.HORIZON):
        action = np.argmax(model.predict(state)[0])
        
        nstate, reward, done, _ = shm.step_simulation(action)

        if(action != 0):
            cost += priceschema[int(np_state[1])]
            #print("action {} min {} hour {} charge {} reward {} cost {} ts {}".format(action, nstate[0], nstate[1], nstate[3], reward, cost, shm.cur_timestamp))
        
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