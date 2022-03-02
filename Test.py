# -*- coding: utf-8 -*-
import numpy as np
import SmartHomeSimulatorModule
from tensorflow import keras
import pickle
#%%

model_name = "policy_from_model.pol" #"test2.pol"
# load the model to disk
print("Loading model..")
model = keras.models.load_model(model_name)

#model = pickle.load(open("test2.h5", 'rb'))

print("Model loaded successfully")

shm = SmartHomeSimulatorModule.SmartHomeSimulatorNBClass("RF_1month.h5")
shm.HORIZON = 1*24*60

nS = shm.state_space #This is only 6
nA = shm.action_space #Actions 2



#Testing
#print('Training complete. Testing started...')
#TEST Time
TEST_Episodes = 1
#   In this section we ALWAYS use exploit don't train any more

rewards = []
epsilons = []

for e_test in range(TEST_Episodes):
    state = shm.reset()
    state = np.reshape(state, [1, nS])
    tot_rewards = 0
    for t_test in range(shm.HORIZON):
        action = np.argmax(model.predict(state)[0])
        nstate, reward, done, _ = shm.step_simulation(action)
        nstate = np.reshape( nstate, [1, nS])
        tot_rewards += reward
        #DON'T STORE ANYTHING DURING TESTING
        state = nstate
        #done: CartPole fell. 
        #t_test == 209: CartPole stayed upright
        if done or t_test == shm.HORIZON - 1: 
            rewards.append(tot_rewards)
            epsilons.append(0) #We are doing full exploit
            print("episode: {}/{}, score: {}, e: {}"
                  .format(e_test, TEST_Episodes, tot_rewards, 0))
            break;