# -*- coding: utf-8 -*-
import SmartHomeModule
import random 

#%%
shm = SmartHomeModule.SmartHomeSimulator("RF_1month.h5")
TOTAL_DAYS = 1
HOUR_IN_A_DAY = 24
MINUTE_IN_A_HOUR = 60
shm.HORIZON = TOTAL_DAYS*HOUR_IN_A_DAY*MINUTE_IN_A_HOUR

nS = shm.state_space #This is only 6
nA = shm.action_space #Actions 2

def generate_dataset(log_file_name, SEED):
    random.seed(SEED)
    f = open(log_file_name, "w")
    f.write("min,hour,week_day,cur_charge,action,next_charge,reward\n")
    state = shm.reset()
    for ts in range(shm.HORIZON):
        action = random.randint(0, nA-1)
        log = str(state[0]) + "," + str(state[1]) + "," + str(state[2]) + "," + str(state[3]) + "," + str(action)
        nstate, reward, done, _ = shm.step_simulation(action)
        log += "," + str(nstate[3]) + "," + str(reward) + "\n"
        f.write(log)
        #DON'T STORE ANYTHING DURING TESTING
        state = nstate
        #done: CartPole fell. 
        #t_test == 209: CartPole stayed upright
        if done or ts == shm.HORIZON - 1: 
            break;
    f.close()

generate_dataset("train_dataset_withdays.csv", 1)
generate_dataset("test_dataset_withdays.csv", 2)