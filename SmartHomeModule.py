#%%
import pickle
import numpy as np
from numpy import asarray

#%%
# importing the module
import json

class SmartHomeSimulator:
    HOUSE_SIZE = 0
    device_name = 'Tesla_S'
    actions = []
    state_space = 4
    action_space = 2
    cur_min = 0
    cur_hour = 0
    cur_week_of_day = 0
    cur_charge_status = 30.0
    cur_timestamp = 0;
    HORIZON = 60
    path = ''
    model = ''
    devices = {}
    preferences = {}
    days = ["Sat", "Sun", "Mon", "Tues", "Wed", "Thurs", "Fri"]
    
    def get_state_space(self):
        return self.state_space
    
    def get_action_space(self):
        return self.action_space
    
    def __init__(self, name_model):
        self.state_space = 4
        self.action_space = 2
        self.load_model(name_model)
        self.load_preferences()
        self.load_device_configuration()
        print("Constructor call successfull")
      
    def simulate():
        print("Simulating smart home")
    
    def save_model(self, model_instance, model_name):
        pickle.dump(model_instance, open(self.path+model_name, 'wb'))
        print("Saving a model")
    
    def load_model(self, model_name):
        # save the model to disk
        self.model = pickle.load(open(model_name, 'rb'))
        print("Model loaded successfully")
    
    def reset(self):
        self.cur_min = 0
        self.cur_hour = 0
        self.cur_week_of_day = 0
        self.cur_charge_status = 30.0
        self.cur_timestamp = 0
        print("Reset function")
        # Reset the model
        # clear all the history of the smart home
        return self.get_cur_state()
      
    def load_device_configuration(self):
        # Opening JSON file
        with open('DeviceDictionary.json') as json_file:
            self.devices = json.load(json_file)[self.HOUSE_SIZE]
            
        self.actions = list(self.devices[self.device_name]['actions'].keys())
        self.action_space = len(self.actions)
        for act_id in range(self.action_space):
            for delta in self.devices[self.device_name]['actions'][self.actions[act_id]]['effects']: # 
                delta['delta'] /= 60
          
    def load_preferences(self):
        with open('Preferences.json') as json_file:
            self.preferences = json.load(json_file)
        for pref in self.preferences[self.device_name]:
            if('time_relation' in pref):
                mean_ts = pref['start_time_distribution'][0]*60+pref['start_time_distribution'][1]
                std_ts = np.random.normal(0, pref['start_time_distribution'][2]*60+pref['start_time_distribution'][3])
                pref["sampled_time1"] = mean_ts + std_ts
                # TODO : Check for negative time samples
                if('end_time_distribution' in pref):
                    mean_ts = pref['end_time_distribution'][0]*60+pref['end_time_distribution'][1]
                    std_ts = np.random.normal(0, pref['end_time_distribution'][2]*60+pref['end_time_distribution'][3])
                    pref["sampled_time2"] = int(mean_ts + std_ts)
                    # TODO : Check for negative time samples
                  
      
    def update_time(self):
        self.cur_timestamp = self.cur_timestamp + 1;
        self.cur_min = self.cur_min + 1
        if(self.cur_min == 60):
          self.cur_min == 0
          self.cur_hour = self.cur_hour + 1
          if(self.cur_hour == 24):
            self.cur_hour == 0
            self.cur_week_of_day = self.cur_week_of_day + 1
            if(self.cur_week_of_day == 7):
              self.cur_week_of_day = 0
    
    def get_cur_state_action_prediction(self, action):
        state = []
        if(action == 0):
          state = asarray([[self.cur_min, self.cur_hour, self.cur_week_of_day, self.cur_charge_status, 1, 0]])
        else: 
          state = asarray([[self.cur_min, self.cur_hour, self.cur_week_of_day, self.cur_charge_status, 0, 1]])
        return state
    
    def get_cur_state(self):
        #state = []
        #if(action == 0):
        state = np.array([self.cur_min, self.cur_hour, self.cur_week_of_day, self.cur_charge_status])
        #else: 
        #  state = np.array([self.cur_min, self.cur_hour, self.cur_week_of_day, self.cur_charge_status, 0, 1])
        return state
      
    # Simulate one step in the smart home
    def step_model(self, action):
        done = False
        log = ""
        input_state_with_action = self.get_cur_state_action_prediction(action)
        prediction = self.model.predict(input_state_with_action)
        ncharge_status = prediction[0][0]
        reward = prediction[0][1]
        self.update_time()
        self.cur_charge_status = ncharge_status
        nstate =  self.get_cur_state()
        if(self.cur_timestamp == self.HORIZON):
          done = True
        return nstate, reward, done, log
    
    def get_delta(self, act_id):
        return self.devices[self.device_name]['actions'][self.actions[act_id]]['effects'][0]['delta']
    
    def is_valid_day(self, pref):
        if('day_relation' not in pref): 
            return True  
        
        for day in pref['day_relation']:
            if(day == self.days[self.cur_week_of_day]):
                return True  
        return False
      
    
      
    def simulation(self, action):
        delta = self.get_delta(action)
        if(self.cur_charge_status + delta < 100 and self.cur_charge_status + delta > 0):
            self.cur_charge_status += delta
        
        for pref in self.preferences[self.device_name]:
            # preference priorities
            # 1. check time & property preferences
            
            if(self.is_valid_day(pref)):
                if('time_relation' in pref):
                    if(pref['time_relation'] == 'before' and pref['sampled_time1'] > self.cur_timestamp):
                        if(pref['property_value'] > self.cur_charge_status):
                            # property need to increase
                            if(delta <= 0):
                                return -1
                    elif(pref['time_relation'] == 'after' and pref['sampled_time1'] < self.cur_timestamp):
                        if(pref['property_value'] > self.cur_charge_status):
                            # property need to increase
                            if(delta <= 0):
                                return -1
                    elif(pref['time_relation'] == 'between' and pref['sampled_time1'] > self.cur_timestamp and pref['sampled_time2'] < self.cur_timestamp):
                        if(pref['property_value'] > self.cur_charge_status):
                            # property need to increase
                            if(delta <= 0):
                                return -1   
                    
            # 2. check day preferences
            # TODO 
        return 1  
    
    def step_simulation(self, action):
        done = False
        log = ""
        reward = self.simulation(action) #prediction[0][1]
        self.update_time()
        nstate =  self.get_cur_state()
        if(self.cur_timestamp == self.HORIZON):
          done = True
        return nstate, reward, done, log  

  # print("Simulating a minute in the smart home for action ", (, reward[0][1]))
  # return nstate, reward, done, _ like the gym environment
#shm = SmartHomeSimulator("RF_1month.h5")
#print(shm.devices[shm.device_name])
#for x in range(10):
#   nstate, reward, done, _ = shm.step_model(1)
#   print(nstate,' ', reward)

#%%
#arr = list(shm.devices[shm.device_name]['actions'].keys())
#print(shm.devices[shm.device_name]['actions']['charge_48a']['effects'][0]['delta'])


