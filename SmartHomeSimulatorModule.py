class Preference:
  device_name = "Tesla"
  charge_status = 90.0
  when = "between"
  start_time_hour = 0
  start_time_min = 30
  start_time_std_min = 5
  end_time_hour = 3
  end_time_min = 30
  end_time_std_min = 5

import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from numpy import asarray

class SmartHomeSimulatorNBClass:
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

  def get_state_space(self):
    return self.state_space

  def get_action_space(self):
      return self.action_space

  def __init__(self, name_model):
    self.state_space = 4
    self.action_space = 2
    self.load_model(name_model)
    print("Inside constructor")
    # body of the constructor

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
    print("Reset function")
    # Reset the model
    # clear all the history of the smart home
    return self.get_cur_state()
    
    
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
  def step(self, action):
    done = False
    log = ""
    input_state_action = self.get_cur_state_action_prediction(action)
    prediction = self.model.predict(input_state_action)
    ncharge_status = prediction[0][0]
    reward = prediction[0][1]
    self.update_time()
    self.cur_charge_status = ncharge_status
    nstate =  self.get_cur_state()
    if(self.cur_timestamp == self.HORIZON):
      done = True
    return nstate, reward, done, log

  # print("Simulating a minute in the smart home for action ", (, reward[0][1]))
  # return nstate, reward, done, _ like the gym environment
#shm = SmartHomeSimulatorNBClass("RF_1month.h5")

#for x in range(120):
#   nstate, reward, done, _ = shm.step(1)
#   print(nstate,' ', reward)