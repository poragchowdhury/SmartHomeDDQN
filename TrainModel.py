# -*- coding: utf-8 -*-
#%%

import pandas as pd
# get the dataset
def get_dataset(file):
  df_X = pd.read_csv(file)
  df_y = df_X[['next_charge', 'reward']] 
  df_X.drop(['next_charge'], axis=1, inplace=True)
  df_X.drop(['reward'], axis=1, inplace=True)
  #df_X = df_X.drop(['device','cur_day_number', 'charge_iRobot_651_battery', 'water_temp_water_heat_sensor', 'temperature_cool_thermostat_cool', 'bake_Kenmore_790_sensor', 'dish_wash_Kenmore_665_sensor', 'laundry_wash_GE_WSM2420D3WW_wash_sensor', 'laundry_dry_GE_WSM2420D3WW_dry_sensor', 'temperature_heat_thermostat_heat', 'cleanliness_dust_sensor'], axis=1);
  #df_y = df_X.copy()
  #df_X.drop(df_X.head(1).index,inplace=True)
  #df_y = df_y.drop(['cur_hour_of_day', 'cur_min_of_hour', 'cur_day_of_week'], axis=1)
  #df_y.drop(df_y.tail(1).index,inplace=True)
  return df_X, df_y


# load dataset
train_file = 'train_dataset_withdays.csv'
test_file = 'test_dataset_withdays.csv'

X_train, y_train = get_dataset(train_file)
X_test, y_test = get_dataset(test_file)

#%%
# transform my input 
def transform_X(df_X):
  add_columns = pd.get_dummies(df_X['action'])
  df_X.drop(['action'], axis=1, inplace=True)

  # Standardization
  #for column in df_X.columns:
  #  df_X[column] = (df_X[column] -
  #                         df_X[column].mean()) / df_X[column].std()                               
  df_X = df_X.join(add_columns)
  return df_X

# transform my output 
from sklearn.preprocessing import LabelEncoder
def transform_y(df_y):
  lable_encoder = LabelEncoder()
  df_y['action'] = lable_encoder.fit_transform(df_y['action']);
  return df_y

#%%

X_train = transform_X(X_train)
X_test = transform_X(X_test)

#%%

from sklearn.ensemble import RandomForestRegressor
max_depth = 30
regr_rf = RandomForestRegressor(n_estimators=100, max_depth=max_depth, random_state=2)
regr_rf.fit(X_train, y_train)

#%%

import pickle
# save the model to disk
filename = 'RF_1month_withdays.h5'
pickle.dump(regr_rf, open(filename, 'wb'))
#regr_rf.save("RF.h5");
#model.save('/content/drive/MyDrive/Models/NN_model.h5')


#%%

loaded_model = pickle.load(open('RF_1month_withdays.h5', 'rb'))

#%%

y_pred = loaded_model.predict(X_test)

#%%

print(y_pred)
print(y_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test[1],y_pred[1])