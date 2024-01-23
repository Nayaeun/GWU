import pandas as pd
import numpy as np
import os
#%%

data_path = os.getcwd()+'/data'
data_files = os.listdir(data_path)

#%%
target_train = pd.read_csv(f'{data_path}/target_train.csv')
print(target_train.head().to_string())
print(target_train.columns)
#%%
rxclms_train = pd.read_csv(f'{data_path}/rxclms_train.csv')
print(rxclms_train.head().to_string())
print(rxclms_train.columns)
#%%
medclms_train = pd.read_csv(f'{data_path}/medclms_train.csv')
print(medclms_train.head().to_string())
print(medclms_train.columns)
#%%
# data_joined = pd.read_csv('/Users/jiwoosuh/Downloads/2023_TAMU_competition_data/data_joined.csv', encoding="UTF-16")
# print(data_joined.head(10).to_string())
# print(data_joined.columns)

data_joined1 = pd.merge(rxclms_train, medclms_train, on ='therapy_id')
data_joined = pd.merge(data_joined1, target_train, on ='therapy_id')
print(data_joined.head(10).to_string())
print(data_joined.columns)

