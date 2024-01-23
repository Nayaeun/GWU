#%%
import pandas as pd
import numpy as np
import os

org_path = os.getcwd()
print (org_path)
data_path = org_path + os.sep + 'data'
print (data_path)
data_files = os.listdir(data_path)

#%%
target_train_path = data_path + os.sep + 'target_train.csv'
target_train = pd.read_csv(target_train_path)
print(target_train.head().to_string())
print(f'\nThe shape of features: {target_train.shape}\n\nfeatures list:\n{target_train.columns}')


#%%
rxclms_train_path = data_path + os.sep + 'rxclms_train.csv'
rxclms_train = pd.read_csv(rxclms_train_path)
print(rxclms_train.head().to_string())
print(f'\nThe number of features: {rxclms_train.shape[1]}\n\nfeatures list:\n{rxclms_train.columns}')


#%%
medclms_train_path = data_path + os.sep + 'medclms_train.csv'
medclms_train = pd.read_csv(medclms_train_path)
print(medclms_train.head().to_string())
print(f'\nThe number of features: {medclms_train.shape[1]}\n\nfeatures list:\n{medclms_train.columns}')


#%%
# data_joined = pd.read_csv('/Users/jiwoosuh/Downloads/2023_TAMU_competition_data/data_joined.csv', encoding="UTF-16")
# print(data_joined.head(10).to_string())
# print(data_joined.columns)

data_joined1 = pd.merge(rxclms_train, medclms_train, on ='therapy_id')
data_joined = pd.merge(data_joined1, target_train, on ='therapy_id')
print(data_joined.head(10).to_string())
print(f'\nThe number of features: {data_joined.shape[1]}\n\nfeatures list:\n{data_joined.columns}')


# %%
