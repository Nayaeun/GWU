# HaeLee & Yoni

#%%
########################################################################################################
# Grouping Pharmacy Claim Dataset
########################################################################################################
import pandas as pd
import numpy as np
import os

data_path = os.getcwd()+'/data'
rxclms_train_path = data_path + os.sep + 'rxclms_train.csv'
rxclms_train = pd.read_csv(rxclms_train_path)
print(rxclms_train.head().to_string())
print(f'\nThe number of datapoints: {rxclms_train.shape[0]}\n\nThe number of features: {rxclms_train.shape[1]}\n\nfeatures list:\n{rxclms_train.columns}')

#%%
selected_rxclms_train = rxclms_train[['therapy_id', 'hum_drug_class_desc']]
print(selected_rxclms_train.head().to_string())

#%%
grouped_rxclms_train = selected_rxclms_train.groupby('therapy_id')['hum_drug_class_desc'].agg(list).reset_index()
grouped_rxclms_train.rename(columns={'hum_drug_class_desc': 'collapsed_drug_classes'}, inplace=True)
print(grouped_rxclms_train.head())

#%%
grouped_rxclms_train['num_drug_classes'] = grouped_rxclms_train['collapsed_drug_classes'].apply(len)
print(grouped_rxclms_train.head())

#%%
unique_classes = set()
for classes_list in grouped_rxclms_train['collapsed_drug_classes']:
    unique_classes.update(classes_list)

for drug_class in unique_classes:
    grouped_rxclms_train[drug_class] = grouped_rxclms_train['collapsed_drug_classes'].apply(lambda x: x.count(drug_class))

grouped_rxclms_train.drop(columns=['collapsed_drug_classes'], inplace=True)

print(f'\nThe number of datapoints: {grouped_rxclms_train.shape[0]}\n\nThe number of features: {grouped_rxclms_train.shape[1]}\n\nfeatures list:\n{grouped_rxclms_train.columns}')
print(grouped_rxclms_train.head())
grouped_rxclms_train.to_csv('grouped_rxclms_train.csv', index=False)
# (1160,75)

#%%
###############################################################################################################################
# Merging Pharmacy Claim Dataset with Target Dataset
###############################################################################################################################

data_path = os.getcwd()+'/data'
target_train_path = data_path + os.sep + 'target_train.csv'
target_train = pd.read_csv(target_train_path)

print(f'\nThe number of datapoints: {target_train.shape[0]}\n\nThe number of features: {target_train.shape[1]}\n\nfeatures list:\n{target_train.columns}')
print(target_train.head().to_string())
# (1232, 10)

#%%
new_target_train = target_train[['therapy_id', 'tgt_ade_dc_ind', 'race_cd', 'est_age', 'sex_cd',
                                 'cms_disabled_ind','cms_low_income_ind']]
print(f'\nThe number of datapoints: {new_target_train.shape[0]}\n\nThe number of features: {new_target_train.shape[1]}\n\nfeatures list:\n{new_target_train.columns}')
new_target_train.to_csv('new_target_train.csv', index=False)
# (1232, 7)

# %%
df1 = pd.read_csv('new_target_train.csv')  
df2 = pd.read_csv('grouped_rxclms_train.csv') 
merged_df1 = pd.merge(df1, df2, on='therapy_id', how='outer')

columns_to_fill_with_zeros = df2.columns.difference(['therapy_id'])  
merged_df1[columns_to_fill_with_zeros] = merged_df1[columns_to_fill_with_zeros].fillna(0)

print(f'\nThe number of datapoints: {merged_df1.shape[0]}\n\nThe number of features: {merged_df1.shape[1]}\n\nfeatures list:\n{merged_df1.columns}')
print(merged_df1.head())
merged_df1.to_csv('merged_df1.csv', index=False)
# (1232, 81)

########################################################################################################
# Grouping Medical Claim Dataset
########################################################################################################
#%%
data_path = os.getcwd()+'/data'
medclms_train_path = data_path + os.sep + 'medclms_train.csv'
medclms_train = pd.read_csv(medclms_train_path)
print(medclms_train.head().to_string())
print(f'\nThe number of datapoints: {medclms_train.shape[0]}\n\nThe number of features: {medclms_train.shape[1]}\n\nfeatures list:\n{medclms_train.columns}')

#%%
selected_medclms_train = medclms_train[['therapy_id', 'primary_diag_cd']]
selected_medclms_train['primary_diag_cd'].isnull().sum()

#%%
grouped_medclms_train = selected_medclms_train.groupby('therapy_id')['primary_diag_cd'].agg(list).reset_index()
grouped_medclms_train['num_diag_cd'] = grouped_medclms_train['primary_diag_cd'].apply(len)

#%%
unique_diag = set()
for diag in grouped_medclms_train['primary_diag_cd']:
    unique_diag.update(diag)

for diag in unique_diag:
    grouped_medclms_train[diag] = grouped_medclms_train['primary_diag_cd'].apply(lambda x: x.count(diag))

grouped_medclms_train.drop(columns=['primary_diag_cd'], inplace=True)

print(grouped_medclms_train.head().to_string())
print(f'\nThe number of datapoints: {grouped_medclms_train.shape[0]}\n\nThe number of features: {grouped_medclms_train.shape[1]}\n\nfeatures list:\n{grouped_medclms_train.columns}')
grouped_medclms_train.to_csv('grouped_medclms_train.csv', index=False)
# (536, 1863)

#%%
########################################################################################################
# Merging Medical Claim Dataset with Merged_df1 (= Target Dataset already merged with Pharmacy Claim Dataset)
########################################################################################################
df1 = pd.read_csv('merged_df1.csv')  
df2 = pd.read_csv('grouped_medclms_train.csv') 
merged_df2 = pd.merge(df1, df2, on='therapy_id', how='outer')

columns_to_fill_with_zeros = df2.columns.difference(['therapy_id'])  
merged_df2[columns_to_fill_with_zeros] = merged_df2[columns_to_fill_with_zeros].fillna(0)

print(f'\nThe number of datapoints: {merged_df2.shape[0]}\n\nThe number of features: {merged_df2.shape[1]}\n\nfeatures list:\n{merged_df2.columns}')
print(merged_df2.head())
merged_df2.to_csv('merged_df2.csv', index=False)
# (1232, 1943)


########################################################################################################
# Final dataset is df 
########################################################################################################
# %%
df = merged_df2
null_counts = df.isnull().sum()
columns_with_null = null_counts[null_counts > 0]
print(columns_with_null)

#%%
df = df.dropna()
print(f'\nThe number of datapoints: {df.shape[0]}\n\nThe number of features: {df.shape[1]}\n\nfeatures list:\n{df.columns}')
print(df.head())
df.to_csv('df.csv', index=False)
# (1114, 1943)

#%%
########################################################################################################
# Feature Selection / Dimensionality Reduction
########################################################################################################
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
# from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.model_selection import GridSearchCV


# Create dummies for categorical variables
df = pd.get_dummies(df, columns=['sex_cd'])
df['sex_cd_F'] = df['sex_cd_F'].astype(float)
df['sex_cd_M'] = df['sex_cd_M'].astype(float)

# Drop columns not needed for X
X = df.drop(columns=['therapy_id', 'tgt_ade_dc_ind'])

# Extract the target variable 'y'
y = df['tgt_ade_dc_ind']  

# Get the datatype of each column in the updated DataFrame
column_datatypes = X.dtypes

# Print the datatype of each column
for column, datatype in column_datatypes.items():
    if datatype != 'float64':
        print(f'Column: {column}')
        
# #%%
# # Check Imblance ratio
# target_variable = 'tgt_ade_dc_ind'

# # Count the number of samples in each class
# class_distribution = df[target_variable].value_counts()

# # Calculate the class imbalance ratio
# imbalance_ratio = class_distribution[0] / class_distribution[1]

# # Display the class distribution and imbalance ratio
# print("Class Distribution:")
# print(class_distribution)
# print("Imbalance Ratio (Class 0 / Class 1):", imbalance_ratio)


# #%%
# # Apply SMOTE to the training data
# smote = SMOTE(sampling_strategy='auto', random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X, y)

 #%%
# Train a Random Forest model to get feature importances
model = RandomForestClassifier()
model.fit(X, y)
feature_importances = model.feature_importances_

# Select the top k most important features
k = 10  # You can choose the number of features you want to keep
selected_features = X.columns[np.argsort(feature_importances)[-k:]]

# Apply Feature Scaling (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[selected_features])

# Dimensionality Reduction (example: using PCA)
pca = PCA(n_components=2)  # Choose the number of components you want
X_pca = pca.fit_transform(X_scaled)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train a classifier and evaluate its performance (e.g., accuracy)

#%%
#=================Random Forest==============================
# Define the parameter grid for Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
}

# Initialize the Random Forest classifier
clf = RandomForestClassifier(random_state=42)

# Perform Grid Search
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Train the model with the best hyperparameters
best_model = RandomForestClassifier(random_state=42, **best_params)
best_model.fit(X_train, y_train)

# Assuming you have already trained 'best_model' using Grid Search
# Make predictions on the test data
y_pred = best_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Random Forest Accuracy:", accuracy)


# clf = RandomForestClassifier()
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)

#%%
#=======================XGBoost===============================
# Initialize the XGBoost classifier
model = xgb.XGBClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("XGBoost Accuracy:", accuracy)

#%%
########################################################################################################
# Preparing Testing Set / Grouping/Merging for Testset
########################################################################################################

# 1. Process rxclms_holdout #############################################################################
data_path = os.getcwd()+'/data'
rxclms_holdout_path = data_path + os.sep + 'rxclms_holdout.csv'
rxclms_holdout = pd.read_csv(rxclms_holdout_path)
selected_rxclms_holdout = rxclms_holdout[['therapy_id', 'hum_drug_class_desc']]
grouped_rxclms_holdout = selected_rxclms_holdout.groupby('therapy_id')['hum_drug_class_desc'].agg(list).reset_index()
grouped_rxclms_holdout.rename(columns={'hum_drug_class_desc': 'collapsed_drug_classes'}, inplace=True)
grouped_rxclms_holdout['num_drug_classes'] = grouped_rxclms_holdout['collapsed_drug_classes'].apply(len)
unique_classes = set()
for classes_list in grouped_rxclms_holdout['collapsed_drug_classes']:
    unique_classes.update(classes_list)
for drug_class in unique_classes:
    grouped_rxclms_holdout[drug_class] = grouped_rxclms_holdout['collapsed_drug_classes'].apply(lambda x: x.count(drug_class))
grouped_rxclms_holdout.drop(columns=['collapsed_drug_classes'], inplace=True)
print(f'\nThe number of datapoints: {grouped_rxclms_holdout.shape[0]}\n\nThe number of features: {grouped_rxclms_holdout.shape[1]}\n\nfeatures list:\n{grouped_rxclms_holdout.columns}')
print(grouped_rxclms_holdout.head())
grouped_rxclms_holdout.to_csv('grouped_rxclms_holdout.csv', index=False)
# (379, 60)

#%%
# 2. Merging target_holdout with Process rxclms_holdout #############################################################################
data_path = os.getcwd()+'/data'
target_holdout_path = data_path + os.sep + 'target_holdout.csv'
target_holdout = pd.read_csv(target_holdout_path)
new_target_holdout = target_holdout[['therapy_id', 'race_cd', 'est_age', 'sex_cd',
                                 'cms_disabled_ind','cms_low_income_ind']]
print(f'\nThe number of datapoints: {new_target_holdout.shape[0]}\n\nThe number of features: {new_target_holdout.shape[1]}\n\nfeatures list:\n{new_target_holdout.columns}')
new_target_holdout.to_csv('new_target_holdout.csv', index=False)
# (420, 6)

# %%
df1 = pd.read_csv('new_target_holdout.csv')  
df2 = pd.read_csv('grouped_rxclms_holdout.csv') 
merged_df1 = pd.merge(df1, df2, on='therapy_id', how='outer')
columns_to_fill_with_zeros = df2.columns.difference(['therapy_id'])  
merged_df1[columns_to_fill_with_zeros] = merged_df1[columns_to_fill_with_zeros].fillna(0)
print(f'\nThe number of datapoints: {merged_df1.shape[0]}\n\nThe number of features: {merged_df1.shape[1]}\n\nfeatures list:\n{merged_df1.columns}')
print(merged_df1.head())
merged_df1.to_csv('test_merged_df1.csv', index=False)
# (420, 65)

#%%
# 3. Process mdclms_holdout #############################################################################
data_path = os.getcwd()+'/data'
medclms_holdout_path = data_path + os.sep + 'medclms_holdout.csv'
medclms_holdout = pd.read_csv(medclms_holdout_path)
selected_medclms_holdout = medclms_holdout[['therapy_id', 'primary_diag_cd']]
selected_medclms_holdout['primary_diag_cd'].isnull().sum()
grouped_medclms_holdout = selected_medclms_holdout.groupby('therapy_id')['primary_diag_cd'].agg(list).reset_index()
grouped_medclms_holdout['num_diag_cd'] = grouped_medclms_holdout['primary_diag_cd'].apply(len)
unique_diag = set()
for diag in grouped_medclms_holdout['primary_diag_cd']:
    unique_diag.update(diag)
for diag in unique_diag:
    grouped_medclms_holdout[diag] = grouped_medclms_holdout['primary_diag_cd'].apply(lambda x: x.count(diag))
grouped_medclms_holdout.drop(columns=['primary_diag_cd'], inplace=True)
print(grouped_medclms_holdout.head().to_string())
print(f'\nThe number of datapoints: {grouped_medclms_holdout.shape[0]}\n\nThe number of features: {grouped_medclms_holdout.shape[1]}\n\nfeatures list:\n{grouped_medclms_holdout.columns}')
grouped_medclms_holdout.to_csv('grouped_medclms_holdout.csv', index=False)
#(185, 856)

#%%
# 4. test_merged_df1 with Process mdclms_holdout #############################################################################
df1 = pd.read_csv('test_merged_df1.csv')  
df2 = pd.read_csv('grouped_medclms_holdout.csv') 
merged_df2 = pd.merge(df1, df2, on='therapy_id', how='outer')
columns_to_fill_with_zeros = df2.columns.difference(['therapy_id'])  
merged_df2[columns_to_fill_with_zeros] = merged_df2[columns_to_fill_with_zeros].fillna(0)
print(f'\nThe number of datapoints: {merged_df2.shape[0]}\n\nThe number of features: {merged_df2.shape[1]}\n\nfeatures list:\n{merged_df2.columns}')
print(merged_df2.head())
merged_df2.to_csv('test_merged_df2.csv', index=False)
# (420, 920)

#%%
# 5. fianl testset #############################################################################
df_test = pd.read_csv('test_merged_df2.csv')
null_counts = df_test.isnull().sum()
columns_with_null = null_counts[null_counts > 0]
df_test = df_test.dropna()
print(f'\nThe number of datapoints: {df_test.shape[0]}\n\nThe number of features: {df_test.shape[1]}\n\nfeatures list:\n{df_test.columns}')
print(df_test.head())
df_test.to_csv('df_test.csv', index=False)
# (385, 920)

#%%
########################################################################################################
# Testing Accuracy on Test Set
########################################################################################################

df_test = pd.get_dummies(df_test, columns=['sex_cd'])
df_test['sex_cd_F'] = df_test['sex_cd_F'].astype(float)
df_test['sex_cd_M'] = df_test['sex_cd_M'].astype(float)
X_test = df_test.drop(columns=['therapy_id'])

X_test_scaled = scaler.transform(X_test[selected_features])
X_test_pca = pca.transform(X_test_scaled)

# Make predictions using RandomForest
y_pred_test = best_model.predict(X_test_pca)
print(y_pred_test)


#%%
y_pred_proba = best_model.predict_proba(X_test_pca)
ranks = (y_pred_proba[:, 1] * -1).argsort().argsort() + 1

results_df = pd.DataFrame({
    'therapy_id': df_test['therapy_id'],  # Assuming you have a 'therapy_id' column
    'predicted_probability': y_pred_proba[:, 1],
    'rank': ranks
})
results_df = results_df.sort_values(by='rank', ascending=True)
results_df.to_csv('RF_submission_results.csv', index=False)


#%%
# CAUTION: We Need True Values to Check Accuracy ###############################################################################################
# Evaluate the model's performance on the test set
# You can use accuracy_score or other appropriate metrics
y_true_test = merged_df2['tgt_ade_dc_ind']  # True labels from the merged_df2 (training data)
accuracy_test = accuracy_score(y_true_test, y_pred_test)

print(f'Accuracy on the test set: {accuracy_test}')
# %%
