# 1- Cover page
# 2- Table of content
# 3- Table of figures and tables.
# 4- Abstract.
# 5- Introduction. An overview of the time series analysis and modeling process and an outline of the report.
# 6- Description of the dataset. Describe the independent variable(s) and dependent variable:
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.stattools import kpss

from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.api import ARIMA, SARIMAX

from statsmodels.tsa.stattools import acf

from scipy import signal

from statsmodels.stats.diagnostic import acorr_ljungbox

# from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM



import warnings 
warnings.filterwarnings('ignore')
#import os
# os.getcwd()
# os.chdir('/Users/yoninayoni/Library/Mobile Documents/com~apple~CloudDocs/GWU/23Spring/DATS 6313 - TIme Series/Project/')

# pd.set_option('display.width', 100)
# np.set_printoptions(linewidth=100)
# np.set_printoptions(suppress=True, precision=4)

#%% 6-a. Pre-processing dataset: Dataset cleaning for missing observation. You must follow the data cleaning techniques for time series dataset.

# Import train and test dataset
data = pd.read_csv('train.csv', parse_dates=['date'])
#data['date'] = pd.to_datetime(data['date'])

# Check dataset
print ("Data Head:\n", data.head())
print ("Data Statistics:\n", data.describe())

# Missing point
print ("Missing values:", data.isna().sum().sum()) 
# No missing point


#%%
data.plot(x='date', y='sales', figsize=(15, 6), title='Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

#%%
# Reshaping train dataset
df = data.copy()

df['date'] =  pd.to_datetime(df['date'], format='%Y-%m').dt.to_period('M')

df = df.groupby(["store", "item", "date"]).sum().reset_index()

df['date'] = pd.to_datetime(df['date'].astype(str))

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

df


#%% 6-b. Plot of the dependent variable versus time. Write down your observations.
#df['date'] = pd.to_datetime(df['date'].astype(str))

# plt.figure(figsize=(14, 6))
# plt.plot(df['sales'], label='Sales')
# # sns.lineplot(x="date", y="sales", label='Sales', data=df)
# plt.xlabel('Date')
# plt.ylabel('Sales')
# plt.legend(loc='best')
# plt.title('Sales Plot')
# plt.show()

# Aggregating sales by date
sales_by_date = df.groupby('date')['sales'].sum()

fig, ax = plt.subplots(figsize=(15,7))

# Plotting the data
ax.plot_date(sales_by_date.index, sales_by_date.values, '-')

# Formatting the y-axis to display in real values
formatter = ticker.FuncFormatter(lambda x, p: format(int(x), ','))
ax.yaxis.set_major_formatter(formatter)

# Setting the title and labels
ax.set_title('Sales by Date')
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
plt.show()

#%% 6-c. ACF/PACF of the dependent variable. Write down your observations.


def ACF_PACF_Plot(y, lags):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()
    return acf, pacf


ACF_PACF_Plot(df['sales'], 50)

# ARMA model
# There may be seasonality

#%% 6-d. Correlation Matrix with seaborn heatmap with the Pearson’s correlation coefficient. Write down your observations.

corr_matrix = df.corr(method='pearson') 

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix,
            cmap='coolwarm',
            fmt='.3f',
            annot=True,
            vmin=-1,
            vmax=1,
            linewidth=1)
plt.title("Correlation Matrix of Dataset", size=15)
plt.show()

# Not correlation betwwen features.

#%% 6-e. Split the dataset into train set (80%) and test set (20%).

# I will do this step right before modeling after checking whether dataset is stationary

#%% 7- Stationarity: 
# Check for a need to make the dependent variable stationary. If the dependent variable is not stationary, you need to use the techniques discussed in class to make it stationary. Perform ACF/PACF analysis for stationarity. You need to perform ADF-test & kpss-test and plot the rolling mean and variance for the raw data and the transformed data. Write down your observations.
def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
        
ADF_Cal(df['sales'])

# non-stationary vs stationary
# reject the null with low p-value which means stationary


def kpss_test(timeseries):
    print('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value
    print(kpss_output)


kpss_test(df['sales'])

# stationary vs non-stationary
# reject the null with low p-value (0.01) which means non-stationary

def rolling(data):
    n = len(data)
    rolling_mean = np.zeros(n)
    rolling_var = np.zeros(n)

    for i in range(n):
        rolling_mean[i] = np.mean(data[:i + 1])
        rolling_var[i] = np.var(data[:i + 1])

    # Plotting
    fig, axs = plt.subplots(2, figsize=(10, 5))

    axs[0].plot(rolling_mean)
    axs[0].set_title('Rolling Mean')
    axs[0].set_xlabel('Samples')
    axs[0].set_ylabel('Magnitude')
    axs[0].set_xlim(-50, n + 50)

    axs[1].plot(rolling_var, label='Varying variance')
    axs[1].set_title('Rolling Variance')
    axs[1].set_xlabel('Samples')
    axs[1].set_ylabel('Magnitude')
    axs[1].legend(loc='lower right')
    axs[1].set_xlim(-50, n + 50)

    plt.tight_layout()
    plt.show()

    # return rolling_mean, rolling_var

rolling(df['sales'])


#%%
# Transformation 

# df['log1'] = np.log(df['sales'])

# ADF_Cal(df['log1'].dropna())
# kpss_test(df['log1'].dropna())
# rolling(df['log1'])


df['diff_'] = df['sales'].diff()

# Nan value replace 
df['diff_'].fillna(method='bfill', inplace=True)

ADF_Cal(df['diff_'].dropna())
kpss_test(df['diff_'].dropna())
rolling(df['diff_'])

ACF_PACF_Plot(df['diff_'], 50)

#%% 8- Time series Decomposition: 
# Approximate the trend and the seasonality and plot the detrended and the seasonally adjusted data set using STL method. Find the out the strength of the trend and seasonality. Refer to the lecture notes for different type of time series decomposition techniques.

from statsmodels.tsa.seasonal import STL

STL = STL(df['diff_'], period=12)
res = STL.fit()

T = res.trend
S = res.seasonal
R = res.resid

# Plot the decomposition
fig, axes = plt.subplots(4, 1, figsize=(15, 10))
axes[0].plot(df['diff_'])
axes[0].set_title("Original Data")
T.plot(ax=axes[1], title="Trend Component")
S.plot(ax=axes[2], title="Seasonal Component")
R.plot(ax=axes[3], title="Residual Component")

plt.tight_layout()
plt.show()

# Calculate the strength of trend and seasonality
F_t = max(0, 1 - np.var(R) / np.var(T + R))
print(f'The strength of trend after diff is {F_t}')

F_s = max(0, 1 - np.var(R) / np.var(S + R))
print(f'The strength of seasonality after diff is {F_s}')




# Seasonally adjusted data and plot
df['diff'] = df['diff_'] - S

from statsmodels.tsa.seasonal import STL

STL = STL(df['diff'], period=12)
res = STL.fit()

T = res.trend
S = res.seasonal
R = res.resid

# Plot the decomposition
fig, axes = plt.subplots(4, 1, figsize=(15, 10))
axes[0].plot(df['diff'])
axes[0].set_title("Original Data")
T.plot(ax=axes[1], title="Trend Component")
S.plot(ax=axes[2], title="Seasonal Component")
R.plot(ax=axes[3], title="Residual Component")

plt.tight_layout()
plt.show()

# Calculate the strength of trend and seasonality
F_t = max(0, 1 - np.var(R) / np.var(T + R))
print(f'The strength of trend after adjusted seasonality is {F_t}')

F_s = max(0, 1 - np.var(R) / np.var(S + R))
print(f'The strength of seasonality after adjusted seasonality is {F_s}')

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['diff_'], label='Sales after differencing')
ax.plot(df['diff'], label='Seasonally Adjusted')
ax.set_title('Transformated Sales Data vs Seasonally Adjusted Data')
ax.set_xlabel('Sample')
ax.set_ylabel('Value')
ax.legend()
plt.show()

plot_acf(df['diff'])


# The strength of trend and seasonality are measures that help you understand the presence of a linear trend and seasonality patterns in your time series data. The values range from 0 to 1, where 0 indicates no trend or seasonality, and 1 indicates a strong trend or seasonality. In your case:

# Trend strength: 5.42e-07 (approximately)
# This value is extremely close to 0, suggesting that there is virtually no linear trend in your dataset. A value closer to 1 would indicate a stronger trend, while a value closer to 0 would indicate little to no trend. In your case, it appears that the trend is negligible.

# Seasonality strength: 0.4068 (approximately)
# This value indicates that there is a moderate seasonality in your dataset. A value closer to 1 would indicate strong seasonality, while a value closer to 0 would indicate little to no seasonality. In your case, seasonality is present and has a moderate impact on the data.

# These measures can help you decide which type of model to use for your time series analysis. If the trend and seasonality are strong, you may want to consider using models that can capture these patterns, such as seasonal ARIMA, Exponential Smoothing State Space Model (ETS), or other models that can handle both trend and seasonality components. On the other hand, if the trend and seasonality are weak or non-existent, you might consider using simpler models, such as ARIMA without a seasonal component.

# It's essential to consider the context of your problem and the specific goals of your project when interpreting these values and selecting an appropriate model.

#%% 6-e. Split the dataset into train set (80%) and test set (20%).
# from sklearn.model_selection import train_test_split

# y = df['adjsted_seasonal']
# X = df[['item', 'store_1', 'store_2', 'store_3', 'store_4', 'store_5', 'store_6', 'store_7', 'store_8', 'store_9', 'store_10']]

# # Split dataset for train and test

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# print(
#     f'X Train set size: {len(X_train)}, X Test set size: {len(X_test)}, y Train set size: {len(y_train)}, y Test set size: {len(y_test)}'
#     )
df.set_index('date', inplace=True)

# from sklearn.model_selection import train_test_split

# y = df['diff']
# X = df.drop(['sales','diff'], axis=1)

# # Split dataset for train and test

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

split_index = int(len(df) * 0.8)
train_df = df[:split_index]
test_df = df[split_index:]

X_train = train_df.drop(columns=['diff', 'sales', 'diff_'])
X_test = test_df.drop(columns=['diff', 'sales', 'diff_'])

y_train = train_df['diff']
y_test = test_df['diff']

print(
    f'X Train set size: {len(X_train)}, X Test set size: {len(X_test)}, y Train set size: {len(y_train)}, y Test set size: {len(y_test)}')


# split_index = int(len(df) * 0.8)

# # Split the data into training and testing sets
# df.drop(columns=['sales','log1', 'diff1'], inplace=True)

# train_model = df[:split_index]

# test_model = df[split_index:]

# print(
#     f'Train set size: {len(train_model)}, Test set size: {len(test_model)}')

# # Nan value replace 
# df['adjsted_seasonal'].fillna(method='bfill', inplace=True)

#%% 9- Holt-Winters method: 
# Using the Holt-Winters method try to find the best fit using the train dataset and make a prediction using the test set.

# Because Holt-Winter method can handle trend and seasonality, I used original dataset


# Fit the Holt-Winters model on the training data
hw_model = ExponentialSmoothing(y_train, seasonal_periods=12, trend=None, seasonal='add').fit()

hw_resid = hw_model.resid
# Make a prediction using the test set
hw_forecast = hw_model.forecast(steps=len(y_test))
# test_forecast = pd.Series(test_forecast, index=test_model.index)

# Evaluate the model's performance

hw_mse = mean_squared_error(y_test, hw_forecast)

print(f'Holt-Winter Model MSE: {hw_mse}')

# Plot the results
plt.figure(figsize=(10, 6))
# sns.lineplot(x=y_train.index, y=y_train.values, label='Train', data=y_train)
sns.lineplot(x=y_test.index, y=y_test.values, label='Test', data=y_test)
sns.lineplot(x=y_test.index, y=hw_forecast.values, label='Holt-Winters Forecast', data=hw_forecast)
plt.title('Holt-Winter Model Forecast')
# plt.plot(np.arange(len(y_train)), y_train, label='Train')
# plt.plot(np.arange(len(y_train),len(y_train)+len(y_test)), y_test, label='Test')
# plt.plot(np.arange(len(y_train),len(y_train)+len(y_test)), hw_forecast, label='Holt-Winters Forecast')
plt.xlabel('Time step')
plt.ylabel('Value')
plt.legend()
plt.show()

# sns.lineplot(x=y_train.index, y=y_train.values, label='Train')
# sns.lineplot(x=y_test.index, y=y_test.values, label='Test')
# sns.lineplot(x=y_test.index, y=hw_forecast.values, label='Holt-Winters Forecast')


#%% 10- Feature selection/elimination: 
# You need to have a section in your report that explains how the feature selection was performed and whether the collinearity exits not. Backward stepwise regression along with SVD and condition number is needed. You must explain that which feature(s) need to be eliminated and why. You are welcome to use other methods like VIF, PCA or random forest for feature elimination.

y = df['diff']
X = df.drop(['sales','diff', 'diff_'], axis=1)

# SVD analysis
H = X.T @ X
s, d, v = np.linalg.svd(H)
print ("Singular values:", d)

# The first singular value (2.5757e+07) is much larger than the others. This indicates that there is a strong linear relationship among your features, and most of the variation in the data can be captured by a single dimension. The remaining singular values are relatively smaller, which means that they capture less variation in the data.

# In the context of collinearity, the large difference between the maximum and minimum singular values can be an indication that there is a collinearity issue in the dataset. This is because collinearity occurs when some features are linearly dependent on others, and this dependency can be captured by a single dimension in the data.


# Condition nu,ber
cond_num = np.linalg.cond(X)
print ("Condition number of X:", {cond_num})
# The condition number is a measure of the sensitivity of the linear system to small changes in the input data. A high condition number indicates that the system is ill-conditioned, which means it is sensitive to small changes in the input data and might produce inaccurate results.

# In your case, the condition number of X is 188.14215327697448. While this value is not extremely high (e.g., values above 1000 are generally considered severely ill-conditioned), it is still relatively large, indicating that there might be some degree of multicollinearity present in the data. In practice, this means that some of the features in your dataset might be correlated, and you should be cautious when interpreting the results of your analysis, as the model might be sensitive to small changes in the data.

# To address the multicollinearity issue, you can consider using feature selection techniques, such as backward stepwise regression, Variance Inflation Factor (VIF), or Principal Component Analysis (PCA), to remove or reduce the correlated features. This can help to improve the stability and interpretability of your model.


def back_elimination(X, y):
    model = sm.OLS(y, sm.add_constant(X)).fit()
    aic = model.aic
    bic = model.bic
    adjstr2 = model.rsquared_adj
    features = list(X.columns)

    print(f'AIC: {aic}')
    print(f'BIC: {bic}')
    print(f'adjR2: {adjstr2}')
    print(f'***Baseline***')

    for f in features:
        model_ = sm.OLS(y, sm.add_constant(X.drop(columns=[f]))).fit()
        aic_ = model_.aic
        bic_ = model_.bic
        adjstr2_ = model_.rsquared_adj
        if aic_ < aic and bic_ < bic and adjstr2_ > adjstr2:  # good cond
            features.remove(f)
            print(f'***Dropped {f}***')
            print(f'AIC: {aic_}')
            print(f'BIC: {bic_}')
            print(f'adjR2: {adjstr2_}')
    return features


final_features = back_elimination(X_train, y_train)
print(final_features)


# VIF 

def calculate_vif(X):
    vif = pd.DataFrame()
    vif["features"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif

vif_df = calculate_vif(X_train)
print(vif_df)

vif_df_after = calculate_vif(X_train.drop(columns=['store', 'month'], axis=1))
print (vif_df_after)
# PCA


# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform PCA
pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Print the explained variance ratio
print("Explained variance ratio: ", pca.explained_variance_ratio_)


# In the analysis, we performed SVD, backward stepwise regression, VIF, and PCA for feature selection. Here's an interpretation of the results:

# SVD:
# The singular values from the Singular Value Decomposition (SVD) are shown, and the condition number of the matrix X is 188.14. A high condition number indicates a high degree of multicollinearity in the dataset. However, in this case, the condition number is not very high, suggesting that multicollinearity might not be a significant issue.

# Backward stepwise regression:
# Using backward elimination, we obtained the following model statistics: AIC is 26961.59, BIC is 27050.54, and the adjusted R-squared is 0.1886. The features included in the model are 'item' and all 'store' variables. The adjusted R-squared value suggests that the model explains around 18.86% of the variance in the target variable.

# VIF:
# The Variance Inflation Factor (VIF) analysis shows that all features have VIF values close to 1, which indicates low multicollinearity. It means that each feature provides unique information and should be considered for the model.

# PCA:
# The Principal Component Analysis (PCA) shows the explained variance ratio for each principal component. The first ten components explain approximately 10.1% of the variance each, while the last component explains a negligible amount. In total, the first ten components explain about 90.9% of the total variance.

# Interpretation and feature selection:
# Based on the SVD and VIF results, multicollinearity does not seem to be a significant issue in the dataset. Therefore, it is not necessary to remove any features based on these methods.

# The backward stepwise regression included all features in the model, which might be due to the low multicollinearity observed. However, the adjusted R-squared value is relatively low, so we might need to consider other modeling techniques or additional features to improve the model performance.

# The PCA results show that the first ten principal components explain the majority of the variance. If you want to reduce the number of features while retaining most of the information, you can consider using these ten principal components instead of the original features.

# In conclusion, there is no strong evidence to eliminate any features based on the performed feature selection methods. You can proceed with the current set of features or use the first ten principal components from PCA for further modeling.

#%% 11- Base-models: 
# average, naïve, drift, simple and exponential smoothing. You need to perform an h-step prediction based on the base models and compare the SARIMA model py_foreerformance with the base model predication.


# # Define function for h-step forecasting
# def h_step_forecast(method, y_train, y_test, alpha=None):
#     y_forecast = np.zeros_like(y_test).astype(float)
#     if method == "Average":
#         y_forecast = np.mean(y_train) * np.ones_like(y_test)
#     elif method == "Naive":
#         y_forecast = y_train[-1] * np.ones_like(y_test)
#     elif method == "Drift":
#         y_forecast = [y_train[-1] + i * ((y_train[-1] - y_train[0]) / (len(y_train) - 1)) for i in range(1, len(y_test) + 1)]
#     elif method == "SES":
#         ses_forecast = [y_train[0]]
#         for i in range(1, len(y_train)):
#             ses_forecast.append(alpha * y_train[i - 1] + (1 - alpha) * ses_forecast[-1])
#         y_forecast = [alpha * y_train[-1] + (1 - alpha) * ses_forecast[-1] for _ in range(len(y_test))]
#     else:
#         print("Invalid method. Choose between 'Average', 'Naive', 'Drift', and 'SES'.")
#         return None
#     error = y_test - y_forecast
#     MSE = np.mean(error**2)
#     return y_forecast, MSE

# # Calculate forecasts and MSEs for different methods
# methods = ['Average', 'Naive', 'Drift', 'SES']
# alpha = 0.5
# for method in methods:
#     forecast, MSE = h_step_forecast(method, y_train, y_test, alpha)
#     print(f"{method} h-step ahead prediction MSE is {MSE}")





# Average

# h-step forecast by average method
avg_y_forecast = np.zeros_like(y_test).astype(float)
for i in range(len(y_test)):
    avg_y_forecast[i] = np.mean(y_train)

# h step error
avg_error_hstep = y_test[:] - avg_y_forecast

avg_sqrerror_hstep = avg_error_hstep ** 2

# h-step MSE
avg_MSE_hstep = avg_sqrerror_hstep.sum() / len(y_test)
print(f'Average h-step ahead predction MSE is {avg_MSE_hstep}')

# # Plot
# plt.figure(figsize=(15,8))
# plt.plot(np.arange(len(y_train)), y_train, 'bo-', label='Training set')
# plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test, 'gs-', label='Test set')
# plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), avg_y_forecast, 'r+-', label='h-step forecast')
# plt.title('Average Forecast Method')
# plt.xlabel('Time step')
# plt.ylabel('Value')
# plt.legend()
# plt.grid()
# plt.show()

# Naive
# h-step forecast
naive_y_forecast = np.zeros_like(y_test).astype(float)
for i in range(len(y_test)):
    naive_y_forecast[i] = y_train[len(y_train)-1]

# h step error
naive_error_hstep = y_test[:] - naive_y_forecast

naive_sqrerror_hstep = naive_error_hstep ** 2

# h-step MSE
naive_MSE_hstep = naive_sqrerror_hstep.sum() / len(y_test)
print(f'Naive h-step ahead predction MSE is {naive_MSE_hstep}')

# # Plot
# plt.figure(figsize=(15,8))
# plt.plot(np.arange(len(y_train)), y_train, 'bo-', label='Training set')
# plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test, 'gs-', label='Test set')
# plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), naive_y_forecast, 'r+-', label='h-step forecast')
# plt.title('Naive Forecast Method')
# plt.xlabel('Time step')
# plt.ylabel('Value')
# plt.legend()
# plt.grid()
# plt.show()

# Drift

# h-step forecast
drift_y_forecast = np.zeros_like(y_test).astype(float)
for i in range(1, len(y_test) + 1):
    drift_y_forecast[i - 1] = y_train[len(y_train)-1] + i * ((y_train[len(y_train)-1] - y_train[0]) / (len(y_train) - 1))

# h step error
drift_error_hstep = y_test[:] - drift_y_forecast

drift_sqrerror_hstep = drift_error_hstep ** 2


# h-step MSE
drift_MSE_hstep = drift_sqrerror_hstep.sum() / len(y_test)
print(f'Drift h-step ahead predction MSE is {drift_MSE_hstep}')

# # Plot
# plt.figure(figsize=(15,8))
# plt.plot(np.arange(len(y_train)), y_train, 'bo-', label='Training set')
# plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test, 'gs-', label='Test set')
# plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), drift_y_forecast, 'r+-', label='h-step forecast')
# plt.title('Drift Forecast Method')
# plt.xlabel('Time step')
# plt.ylabel('Value')
# plt.legend()
# plt.grid()
# plt.show()

# Simple Exponential Smoothing

# 1-step prediction
alpha = 0.5
ses5_forecast = np.zeros(len(y_train))
ses5_forecast[0] = y_train[0]

for i in range(1, len(y_train)):
    ses5_forecast[i] = alpha * y_train[i - 1] + (1 - alpha) * ses5_forecast[i - 1]

# h-step forecast
ses5_y_forecast = np.zeros_like(y_test).astype(float)
for i in range(1, len(y_test) + 1):
    ses5_y_forecast[i - 1] = alpha * y_train[len(y_train)-1] + (1 - alpha) * ses5_forecast[-1]

# h step error
ses5_error_hstep = y_test[:] - ses5_y_forecast

ses5_sqrerror_hstep = ses5_error_hstep ** 2


# h-step MSE
ses5_MSE_hstep = ses5_sqrerror_hstep.sum() / len(y_test)
print(f'SES h-step ahead predction MSE is {ses5_MSE_hstep}')

# # Plot
# plt.figure(figsize=(15,8))
# plt.plot(np.arange(len(y_train)), y_train, 'bo-', label='Training set')
# plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test, 'gs-', label='Test set')
# plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), ses5_y_forecast, 'r+-', label='h-step forecast')
# plt.title('SES Forecast Method w/ alpha = 0.5')
# plt.xlabel('Time step')
# plt.ylabel('Value')
# plt.legend()
# plt.grid()
# plt.show()

fig, axs = plt.subplots(2, 2, figsize=(20, 12))
fig.suptitle('Basic Method Forecast', fontsize=25)

# Average Forecast Method
axs[0, 0].plot(np.arange(len(y_train)), y_train, 'bo-', label='Training set')
axs[0, 0].plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test, 'gs-', label='Test set')
axs[0, 0].plot(np.arange(len(y_train), len(y_train) + len(y_test)), avg_y_forecast, 'r+-', label='h-step forecast')
axs[0, 0].set_title('Average Forecast Method', fontsize=16)
axs[0, 0].set_xlabel('Time step')
axs[0, 0].set_ylabel('Value')
axs[0, 0].legend()
axs[0, 0].grid()

# Naive Forecast Method
axs[0, 1].plot(np.arange(len(y_train)), y_train, 'bo-', label='Training set')
axs[0, 1].plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test, 'gs-', label='Test set')
axs[0, 1].plot(np.arange(len(y_train), len(y_train) + len(y_test)), naive_y_forecast, 'r+-', label='h-step forecast')
axs[0, 1].set_title('Naive Forecast Method', fontsize=16)
axs[0, 1].set_xlabel('Time step')
axs[0, 1].set_ylabel('Value')
axs[0, 1].legend()
axs[0, 1].grid()

# Drift Forecast Method
axs[1, 0].plot(np.arange(len(y_train)), y_train, 'bo-', label='Training set')
axs[1, 0].plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test, 'gs-', label='Test set')
axs[1, 0].plot(np.arange(len(y_train), len(y_train) + len(y_test)), drift_y_forecast, 'r+-', label='h-step forecast')
axs[1, 0].set_title('Drift Forecast Method', fontsize=16)
axs[1, 0].set_xlabel('Time step')
axs[1, 0].set_ylabel('Value')
axs[1, 0].legend()
axs[1, 0].grid()

# SES Forecast Method w/ alpha = 0.5
axs[1, 1].plot(np.arange(len(y_train)), y_train, 'bo-', label='Training set')
axs[1, 1].plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test, 'gs-', label='Test set')
axs[1, 1].plot(np.arange(len(y_train), len(y_train) + len(y_test)), ses5_y_forecast, 'r+-', label='h-step forecast')
axs[1, 1].set_title('SES Forecast Method w/ alpha = 0.5', fontsize=16)
axs[1, 1].set_xlabel('Time step')
axs[1, 1].set_ylabel('Value')
axs[1, 1].legend()
axs[1, 1].grid()

plt.tight_layout()
plt.show()

# After doing SARIMA, I will compare all these models.



#%% 12- Develop the multiple linear regression model that represent the dataset. Check the accuracy of the developed model.



# features : all

model = sm.OLS(y_train, sm.add_constant(X_train))
modelfit = model.fit()


y_pred = modelfit.predict(sm.add_constant(X_test))

# Calculate the mean squared error and R-squared score
ols_mse = mean_squared_error(y_test, y_pred)
print(f'All features MSE: {ols_mse:.2f}')


# 12-b. Hypothesis tests analysis: F-test, t-test.

modelresult = modelfit.summary()
modelresult



# features : features selection
model_store = sm.OLS(y_train, sm.add_constant(X_train[final_features]))
modelfit = model_store.fit()


# 12-a. You need to include the complete regression analysis into your report. Perform one-step ahead prediction and compare the performance versus the test set.

y_pred = modelfit.predict(sm.add_constant(X_test[final_features]))

# Calculate the mean squared error and R-squared score
mse = mean_squared_error(y_test, y_pred)
print(f'Feature Selection MSE: {mse:.2f}')

# 12-b. Hypothesis tests analysis: F-test, t-test.

modelresult = modelfit.summary()
modelresult


plt.figure(figsize=(14, 6))
# sns.lineplot(x=y_train.index, y=y_train.values, label='Train', data=y_train)
sns.lineplot(x=y_test.index, y=y_test.values, label='Test', data=y_test)
plt.scatter(x=y_pred.index, y=y_pred.values, label='Predicted')
plt.xlabel('Time step')
plt.ylabel('Value')
plt.legend(loc='best')
plt.title('Test and Predicted Plot')
plt.legend()
plt.show()

f_test = modelfit.f_pvalue
t_test = modelfit.tvalues

print (f'F-test: {f_test}')
print (f't-test: \n{t_test}')

# 12-d. ACF of residuals.
residuals = y_test - y_pred

plot_acf(residuals, lags=50)

# 12-e. Q-value

q_value = acorr_ljungbox(residuals, lags=10, return_df=True)
print(f"Q value: \n{q_value}")



# p value is more than good


# 12-f. Variance and mean of the residuals.
residuals_variance = np.var(residuals)
residuals_mean = np.mean(residuals)

print(f"Residuals Variance: {residuals_variance:.2f}")
print(f"Residuals Mean: {residuals_mean:.2f}")


#%% 13- ARMA and ARIMA and SARIMA model order determination: Develop an ARMA, ARIMA and SARIMA model that represent the dataset.


ACF_PACF_Plot(y_train, 25)

#AR(12)
#ARMA(12,0)
#ARIMA(12,0,0)
#SARIMA(1,0,0,12)

# Based on the ACF and PACF plots you provided, it appears that there may be a significant autocorrelation in the first lag and some correlation at the seasonal lags (multiples of 12).

# For ARMA modeling, I can start with a simple AR(1) model, as there is a significant autocorrelation at the first lag in the ACF. I can also try adding an MA term if there is still significant autocorrelation at the lags where the PACF crosses the significance threshold.

# For ARIMA modeling, I can start with an ARIMA(1,1,0) model, which is an AR(1) model applied to the first difference of the time series, since there is some indication of a trend. You can also try including a seasonal difference and a seasonal AR term if there is still significant correlation at the seasonal lags.

# For SARIMA modeling, you can start with a SARIMA(1,1,0)(1,0,0,12) model, which includes an AR(1) term, a first difference, and a seasonal AR term at lag 12, since there is some indication of seasonal correlation at the lag 12. You can also try including other AR and MA terms at non-seasonal lags and a seasonal MA term if there is still significant correlation at the seasonal lags.

# Note that these are just initial suggestions and you may need to try different model orders to find the best one that fits your data. It's also important to check the residuals for any remaining autocorrelation or other patterns that the model may have missed.

def GPAC(ry, j=7, k=7):
    c = len(ry) // 2
    gpac_table = np.zeros((j,k-1))

    for i in range(j):
        for l in range(1, k):
            
            den_matrix = np.zeros((l,l))
            for row in range(l):
                den_matrix[row] = ry2[c - i - row : c - i + l - row]
            
            num_matrix = den_matrix.copy().T
            num_matrix[-1] = ry2[c + i + 1 : c + i + 1 + l]
            num_matrix = num_matrix.T

            phi = np.linalg.det(num_matrix) / np.linalg.det(den_matrix)
            
            if num_matrix.shape[0] == num_matrix.shape[1] and den_matrix.shape[0] == den_matrix.shape[1]:
                num = np.linalg.det(num_matrix)
                den = np.linalg.det(den_matrix)
                if den != 0:
                    gpac_table[i, l-1] = num / den
                else:
                    gpac_table[i, l-1] = np.nan
    
    plt.figure(figsize=(20, 12))
    # Create a Seaborn heatmap
    sns.heatmap(gpac_table, 
                mask = np.isnan(gpac_table), 
                fmt = ".4f", 
                cmap = 'Reds_r', 
                annot = True,
                vmin = -1,
                vmax = 1,
                linewidth = 1)
    plt.xticks(np.arange(0.5, (k-1)+0.5, 1), np.arange(1, k, 1))
    plt.title("GPAC Table", size = 15)
    plt.show()
    
    return gpac_table



ry = acf(y_train, nlags=50)
ry1 = ry[::-1]
ry2 = np.concatenate((ry1, ry[1:]))

GPAC(ry2,15,15)

# 13-a. Preliminary model development procedures and results. (ARMA model order determination). Pick at least two orders using GPAC table.
# 13-b. Should include discussion of the autocorrelation function and the GPAC. Include a plot of the autocorrelation function and the GPAC table within this section).
# 13-c. Include the GPAC table in your report and highlight the estimated order.

# ARIMA (12,0,0)
# ARIMA (13,0,1)
# ARIMA (12,0,12)
# SARIMA (0,0,0)(1,0,1,12)
# SARIMA (1,0,1)(1,0,1,12)
# SARIMA (2,0,2)(1,0,1,12)



#%%

# ARIMA (12,0,0)
arima120_ = ARIMA(y_train, order = (12,0,0))
arima120_fit = arima120_.fit()
print (arima120_fit.summary())

# ARIMA (13,0,1)
arima131_ = ARIMA(y_train, order = (13,0,1))
arima131_fit = arima131_.fit()
print (arima131_fit.summary())

# ARIMA (12,0,12)
arima1212_ = ARIMA(y_train, order = (12,0,12))
arima1212_fit = arima1212_.fit()
print (arima1212_fit.summary())


print("ARIMA(12, 0, 0) AIC:", arima120_fit.aic)
print("ARIMA(13, 0, 1) AIC:", arima131_fit.aic)
print("ARIMA(12, 0, 12) AIC:", arima1212_fit.aic)

print("ARIMA(12, 0, 0) BIC:", arima120_fit.bic)
print("ARIMA(13, 0, 1) BIC:", arima131_fit.bic)
print("ARIMA(12, 0, 12) BIC:", arima1212_fit.bic)

arima120_pred = arima120_fit.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)
arima120_mse = mean_squared_error(y_test, arima120_pred)

arima131_pred = arima131_fit.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)
arima131_mse = mean_squared_error(y_test, arima131_pred)

arima1212_pred = arima1212_fit.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)
arima1212_mse = mean_squared_error(y_test, arima1212_pred)

print("ARIMA(12, 0, 0) MSE:", arima120_mse)
print("ARIMA(13, 0, 1) MSE:", arima131_mse)
print("ARIMA(12, 0, 12) MSE:", arima1212_mse)

ACF_PACF_Plot(arima120_fit.resid, lags=24)
ACF_PACF_Plot(arima131_fit.resid, lags=24)
ACF_PACF_Plot(arima1212_fit.resid, lags=24)

# Selected ARIMA (12,0,12)
arima_resid = arima1212_fit.resid


# SARIMA (0,0,0)(1,0,1,12)
sarima1_ = SARIMAX(y_train, order = (0,0,0), seasonal_order = (1,0,1,12))
sarima1_fit = sarima1_.fit()
print (sarima1_fit.summary())


# SARIMA (1,0,1)(1,0,1,12)
sarima2_ = SARIMAX(y_train, order = (1,0,1), seasonal_order = (1,0,1,12))
sarima2_fit = sarima2_.fit()
print (sarima2_fit.summary())

# SARIMA (2,0,2)(1,0,1,12)
sarima3_ = SARIMAX(y_train, order = (2,0,2), seasonal_order = (1,0,1,12))
sarima3_fit = sarima3_.fit()
print (sarima3_fit.summary())



print("SARIMA(0,0,0)(1, 0, 1, 12) AIC:", sarima1_fit.aic)
print("SARIMA(0,0,0)(1, 0, 1, 12) AIC:", sarima1_fit.bic)

sarima1_pred = sarima1_fit.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)
sarima1_mse = mean_squared_error(y_test, sarima1_pred)

print("SARIMA(0,0,0)(1, 0, 1, 12) MSE:", sarima1_mse)


print("SARIMA(1,0,1)(1, 0, 1, 12) AIC:", sarima2_fit.aic)
print("SARIMA(1,0,1)(1, 0, 1, 12) AIC:", sarima2_fit.bic)

sarima2_pred = sarima2_fit.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)
sarima2_mse = mean_squared_error(y_test, sarima2_pred)

print("SARIMA(1,0,1)(1, 0, 1, 12) MSE:", sarima2_mse)


print("SARIMA(2,0,2)(1, 0, 1, 12) AIC:", sarima3_fit.aic)
print("SARIMA(2,0,2)(1, 0, 1, 12) AIC:", sarima3_fit.bic)

sarima3_pred = sarima3_fit.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)
sarima3_mse = mean_squared_error(y_test, sarima3_pred)

print("SARIMA(2,0,2)(1, 0, 1, 12) MSE:", sarima3_mse)

ACF_PACF_Plot(sarima1_fit.resid, lags=24)
ACF_PACF_Plot(sarima2_fit.resid, lags=24)
ACF_PACF_Plot(sarima3_fit.resid, lags=24)



# SARIMA (1,0,1)(1,0,1,12)
sarima_resid = sarima2_fit.resid

#%% 14- Estimate ARMA model parameters using the Levenberg Marquardt algorithm. Display the parameter estimates, the standard deviation of the parameter estimates and confidence intervals.

# ARIMA (12,0,12)
arima_params = arima1212_fit.params
arima_std = arima1212_fit.bse
arima_ci = arima1212_fit.conf_int()

print(f"Coefficients: \n{arima_params}")
print(f"\nStandard Errors: \n{arima_std}")
print(f"\nConfidence Intervals: \n{arima_ci}")



sarima_params = sarima2_fit.params
sarima_std = sarima2_fit.bse
sarima_ci = sarima2_fit.conf_int()

print(f"Coefficients: \n{sarima_params}")
print(f"\nStandard Errors: \n{sarima_std}")
print(f"\nConfidence Intervals: \n{sarima_ci}")

# SARIMA
#%% 15- Diagnostic Analysis: Make sure to include the followings:
# 15-a. Diagnostic tests (confidence intervals, zero/pole cancellation, chi-square test).

# 15-b. Display the estimated variance of the error and the estimated covariance of the estimated parameters.

# ARIMA(12,0,12)
print(f'The estimated variance of error for ARIMA(12,0,12): \n{arima_resid.var()}')
arima_cov_theta_hat = arima1212_fit.cov_params()
print(f'The covariance for ARIMA(12,0,12): \n{arima_cov_theta_hat}')

# SARIMA(1,0,1)(1,0,1,12)
print(f'The estimated variance of error for SARIMA(1,0,1)(1,0,0,12): \n{sarima_resid.var()}')
sarima_cov_theta_hat = sarima2_fit.cov_params()
print(f'The covariance for SARIMA(1,0,1)(1,0,0,12): \n{sarima_cov_theta_hat}')

# ARIMA 

#%%
# 15-c. Is the derived model biased or this is an unbiased estimator?

# Mean of ARIMA (12,0,12)
arima_bias = np.mean(arima_resid)
print(f"Mean of arima_Residuals is {arima_bias}\n")



# SARIMA (1,0,1)(1,0,1,12)

# Mean of SARIMA
sarima_bias = np.mean(sarima_resid)
print(f"Mean of sarima_Residuals is {sarima_bias}\n")



# ARIMA

#%%
# 15-d. Check the variance of the residual errors versus the variance of the forecast errors.

# ARIMA(12,0,12)
arima_forecast = arima1212_fit.forecast(steps=len(y_test))

# Calculate the variance of the forecast errors and the residual errors
forecast_errors_variance = np.var(y_test.values - arima_forecast)
residual_errors_variance = np.var(arima_resid)

print("Variance of the ARIMA forecast errors:", forecast_errors_variance)
print("Variance of the ARIMA residual errors:", residual_errors_variance)

# # Plot the 50-step predicted values versus the test set
# plt.figure(figsize=(10, 6))
# plt.plot(np.arange(len(y_train),len(y_train)+len(y_test))[:100],y_test.values[:100], label='test')
# plt.plot(np.arange(len(y_train),len(y_train)+len(y_test))[:100],arima_forecast[:100], label="Predicted Values")
# plt.xlabel("Sample")
# plt.ylabel("Value")
# plt.title("Predicted Values vs Test Set")
# plt.legend()
# plt.show()



# SARIMA (1,0,1)(1,0,0,12)
sarima_forecast = sarima2_fit.forecast(steps=len(y_test))

# Calculate the variance of the forecast errors and the residual errors
forecast_errors_variance = np.var(y_test.values - sarima_forecast)
residual_errors_variance = np.var(sarima_resid)

print("Variance of the SARIMA forecast errors:", forecast_errors_variance)
print("Variance of the SARIMA residual errors:", residual_errors_variance)



# Comparing the forecast error variances, the SARIMA model has a slightly lower value (13,349.45) than the ARIMA model (13,360.26). This means that the SARIMA model is providing marginally more accurate predictions than the ARIMA model.

# However, when comparing the residual error variances, the ARIMA model has a lower value (5,390.83) compared to the SARIMA model (5,564.54). This indicates that the ARIMA model has smaller residuals or unexplained variation in the data than the SARIMA model.

# 15-e. If you find out that the ARIMA or SARIMA model may better represents the dataset, then you can find the model accordingly. You are not constraint only to use of ARMA model. Finding an ARMA model is a minimum requirement and making the model better is always welcomed.


# SARIMA

#%% 16- Deep Learning Model: 
# Fit the dataset into multivariate LSTM model. You also need to perform h- step prediction using LSTM model. You can use tensorflow package in python for this section.

def scale_data(train_set,test_set):
    """Scales data using MinMaxScaler and separates data into X_train, y_train,
    X_test, and y_test."""
    
    # Apply Min Max Scaler:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train_set)
    
    train_array = train_set.values
    test_array = test_set.values

    # Reshape training set:
    train_set = train_array.reshape(train_array.shape[0], train_array.shape[1])
    train_set_scaled = scaler.transform(train_array)
    
    # Reshape test set:
    test_set = test_array.reshape(test_array.shape[0], 
                                test_array.shape[1])
    test_set_scaled = scaler.transform(test_array)
    
    X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1].ravel() # returns the array, flattened!
    X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1].ravel()
    
    return X_train, y_train, X_test, y_test, scaler

X_train, y_train, X_test, y_test, scaler_object = scale_data(train_df, test_df)
print(f"Shape of X Train: {X_train.shape}\nShape of y Train: {y_train.shape}\nShape of X Test: {X_test.shape}\nShape of y Test: {y_test.shape}")

def re_scaling(y_pred, x_test, scaler_obj, lstm=False):
    """For visualizing and comparing results, undoes the scaling effect on predictions."""
   # y_pred: model predictions
   # x_test: features from the test set used for predictions
   # scaler_obj: the scaler objects used for min-max scaling
   # lstm: indicate if the model run is the lstm. If True, additional transformation occurs 
    
    # Reshape y_pred:
    y_pred = y_pred.reshape(y_pred.shape[0],
                            1,
                            1)

    if not lstm:
        x_test = x_test.reshape(x_test.shape[0],
                                1, 
                                x_test.shape[1])

    # Rebuild test set for inverse transform:
    pred_test_set = []
    for index in range(0, len(y_pred)):
        pred_test_set.append(np.concatenate([y_pred[index], 
                                             x_test[index]],
                                             axis=1) )

    # Reshape pred_test_set:
    pred_test_set = np.array(pred_test_set)
    pred_test_set = pred_test_set.reshape(pred_test_set.shape[0],
                                          pred_test_set.shape[2])

    # Inverse transform:
    pred_test_set_inverted = scaler_obj.inverse_transform(pred_test_set)

    return pred_test_set_inverted

def prediction_df(unscale_predictions, origin_df):
    """Generates a dataframe that shows the predicted sales for each month
    for plotting results."""
    
    # unscale_predictions: the model predictions that do not have min-max or other scaling applied
    # origin_df: the original monthly sales dataframe
    
    # Create dataframe that shows the predicted sales:
    result_list = []
    sales_dates = list(origin_df[-13:].date)
    act_sales = list(origin_df[-13:].sales)

    for index in range(0, len(unscale_predictions)):
        result_dict = {}
        result_dict['pred_value'] = int(unscale_predictions[index][0] + act_sales[index])
        result_dict['date'] = sales_dates[index + 1]
        result_list.append(result_dict)

    df_result = pd.DataFrame(result_list)

    return df_result

model_scores = {}


def get_scores(unscale_df, origin_df, model_name):
    """Prints the root mean squared error, mean absolute error, and r2 scores
    for each model. Saves all results in a model_scores dictionary for
    comparison."""
    
    rmse = np.sqrt(mean_squared_error(origin_df.sales[-12:], 
                                      unscale_df.pred_value[-12:]))
    
    mae = mean_absolute_error(origin_df.sales[-12:], 
                              unscale_df.pred_value[-12:])
    
    r2 = r2_score(origin_df.sales[-12:], 
                  unscale_df.pred_value[-12:])
    
    model_scores[model_name] = [rmse, mae, r2]

    print(f"RMSE: {rmse}\nMAE: {mae}\nR2 Score: {r2}")


def plot_results(results, origin_df, model_name):
# results: a dataframe with unscaled predictions

    fig, ax = plt.subplots(figsize=(15,5))
    sns.lineplot(origin_df.date, origin_df.sales, data=origin_df, ax=ax, 
                 label='Original', color='blue')
    sns.lineplot(results.date, results.pred_value, data=results, ax=ax, 
                 label='Predicted', color='red')
    
    
    ax.set(xlabel = "Date",
           ylabel = "Sales",
           title = f"{model_name} Sales Forecasting Prediction")
    
    ax.legend(loc='best')



def lstm_model(train_data, test_data):
    """Runs a long-short-term-memory neural net with 2 dense layers. 
    Generates predictions that are then unscaled. 
    Scores are printed and the results are plotted and saved."""
    # train_data: dataset used to train the model
    # test_data: dataset used to test the model
   
    
    # Split into X & y and scale data:
    X_train, y_train, X_test, y_test, scaler_object = scale_data(train_data, test_data)
    
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
   
    
    # Build LSTM:
    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), 
                   stateful=True))
    model.add(Dense(1))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=1, 
              shuffle=False)
    predictions = model.predict(X_test,batch_size=1)
    
    # Undo scaling to compare predictions against original data:
    origin_df = df['sales']
    unscaled = re_scaling(predictions, X_test, scaler_object, lstm=True)
    unscaled_df = prediction_df(unscaled, origin_df)
    
    get_scores(unscaled_df, origin_df, 'LSTM')
    plot_results(unscaled_df, origin_df, 'LSTM')
    
lstm_model(train_df,test_df)


#%% 17- Final Model selection: 
# There should be a complete description of why your final model was picked over base-models ARMA, ARIMA, SARIMA and LSTM. You need to compare the performance of various models developed for your dataset and come up with the best model that represent the dataset the best.

# MSE 

print(f'Average h-step ahead predction MSE: {avg_MSE_hstep}')
print(f'Naive h-step ahead predction MSE: {naive_MSE_hstep}')
print(f'Drift h-step ahead predction MSE: {drift_MSE_hstep}')
print(f'SSE 0.05 h-step ahead predction MSE: {ses5_MSE_hstep}')

print(f'Multiple Linear Regression MSE: {ols_mse}')

print("ARIMA(12, 0, 12) MSE:", arima1212_mse)
print("SARIMA(1,0,1)(1, 0, 1, 12) MSE:", sarima2_mse)


# From mse -> SARIMA

# linear regreassion 이 제일 낮지만, 타임 시리즈 데이터셋의 경우, 시간에 따른 데이터 포인트들의 순서가 중요하며, 이를 고려하는 예측 모델을 사용하는 것이 좋습니다. 주어진 모델들 중 ARIMA(12, 0, 0)와 SARIMA(1, 0, 0, 12)는 타임 시리즈 데이터에 특화된 모델입니다.

# 위에서 계산한 MSE 값들을 고려할 때, Multiple Linear Regression 모델이 가장 낮은 MSE를 가지지만, 타임 시리즈 데이터에 대한 다양한 패턴(계절성, 추세 등)을 반영하지 않을 수 있습니다.

# ARIMA(12, 0, 0)와 SARIMA(1, 0, 0, 12) 모델은 타임 시리즈 데이터에 대한 패턴을 포착하고 예측하는 데 더 적합합니다. 이러한 모델들의 성능은 Multiple Linear Regression에 비해 다소 낮지만, 실제로는 데이터의 패턴을 더 잘 반영할 수 있습니다.

# 따라서, 타임 시리즈 데이터에 대해 ARIMA(12, 0, 0) 또는 SARIMA(1, 0, 0, 12) 모델을 사용하는 것이 더 적합할 수 있습니다. 이 두 모델의 성능이 비슷하므로, 계산 복잡성이 더 낮은 SARIMA(1, 0, 0, 12) 모델을 선택하는 것이 좋을 수 있습니다. 그러나 최종 결정은 여러 가지 요소와 데이터의 특성을 고려하여 내릴 필요가 있습니다.

#%%

# AIC -> SARIMA


print(f'Multiple Linear Regression AIC: {modelfit.aic}')

print("ARIMA(12, 0, 12) AIC:", arima1212_fit.aic)
print("SARIMA(1,0,1)(1, 0, 1, 12) AIC:", sarima2_fit.aic)




# BIC -> SARIMA
print(f'Multiple Linear Regression BIC: {modelfit.bic}')

print("ARIMA(12, 0, 12) BIC:", arima1212_fit.bic)
print("SARIMA(1,0,1)(1, 0, 1, 12) AIC:", sarima2_fit.bic)



# ACF PACF Plot


ACF_PACF_Plot(avg_error_hstep, lags=24)
ACF_PACF_Plot(naive_error_hstep, lags=24)
ACF_PACF_Plot(drift_error_hstep, lags=24)
ACF_PACF_Plot(ses5_error_hstep, lags=24)

ols_resid = modelfit.resid
ACF_PACF_Plot(ols_resid, lags=24)

ACF_PACF_Plot(arima_resid, lags=24)
ACF_PACF_Plot(sarima_resid, lags=24)

# SARIMA (1,0,0,12)
#%%
# 18- Forecast function: 
# Once the final mode is picked (SARIMA), the forecast function needs to be developed and included in your report.

# 19- h-step ahead Predictions: 
# You need to make a multiple step ahead prediction for the duration of the test data set. Then plot the predicted values versus the true value (test set) and write down your observations.


# Plot the h-step predicted values versus the test set
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(y_train),len(y_train)+len(y_test))[:100], y_test.values[:100], label='test')
plt.plot(np.arange(len(y_train),len(y_train)+len(y_test))[:100], sarima_forecast[:100], label="Predicted Values")
plt.xlabel("Sample")
plt.ylabel("Value")
plt.title("SARIMA Model: 100-step ahead Predicted Values vs Test Set")
plt.legend()
plt.show()


# %%
