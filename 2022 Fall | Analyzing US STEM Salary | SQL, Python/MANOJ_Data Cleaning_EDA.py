#%% [markdown]
#
# # 6103: Intro to Data Mining - Final Project
# ## By: Team 02
# ### Topic: Data Science and STEM Salaries
#

#%%
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rfit 
import plotly.express as px

#%%
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)

#%%
old_df=pd.read_csv('Levels_Fyi_Salary_Data.csv')

print(old_df.info())
old_df.head()

#%%
df = old_df[['timestamp','company','title','location','yearsofexperience','yearsatcompany']].copy()

df['yearlysalary'] = old_df['totalyearlycompensation'] + old_df['basesalary']

df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y')

# %%
df.to_csv('salary_data', index=False)
df.head()

#%%
df.describe()

#%%
df['title'].value_counts()


#%%
top_companies = df["company"].value_counts().head(10)
plt.figure(figsize = (7,5))
fig = px.bar(x = top_companies.index, y = top_companies.values,
            labels = {"x" : "Company" , "y" : "Number of Jobs" })
fig.show()

#%%
## Smart Question 1
sns.boxplot(data=df,x = 'timestamp', y = 'yearlysalary')
plt.title("timestamp vs yearlysalary")
plt.ylim(100000,1000000,50000)

# %%
# Smart Question 2
sns.boxplot(data=df,x = 'title', y = 'yearlysalary')
plt.title("Title vs Yearlysalary")
plt.xticks(rotation='vertical')

#%%
# Smart Question 4
sns.scatterplot(data=df,x = 'yearsofexperience',y ='yearlysalary')
plt.title("YearosExperience vs Yearlysalary")
plt.xticks(rotation='vertical')

# %%
# Smart Question 4
sns.barplot(data=df,x = 'yearsofexperience',y ='yearlysalary')
plt.title("YearofExperience vs Yearlysalary")
plt.xticks(rotation='vertical')
plt.ylim(100000,1000000,50000)
plt.xlim(0,60,5)


# %%
# Smart Question 4
sns.lineplot(data=df,x = 'yearsofexperience',y ='yearlysalary')
plt.title("YearofExperience vs Yearlysalary")
plt.xticks(rotation='vertical')
plt.ylim(100000,1000000,50000)
plt.xlim(0,60,5)

# %%
# Smart Question 5
sns.scatterplot(data=df,x = 'yearsatcompany',y ='yearlysalary')
plt.title("Yearatcompany vs Yearlysalary")
plt.xticks(rotation='vertical')
plt.ylim(100000,1000000,50000)
plt.xlim(0,60,5)


#%%
## Additional Smart Questions
sns.countplot(data=df,x = 'timestamp')
plt.title("Time Stamp")


# %%
plt.figure(figsize=(12,8))
df["location"].value_counts().iloc[:10].plot(kind="bar", color="blue")
plt.title("Top 10 Locations of Workers")
plt.xlabel("Location", size=23)
plt.ylabel("Count", size=23)
plt.xticks(rotation='vertical')
plt.show()

#%%[markdown]
# We consider this below part for EDA but these columns have lot of NA values. 


# %%
# Additional Smart Question
sns.barplot(data=old_df,x = 'title',y ='totalyearlycompensation',hue='Bachelors_Degree')
plt.xticks(rotation='vertical')

# %%
# Additional Smart Question
sns.barplot(data=df,x = 'title',y ='totalyearlycompensation',hue='Masters_Degree')
plt.xticks(rotation='vertical')

# %%
# Additional Smart Question
sns.barplot(data=df,x = 'title',y ='totalyearlycompensation',hue='Doctorate_Degree')
plt.xticks(rotation='vertical')

#%%
## Additional Smart Questions
sns.boxplot(data=df,x = 'gender', y = 'yearlysalary')
plt.title("Gender vs Income")
plt.ylim(100000,1000000,50000)