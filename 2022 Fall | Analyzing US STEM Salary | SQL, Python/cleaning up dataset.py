#%%
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)

#%%
old_df=pd.read_csv('Levels_Fyi_Salary_Data.csv')

old_df.info()
old_df.head()

#%%
df = old_df[['company','title','location','yearsofexperience','yearsatcompany']].copy()

df['yearlysalary'] = old_df['totalyearlycompensation'] + old_df['basesalary']
# %%
df.to_csv('salary_data', index=False)
# %%
