import pandas as pd
import numpy as np

df=pd.read_csv('heart_disease_uci.csv')
print(df.head())

df.info()
op=df.describe(include='all')
print(op)
print(df.isnull().sum())
 
cols_with_zero_as_missing = ['trestbps','chol','thalch','oldpeak','thal']
df[cols_with_zero_as_missing] = df[cols_with_zero_as_missing].replace(0, np.nan)
print(df.isnull().sum())

numeric_cols=['trestbps', 'chol', 'thalch', 'oldpeak']
for i in numeric_cols:
    df[i].fillna(df[i].median(),inplace=True)

categorical_cols=['fbs', 'exang', 'restecg', 'slope', 'thal','ca']
for j in categorical_cols:
    df[j].fillna(df[j].mode()[0],inplace=True)

print(df.isnull().sum())

df.to_csv("heart_disease_cleaned.csv", index=False)
