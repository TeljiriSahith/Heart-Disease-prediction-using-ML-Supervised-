import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df=pd.read_csv('heart_disease_cleaned.csv')
#df.info()
#print(df.isnull().sum())

df['target']=df['num'].apply(lambda x: 0 if x==0 else 1)
#df.info()
#print(df.isnull().sum())
X = df.drop(['target','num','dataset'], axis=1)
y = df['target']
numerical_cols = ['age','trestbps','chol','thalch','oldpeak','ca']
categoricals_cols=['sex','cp','fbs','restecg','exang','slope','thal']

#encoding categorical values
X_encode=pd.get_dummies(X, columns=categoricals_cols,drop_first=True)
#preeivewing the dataset for conformation
#X_encode.info()
#op=X_encode.head(10)
#print(op)

#splitting the dataset
X_train,X_test,y_train,y_test=train_test_split(X_encode,y,test_size=0.2,stratify=y,random_state=43)

#scaling data
scaler=StandardScaler()
X_train[numerical_cols]=scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols]=scaler.transform(X_test[numerical_cols])

#model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

#Evaluation
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

#threshold tunning 

#for threshold in [0.3, 0.2,]:
    #ypred_custom1=(y_prob>=threshold).astype(int)
    #print('\n threshold=',threshold)
    #print(confusion_matrix(y_test, ypred_custom1))
    #print(classification_report(y_test,ypred_custom1))

#Final Choice: threshold = 0.3
#Threshold 0.2 is too aggressive (too many false positives).
threshold=0.3
ypred_custom=(y_prob>=threshold).astype(int)
print("Confusion Matrix (threshold = 0.3)")
print(confusion_matrix(y_test, ypred_custom))

print("\nClassification Report (threshold = 0.3)")
print(classification_report(y_test, ypred_custom))

import joblib

joblib.dump(model, "heart_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X_train.columns, "model_columns.pkl")
