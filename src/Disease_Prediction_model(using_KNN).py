import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

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
model=KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)

#predictions
ypred=model.predict(X_test)
yprob=model.predict_proba(X_test)[:,1]

#model evaluation
print("Confusion Matrix (KNN):")
print(confusion_matrix(y_test, ypred))

print("\nClassification Report (KNN):")
print(classification_report(y_test, ypred))

roc_auc_knn = roc_auc_score(y_test, yprob)
print("ROC-AUC (KNN):", roc_auc_knn)

for k in [3, 2, 7, 9, 11]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(f"K={k} | Accuracy:", knn.score(X_test, y_test))


