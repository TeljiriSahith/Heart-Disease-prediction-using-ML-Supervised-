import pandas as pd
import joblib

# Load saved objects
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")

#dataset
patient = {
    'age': 55,
    'trestbps': 140,
    'chol': 250,
    'thalch': 150,
    'oldpeak': 2.3,
    'ca': 0,
    'sex': 'male',
    'cp': 'typical angina',
    'fbs': True,
    'restecg': 'normal',
    'exang': True,
    'slope': 'flat',
    'thal': 'reversible defect'
}

df_patient = pd.DataFrame([patient])
categoricals_cols=['sex','cp','fbs','restecg','exang','slope','thal']
numeric_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
#encoding
df_patient_encoding=pd.get_dummies(df_patient,columns=categoricals_cols,drop_first=True)

#reindexing
df_patient_encoding = df_patient_encoding.reindex(columns=model_columns,fill_value=0)

#scaling
df_patient_encoding[numeric_cols] = scaler.transform(df_patient_encoding[numeric_cols])

probability = model.predict_proba(df_patient_encoding)[0][1]
prediction = (probability >= 0.3).astype(int)

print("Heart Disease Probability:", round(probability, 2))

if prediction == 1:
    print("⚠️ High Risk: Heart Disease Detected")
else:
    print("✅ Low Risk: No Heart Disease")



