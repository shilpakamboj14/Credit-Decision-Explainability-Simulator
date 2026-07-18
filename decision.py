import pandas as pd
import joblib
from explain import get_decision_text

model = joblib.load("/Users/shilpa/credit_project/data/model.pkl")
scaler = joblib.load("/Users/shilpa/credit_project/data/scaler.pkl")

data = pd.read_csv("/Users/shilpa/credit_project/data/cleaned_data.csv")
features = data.drop(["SeriousDlqin2yrs", "cust_id"], axis=1)
features_scaled = scaler.transform(features)

new_applicant = pd.DataFrame({
    "RevolvingUtilizationOfUnsecuredLines": [0.99],
    "age": [23],
    "NumberOfTime30-59DaysPastDueNotWorse": [6],
    "DebtRatio": [0.95],
    "MonthlyIncome": [1200],
    "NumberOfTimes90DaysLate": [4]
})

applicant_scaled = scaler.transform(new_applicant)
prediction = model.predict(applicant_scaled)

result = get_decision_text(model, features_scaled, features.columns, applicant_scaled, prediction)

print(result)