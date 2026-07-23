import pandas as pd
import joblib
from pathlib import Path
from step3_explain import get_decision_text

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"


model = joblib.load(DATA_DIR / "model.pkl")
scaler = joblib.load(DATA_DIR / "scaler.pkl")
data = pd.read_csv(DATA_DIR / "cleaned_data.csv")

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