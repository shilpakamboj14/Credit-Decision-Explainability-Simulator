import pandas as pd
import joblib
import shap
from step3_explain import feature_labels

model = joblib.load("data/model.pkl")
scaler = joblib.load("data/scaler.pkl")
data = pd.read_csv("data/cleaned_data.csv")

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

print("STEP 0: Prediction")
print(prediction)
print("---")

explainer = shap.Explainer(model, features_scaled)
shap_values = explainer(applicant_scaled)

print("STEP 1: Raw SHAP values")
print(shap_values.values)
print("---")

reasons = pd.Series(shap_values.values[0], index=features.columns)
print("STEP 2: Feature names + SHAP values together")
print(reasons)
print("---")

reasons_sorted = reasons.sort_values(ascending=False)
print("STEP 3: Sorted (biggest impact first)")
print(reasons_sorted)
print("---")

top_reasons = reasons_sorted.head(3)
print("STEP 4: Top 3 only")
print(top_reasons)
print("---")

top_feature_names = top_reasons.index.tolist()
print("STEP 5: Just the names, as a list")
print(top_feature_names)
print("---")

readable_reasons = []
for name in top_feature_names:
    readable_reasons.append(feature_labels[name])

print("STEP 6: Converted to plain English")
print(readable_reasons)
print("---")

reason_text = ", ".join(readable_reasons)
print("STEP 7: Joined into one string")
print(reason_text)
print("---")

final_sentence = "This application was declined primarily due to: " + reason_text + "."
print("STEP 8: Final sentence")
print(final_sentence)