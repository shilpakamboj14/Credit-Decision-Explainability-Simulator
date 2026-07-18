import pandas as pd
import shap

feature_labels = {
    "NumberOfTime30-59DaysPastDueNotWorse": "a history of late payments (30-59 days past due)",
    "age": "applicant age",
    "DebtRatio": "high debt-to-income ratio",
    "MonthlyIncome": "monthly income level",
    "NumberOfTimes90DaysLate": "a history of severely late payments (90+ days)",
    "RevolvingUtilizationOfUnsecuredLines": "high credit utilization"
}

def get_decision_text(model, features_scaled, feature_columns, applicant_scaled, prediction):

    if prediction[0] == 0:
        return "This application was approved."

    explainer = shap.Explainer(model, features_scaled)
    shap_values = explainer(applicant_scaled)

    reasons = pd.Series(shap_values.values[0], index=feature_columns)
    reasons_sorted = reasons.sort_values(ascending=False)
    top_reasons = reasons_sorted.head(3)
    top_feature_names = top_reasons.index.tolist()

    readable_reasons = []
    for name in top_feature_names:
        readable_reasons.append(feature_labels[name])

    reason_text = ", ".join(readable_reasons)
    final_sentence = "This application was declined primarily due to: " + reason_text + "."

    return final_sentence