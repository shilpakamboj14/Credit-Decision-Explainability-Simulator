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

    # FIRST CHECK IF APPROVED OR DECLINED
    if prediction[0] == 0:
        
        # IF APPROVED THEN SEND THIS MESSAGE
        final_sentence = "This application was approved."

    else:
        # IF DECLINED THEN FIND SHAP REASON

        # Step 1: MAKE SHAP EXPLAINER THAT UNDERSTAND THE MODEL
        explainer = shap.Explainer(model, features_scaled)

        # Step 2: FIND SHAP VALUES FOR A SPECIFIC APPLICANT
        shap_values = explainer(applicant_scaled)

        # Step 3: GIVE FEATURE NAME TO THE VALUES
        reasons = pd.Series(shap_values.values[0], index=feature_columns)

        # Step 4: ORDER THE REASONS BASED ON THEIR RISK SCORE FROM HIGH TO LOW
        reasons_sorted = reasons.sort_values(ascending=False)

        # Step 5: FETCH ONLY TOP 3 REASONS AS PER ECOA
        top_reasons = reasons_sorted.head(3)

        # Step 6: ONLY GET THE NAME OF FEATURES SO REMOVE THE NUMERS
        top_feature_names = top_reasons.index.tolist()

        # Step 7: CHANGE FEATURE COLUMN NAMES TO THE FEATURE LABELS TO UNDESTAND IT IN PLAIN ENGLISH
        readable_reasons = []
        for name in top_feature_names:
            plain_english_phrase = feature_labels[name]
            readable_reasons.append(plain_english_phrase)

        # Step 8: JOIN ALL THE REASONS TOGETHER WITH COMMA
        reason_text = ", ".join(readable_reasons)

        # Step 9: CREATE FINAL REASONING SENTENCE
        final_sentence = "This application was declined primarily due to: " + reason_text + "."

    # RETURN THE FINAL SENTENCE FOR APPROVAL OR REJECTION
    return final_sentence