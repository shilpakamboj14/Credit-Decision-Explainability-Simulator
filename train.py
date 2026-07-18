import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

data = pd.read_csv("/Users/shilpa/credit_project/data/cleaned_data.csv")

target = data["SeriousDlqin2yrs"]
features = data.drop(["SeriousDlqin2yrs", "cust_id"], axis=1)

features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

model = LogisticRegression(max_iter=1000, class_weight={0: 1, 1: 9})
model.fit(features_train, target_train)

predictions = model.predict(features_test)

print(confusion_matrix(target_test, predictions))
print(classification_report(target_test, predictions))

joblib.dump(model, "/Users/shilpa/credit_project/data/model.pkl")
joblib.dump(scaler, "/Users/shilpa/credit_project/data/scaler.pkl")