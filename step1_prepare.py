import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

data = pd.read_csv(DATA_DIR/ "cs-training.csv")

data = data.rename(columns={"Unnamed: 0": "cust_id"})

columns_needed = ["cust_id", "SeriousDlqin2yrs", "RevolvingUtilizationOfUnsecuredLines", "age",
                   "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio",
                   "MonthlyIncome", "NumberOfTimes90DaysLate"]

data = data[columns_needed]



median_income = data["MonthlyIncome"].median()
data["MonthlyIncome"] = data["MonthlyIncome"].fillna(median_income)


data.to_csv(DATA_DIR / "cleaned_data.csv", index=False)
print(data[columns_needed].isnull().sum())

print(data.shape)