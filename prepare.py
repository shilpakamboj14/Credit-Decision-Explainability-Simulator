import pandas as pd

data = pd.read_csv("/Users/shilpa/credit_project/data/cs-training.csv")

data = data.rename(columns={"Unnamed: 0": "cust_id"})

columns_needed = ["cust_id", "SeriousDlqin2yrs", "RevolvingUtilizationOfUnsecuredLines", "age",
                   "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio",
                   "MonthlyIncome", "NumberOfTimes90DaysLate"]

data = data[columns_needed]

median_income = data["MonthlyIncome"].median()
data["MonthlyIncome"] = data["MonthlyIncome"].fillna(median_income)

data.to_csv("/Users/shilpa/credit_project/data/cleaned_data.csv", index=False)