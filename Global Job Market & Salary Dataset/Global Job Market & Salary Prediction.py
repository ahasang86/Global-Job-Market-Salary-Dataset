"""
PROJECT : GLOBAL JOB MARKET & SALARY PREDICTION
------------------------------------------------
This project analyzes worldwide salary data, explores trends,
and builds a salary prediction model using Linear Regression.

Skills demonstrated:
- Pandas data cleaning & wrangling
- NumPy numerical operations
- Seaborn & Matplotlib visualizations
- SciPy hypothesis testing
- Machine learning (Linear Regression)
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv("data/worldwide_salary_data.csv")

print(df.head())      # Preview first rows
print(df.info())      # Check column types
#print("describe data")
print(df.describe())  # Summary statistics
#sys.exit()

# ---------------------------------------------------------
# 2. CLEAN DATA
# ---------------------------------------------------------
df = df.dropna()  # Remove missing values

# Remove extreme outliers (top 1% salary)
df = df[df["salary_in_usd"] < df["salary_in_usd"].quantile(0.99)]

# ---------------------------------------------------------
# 3. VISUALIZE SALARY DISTRIBUTION
# ---------------------------------------------------------
plt.figure(figsize=(10,5))
sns.histplot(df["salary_in_usd"], kde=True)
plt.title("Salary Distribution (USD)")
plt.show()
#sys.exit()
# ---------------------------------------------------------
# 4. SALARY BY COUNTRY
# ---------------------------------------------------------
plt.figure(figsize=(12,6))
sns.boxplot(data=df, x="employee_residence", y="salary_in_usd")
plt.xticks(rotation=0)
plt.title("Salary by Country")
plt.show()
#sys.exit()
# ---------------------------------------------------------
# 5. CORRELATION HEATMAP
# ---------------------------------------------------------
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
#sys.exit()
# ---------------------------------------------------------
# 6. PREPARE DATA FOR SALARY PREDICTION
# ---------------------------------------------------------
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("salary_in_usd", axis=1)
y = df_encoded["salary_in_usd"]
print(X)
print(y)
sys.exit()

# Split into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 7. TRAIN LINEAR REGRESSION MODEL
# ---------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# Predict salaries
pred = model.predict(X_test)

# Evaluate model
print("MAE:", mean_absolute_error(y_test, pred))
print("R2 Score:", r2_score(y_test, pred))

# ---------------------------------------------------------
# 8. HYPOTHESIS TEST: REMOTE VS ONSITE SALARY
# ---------------------------------------------------------
remote = df[df["remote_ratio"] == 100]["salary_in_usd"]
onsite = df[df["remote_ratio"] == 0]["salary_in_usd"]

t_stat, p_value = stats.ttest_ind(remote, onsite)

print("T-test:", t_stat)
print("P-value:", p_value)