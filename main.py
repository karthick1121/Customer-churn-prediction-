# Predicting Customer Churn using Machine Learning
# ------------------------------------------------
# Objective: Use machine learning techniques to predict customer churn
# and uncover hidden patterns in customer behavior.

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 2: Load Dataset
# Replace with your own dataset if available
df = pd.read_csv("customer_churn.csv")
df.head()

# Step 3: Explore and Clean Data
print(df.info())
print(df.isnull().sum())

# Optional: Handle missing values
df.fillna(method='ffill', inplace=True)

# Step 4: Encode Categorical Columns
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Step 5: Feature Selection
X = df.drop("Churn", axis=1)  # 'Churn' is the target column
y = df["Churn"]

# Step 6: Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 8: Train Model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 9: Evaluate Model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 10: Feature Importance
importances = model.feature_importances_
feat_importance = pd.Series(importances, index=X.columns)
feat_importance.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.show()
