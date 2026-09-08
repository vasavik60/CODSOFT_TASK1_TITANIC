import pandas as pd

# Load Titanic dataset
data = pd.read_csv("titanic-dataset.csv")

# Display first 5 rows
print("First 5 rows:")
print(data.head())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())
# Data Cleaning

# Fill missing Age values with median
data["Age"] = data["Age"].fillna(data["Age"].median())

# Fill missing Embarked values with most frequent value
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

# Drop Cabin column because it has too many missing values
data = data.drop("Cabin", axis=1)

# Check missing values again
print("\nMissing Values After Cleaning:")
print(data.isnull().sum())
# Convert categorical values into numbers

data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

data["Embarked"] = data["Embarked"].map({"S": 0, "C": 1, "Q": 2})

print("\nData After Preprocessing:")
print(data.head())
# Select features and target

X = data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]

y = data["Survived"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())
# Split data into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)
# Train Logistic Regression model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred[:10])
# Check model accuracy

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("Model Accuracy Percentage:", accuracy * 100, "%")
# Classification Report and Confusion Matrix

from sklearn.metrics import classification_report, confusion_matrix

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Visualization

import matplotlib.pyplot as plt

survival_counts = data["Survived"].value_counts()

plt.bar(["Did Not Survive", "Survived"], survival_counts)

plt.title("Titanic Survival Count")
plt.xlabel("Survival Status")
plt.ylabel("Number of Passengers")

plt.show()