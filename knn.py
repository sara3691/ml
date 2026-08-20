import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/sara3691/ml/main/student_dataset.csv")

# Input
X = df[["Study_Hours", "Attendance", "Internal_Mark"]]
y = df["Result"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SVM model
model = SVC(kernel="linear")

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("SVM Binary Classification")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot
plt.scatter(
    df["Study_Hours"],
    df["Internal_Mark"],
    c=df["Result"]
)

plt.xlabel("Study Hours")
plt.ylabel("Internal Mark")
plt.title("SVM Binary Classification")
plt.colorbar(label="Result (0 = Fail, 1 = Pass)")
plt.show()
