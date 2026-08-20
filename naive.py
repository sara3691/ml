import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/sara3691/ml/main/student_dataset.csv")

# Input and output
X = df[["Study_Hours", "Attendance", "Internal_Mark"]]
y = df["Result"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Naive Bayes model
model = GaussianNB()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("Naive Bayes")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Plot
plt.scatter(
    df["Study_Hours"],
    df["Internal_Mark"],
    c=df["Result"]
)

plt.xlabel("Study Hours")
plt.ylabel("Internal Mark")
plt.title("Naive Bayes Classification")
plt.colorbar(label="Result")
plt.show()
