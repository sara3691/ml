import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/sara3691/ml/main/student_dataset.csv")

print(df)

print("Mean:", df["Final_Mark"].mean())
print("Median:", df["Final_Mark"].median())
print("Mode:", df["Final_Mark"].mode()[0])
print("Variance:", df["Final_Mark"].var())
print("Standard Deviation:", df["Final_Mark"].std())
