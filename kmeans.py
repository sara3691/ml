import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/sara3691/ml/main/student_dataset.csv")

# Select data for clustering
X = df[["Study_Hours", "Final_Mark"]]

# Create K-Means model
model = KMeans(n_clusters=3, random_state=42, n_init=10)

# Train model
model.fit(X)

# Get cluster labels
clusters = model.labels_

# Get cluster centers
centers = model.cluster_centers_

print("Cluster Labels:")
print(clusters)

print("\nCluster Centers:")
print(centers)

# Plot clusters
plt.scatter(
    X["Study_Hours"],
    X["Final_Mark"],
    c=clusters
)

# Plot cluster centers
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="X",
    s=200
)

plt.xlabel("Study Hours")
plt.ylabel("Final Mark")
plt.title("K-Means Clustering")
plt.show()
