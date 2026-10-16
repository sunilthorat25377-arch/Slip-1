# Slip-1 Q2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Load the Iris dataset
iris = load_iris()
X = iris.data

# Apply K-Means clustering (K = 3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

# Get cluster centroids
centroids = kmeans.cluster_centers_

# Plot clusters using the first two features
plt.figure(figsize=(9, 6))

for i in range(3):
    plt.scatter(
        X[labels == i, 0],
        X[labels == i, 1],
        label=f'Cluster {i + 1}'
    )

# Plot centroids
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker='X',
    s=220,
    color='red',
    edgecolors='black',
    label='Centroids'
)

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('K-Means Clustering on Iris Dataset (K=3)')
plt.legend()
plt.grid(True, alpha=0.25)
plt.show()

# Display centroid values
print("Cluster Centroids:")
print(np.round(centroids, 3))
