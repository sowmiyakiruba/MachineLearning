import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/mnt/data/laptop.csv'  # Update the path if running locally
data = pd.read_csv(file_path)

# Data preprocessing
# Drop unnecessary columns
data_cleaned = data.drop(columns=["Unnamed: 0", "Name", "Brand"])
data_cleaned = data_cleaned.dropna()  # Remove rows with missing values

# Separate features (we'll exclude the 'Price' column for clustering)
X = data_cleaned.drop(columns=["Price"])

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust eps (distance threshold) and min_samples (minimum points to form a cluster)

# Fit DBSCAN to the data
dbscan.fit(X_scaled)

# Get the cluster labels
labels = dbscan.labels_

# Add the cluster labels to the original dataset
data_cleaned['Cluster'] = labels

# Print the number of clusters and outliers
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # excluding noise points labeled as -1
n_noise = list(labels).count(-1)
print(f"Number of clusters: {n_clusters}")
print(f"Number of outliers/noise points: {n_noise}")

# Visualize the clustering (Optional, for 2D features; using first two principal components here)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', marker='o')
plt.title('DBSCAN Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster ID')
plt.show()
