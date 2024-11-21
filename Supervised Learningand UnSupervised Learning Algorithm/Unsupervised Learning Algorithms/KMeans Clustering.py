import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/mnt/data/laptop.csv'  # Update the path if running locally
data = pd.read_csv(file_path)

# Data preprocessing
# Drop unnecessary columns
data_cleaned = data.drop(columns=["Unnamed: 0", "Name", "Brand"])
data_cleaned = data_cleaned.dropna()  # Remove rows with missing values

# Separate features (excluding the target variable 'Price' for clustering)
X = data_cleaned.drop(columns=["Price"])

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize K-Means with a chosen number of clusters (e.g., 3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the K-Means model
kmeans.fit(X_scaled)

# Get the cluster labels
labels = kmeans.labels_

# Add the cluster labels to the original dataset
data_cleaned['Cluster'] = labels

# Print cluster centers
print("Cluster Centers:")
print(kmeans.cluster_centers_)

# Visualize the clustering (optional, using PCA to reduce dimensions to 2 for visualization)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Scatter plot of the clusters
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', marker='o')
plt.title('K-Means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster ID')
plt.show()
