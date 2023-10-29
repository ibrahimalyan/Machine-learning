import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate data for Group 1
mean1 = [-1, -1]
cov1 = [[0.8, 0], [0, 0.8]]
group1_data = np.random.multivariate_normal(mean1, cov1, 700)

# Generate data for Group 2
mean2 = [1, 1]
cov2 = [[0.75, -0.2], [-0.2, 0.6]]
group2_data = np.random.multivariate_normal(mean2, cov2, 300)

# Plot the data
plt.scatter(group1_data[:, 0], group1_data[:, 1], color='blue', label='Group 1')
plt.scatter(group2_data[:, 0], group2_data[:, 1], color='red', label='Group 2')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Generated Data')
plt.show()



from sklearn.cluster import KMeans

# Combine the data from both groups
data = np.concatenate((group1_data, group2_data))

# Perform K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(data)

# Get the predicted labels for the data points
predicted_labels = kmeans.labels_

# Plot the clusters with their predicted labels
plt.scatter(data[predicted_labels == 0, 0], data[predicted_labels == 0, 1], color='blue', label='Predicted Group 1')
plt.scatter(data[predicted_labels == 1, 0], data[predicted_labels == 1, 1], color='red', label='Predicted Group 2')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('K-means Clustering')
plt.show()