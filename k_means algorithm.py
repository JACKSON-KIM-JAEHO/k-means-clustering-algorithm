import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import defaultdict

# make data (random position)
num_samples = 100  # number of package (customer)
num_trucks = 10  # number of truck
max_customers_per_cluster = 10 # same with number of drone

x_min, x_max = -10, 10
y_min, y_max = -10, 10

X = np.column_stack((
    np.random.uniform(x_min, x_max, num_samples), 
    np.random.uniform(y_min, y_max, num_samples)
))

# number of clusters
num_initial_clusters = num_samples // max_customers_per_cluster
if num_samples % max_customers_per_cluster != 0:
    num_initial_clusters += 1 

# Perform 1st K-means clustering
max_iter_kmeans = 200  # number of iteration
initial_kmeans = KMeans(n_clusters=num_initial_clusters, init='k-means++', n_init=10, max_iter=max_iter_kmeans, random_state=None)
initial_labels = initial_kmeans.fit_predict(X)

# check if each cluster has less than number of drones (max_customers_per_cluster)
cluster_dict = defaultdict(list)
for i, label in enumerate(initial_labels):
    cluster_dict[label].append(X[i])

adjusted_clusters = []
for cluster_data in cluster_dict.values():
    cluster_data = np.array(cluster_data)
    
    # make sure if each cluster has less than number of drones (max_customers_per_cluster)
    while len(cluster_data) > max_customers_per_cluster:
        split_kmeans = KMeans(n_clusters=2, init='k-means++', n_init=5, max_iter=max_iter_kmeans, random_state=None)
        split_labels = split_kmeans.fit_predict(cluster_data)
        
        # when each cluster has more number of max_customers_per_cluster than split cluster
        cluster1 = cluster_data[split_labels == 0]
        cluster2 = cluster_data[split_labels == 1]

        if len(cluster1) > max_customers_per_cluster:
            adjusted_clusters.append(cluster1[:max_customers_per_cluster]) 
            cluster_data = np.vstack((cluster1[max_customers_per_cluster:], cluster2))  
        elif len(cluster2) > max_customers_per_cluster:
            adjusted_clusters.append(cluster2[:max_customers_per_cluster])  
            cluster_data = np.vstack((cluster1, cluster2[max_customers_per_cluster:]))
        else:
            adjusted_clusters.append(cluster1)
            cluster_data = cluster2  
    adjusted_clusters.append(cluster_data)

final_initial_centers = np.array([cluster.mean(axis=0) for cluster in adjusted_clusters])
num_adjusted_clusters = len(adjusted_clusters)

# Perform 2nd K-means clustering
truck_kmeans = KMeans(n_clusters=num_trucks, init='k-means++', n_init=10, max_iter=max_iter_kmeans, random_state=None)
truck_labels = truck_kmeans.fit_predict(final_initial_centers)
truck_centers = truck_kmeans.cluster_centers_

#visualize
plt.figure(figsize=(14, 6))

# visualize pf 1st K-means
ax1 = plt.subplot(1, 2, 1)
for idx, cluster in enumerate(adjusted_clusters):
    ax1.scatter(cluster[:, 0], cluster[:, 1], s = 100, label=f'Cluster {idx}')
ax1.scatter(final_initial_centers[:, 0], final_initial_centers[:, 1], c='red', marker='X', s=10, label='Adjusted Centers')
ax1.set_title(f"Customer Clustering (Max {max_customers_per_cluster} per group) ({num_adjusted_clusters} clusters)")
ax1.legend()
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, y_max)

# visualize pf 1st K-means
ax2 = plt.subplot(1, 2, 2)
ax2.scatter(final_initial_centers[:, 0], final_initial_centers[:, 1], c=truck_labels, cmap='tab10', marker='X', s=100, label='Truck Assignment')
ax2.scatter(truck_centers[:, 0], truck_centers[:, 1], c='black', marker='P', s=150, label='Truck Centers')
ax2.set_title(f"Truck Assignment Clustering (User-defined: {num_trucks} trucks, Max Iter: {max_iter_kmeans})")
ax2.legend()
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(y_min, y_max)

plt.tight_layout()
plt.show()
