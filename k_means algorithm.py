import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import defaultdict

# (1) 고객 데이터 생성 (완전 랜덤한 배치)
num_samples = 50  # 고객 수
x_min, x_max = -10, 10  # X 좌표 범위
y_min, y_max = -10, 10  # Y 좌표 범위

X = np.column_stack((
    np.random.uniform(x_min, x_max, num_samples),  # X 좌표
    np.random.uniform(y_min, y_max, num_samples)   # Y 좌표
))

# (2) 1차 클러스터 개수 결정
num_initial_clusters = num_samples // 3
if num_samples % 3 != 0:
    num_initial_clusters += 1  # 나머지가 있다면 하나 더 추가

# (3) 1차 K-means 클러스터링 수행
max_iter_kmeans = 200  # 사용자가 설정할 수 있는 반복 횟수
initial_kmeans = KMeans(n_clusters=num_initial_clusters, init='k-means++', n_init=10, max_iter=max_iter_kmeans, random_state=None)
initial_labels = initial_kmeans.fit_predict(X)

# (4) 1차 클러스터링 보정 (각 클러스터가 3명 이하인지 확인)
cluster_dict = defaultdict(list)
for i, label in enumerate(initial_labels):
    cluster_dict[label].append(X[i])

adjusted_clusters = []
for cluster_data in cluster_dict.values():
    cluster_data = np.array(cluster_data)
    
    # **4명이 넘는 경우 계속 분할하도록 개선**
    while len(cluster_data) > 3:
        split_kmeans = KMeans(n_clusters=2, init='k-means++', n_init=5, max_iter=max_iter_kmeans, random_state=None)
        split_labels = split_kmeans.fit_predict(cluster_data)
        
        # 두 개의 클러스터로 나누기
        cluster1 = cluster_data[split_labels == 0]
        cluster2 = cluster_data[split_labels == 1]

        # 클러스터 크기 확인 후 다시 나눌지 결정
        if len(cluster1) > 3:
            adjusted_clusters.append(cluster1[:3])  # 3명만 저장
            cluster_data = np.vstack((cluster1[3:], cluster2))  # 남은 데이터 다시 검사
        elif len(cluster2) > 3:
            adjusted_clusters.append(cluster2[:3])  # 3명만 저장
            cluster_data = np.vstack((cluster1, cluster2[3:]))  # 남은 데이터 다시 검사
        else:
            adjusted_clusters.append(cluster1)
            cluster_data = cluster2  # 남은 데이터 반복
    adjusted_clusters.append(cluster_data)  # 최종 데이터 추가

# (5) 1차 클러스터링 후 최종 클러스터 중심 계산
final_initial_centers = np.array([cluster.mean(axis=0) for cluster in adjusted_clusters])
num_adjusted_clusters = len(adjusted_clusters)

# (6) 2차 K-means (트럭 클러스터링, 사용자가 직접 트럭 개수 설정)
num_trucks = 7  # 🚛 사용자가 트럭 개수를 직접 설정
truck_kmeans = KMeans(n_clusters=num_trucks, init='k-means++', n_init=10, max_iter=max_iter_kmeans, random_state=None)
truck_labels = truck_kmeans.fit_predict(final_initial_centers)
truck_centers = truck_kmeans.cluster_centers_

# (7) 시각화 (축 범위 동일하게 설정)
plt.figure(figsize=(14, 6))

# 1차 클러스터링 결과 (고객 배분)
ax1 = plt.subplot(1, 2, 1)
for idx, cluster in enumerate(adjusted_clusters):
    ax1.scatter(cluster[:, 0], cluster[:, 1], s = 100, label=f'Cluster {idx}')
ax1.scatter(final_initial_centers[:, 0], final_initial_centers[:, 1], c='red', marker='X', s=10, label='Adjusted Centers')
ax1.set_title(f"Customer Clustering (Max 3 per group) ({num_adjusted_clusters} clusters)")
ax1.legend()
ax1.set_xlim(x_min, x_max)  # X 축 고정
ax1.set_ylim(y_min, y_max)  # Y 축 고정

# 2차 클러스터링 결과 (트럭 배정)
ax2 = plt.subplot(1, 2, 2)
ax2.scatter(final_initial_centers[:, 0], final_initial_centers[:, 1], c=truck_labels, cmap='tab10', marker='X', s=100, label='Truck Assignment')
ax2.scatter(truck_centers[:, 0], truck_centers[:, 1], c='black', marker='P', s=150, label='Truck Centers')
ax2.set_title(f"Truck Assignment Clustering (User-defined: {num_trucks} trucks, Max Iter: {max_iter_kmeans})")
ax2.legend()
ax2.set_xlim(x_min, x_max)  # X 축 고정 (왼쪽과 동일)
ax2.set_ylim(y_min, y_max)  # Y 축 고정 (왼쪽과 동일)

plt.tight_layout()
plt.show()
