import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 1. 클러스터 개수 설정 (사용자가 변경 가능)
num_clusters = 6

# 2. 데이터 생성 (매번 랜덤한 위치에서 시작하도록 random_state 제거)
X, y = make_blobs(n_samples=30, centers=num_clusters, cluster_std=0.75, shuffle=True)

# 3. K-means 훈련 과정 시각화
fig, axs = plt.subplots(2, 3, figsize=(10, 6))
axs = axs.flatten()

counter = 0
# 4. 반복마다 클러스터링 상태를 시각화
for i in [1, 10, 20, 30, 40]:  
    # K-means 모델 생성
    kmeans = KMeans(n_clusters=num_clusters, max_iter=i)
    kmeans.fit(X)

    # 예측 값
    y_kmeans = kmeans.predict(X)

    # 각 군집에 대한 데이터 포인트와 군집 중심을 시각화
    axs[counter].scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

    # 군집 중심
    centers = kmeans.cluster_centers_
    axs[counter].scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5, marker='X')

    # 제목 설정
    axs[counter].set_title(f"Iteration {i}")
    counter += 1

# 5. 정답 scatter해서 비교하기
axs[5].scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')
axs[5].set_title("ANSWER")

# 그래프 제목과 레이블
plt.suptitle(f"K-means Clustering Progress ({num_clusters} Clusters)", fontsize=16)
plt.tight_layout()
plt.show()
