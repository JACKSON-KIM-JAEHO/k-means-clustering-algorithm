import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 1. 클러스터 개수 설정 (사용자가 변경 가능)
num_clusters = 6

# 2. 데이터 생성 (매번 랜덤한 위치에서 시작하도록 random_state 제거)
X, y = make_blobs(n_samples=134, centers=num_clusters, cluster_std=0.75, shuffle=True)

# 3. 사용자로부터 반복 횟수 입력 받기
max_iterations = 60

# 4. K-means 모델 생성 및 학습
kmeans = KMeans(n_clusters=num_clusters, init='random', n_init=20, max_iter=max_iterations)
kmeans.fit(X)

# 5. 예측 값
y_kmeans = kmeans.predict(X)

# 6. 시각화
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# 6.1 K-means 결과 시각화
axs[0].scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
axs[0].scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5, marker='X')
axs[0].set_title(f"K-means Result after {max_iterations} Iterations")

# 6.2 실제 정답 시각화
axs[1].scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')
axs[1].set_title("Actual Clusters (ANSWER)")

# 그래프 제목과 레이블
plt.suptitle(f"K-means Clustering with {num_clusters} Clusters", fontsize=16)
plt.tight_layout()
plt.show()