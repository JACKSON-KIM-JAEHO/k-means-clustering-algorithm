# 모듈 임포트 및 데이터 생성

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import os

os.environ["OMP_NUM_THREADS"] = "2"
# 1. 데이터 생성
X,y = make_blobs(n_samples = 500, centers = 3, cluster_std = 0.75, random_state = 0, shuffle = True)

# 2. K-means 훈련 과정 시각화
fig, axs = plt.subplots(2, 3, figsize = (10,6))
axs = axs.flatten()

counter = 0
# 3. 반복마다 클러스터링 상태를 시각화
for i in (1,10,20,30,40): # 최대 40번의 반복을 보여줌
    #K-means 모델 생성
    kmeans = KMeans(n_clusters=3, random_state=0, max_iter=i)

    kmeans.fit(X)

    #예측 값
    y_kmeans = kmeans.predict(X)
    
    #각 군집에 대한 데이터 포인트와 군집 중심을 시각화
    axs[counter].scatter(X[:,0], X[:, 1], c = y_kmeans, s=50, cmap = 'viridis')

    #군집 중심
    centers = kmeans.cluster_centers_
    axs[counter].scatter(centers[:, 0], centers[:, 1], c = 'red', s=200, alpha=0.5, marker = 'X')

    #제목 설정
    axs[counter].set_title(f"iteration {i}")
    counter += 1

    # 4. 정답 scatter해서 비교하기
    axs[5].scatter(X[:, 0], X[:, 1], c=y, s=50, cmap = 'viridis')
    axs[5] .set_title("ANSWER")

#그래프 제목과 레이블
plt.suptitle("K-means Clustering Progress", fontsize = 16)
plt.tight_layout()
plt.show()