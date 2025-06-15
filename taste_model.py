import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

fruits = ['살구','복숭아','딸기','사과','파인애플','망고','포도']
responses = np.array([
    [10, 0.6, 0.05, 0.02],
    [9, 0.5, 0.04, 0.03],
    [7.5, 1.0, 0.02, 0.01],
    [10, 0.3, 0.03, 0.02],
    [13, 1.2, 0.05, 0.03],
    [14, 0.4, 0.04, 0.04],
    [15, 0.2, 0.03, 0.02],
])
resp_norm = responses / responses.max(axis=0)
pca = PCA(n_components=3)
reduced = pca.fit_transform(resp_norm)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i, fruit in enumerate(fruits):
    ax.scatter(*reduced[i], label=fruit)
    ax.text(*reduced[i], fruit, fontsize=9)
ax.set_title("3D Taste Coding of Fruits")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.legend()
plt.tight_layout()
plt.show()
