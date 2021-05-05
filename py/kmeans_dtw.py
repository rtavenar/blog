import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tslearn.datasets import CachedDatasets
from tslearn.clustering import TimeSeriesKMeans
from utils import set_fig_style

shift = 15
length = 60

fig = plt.figure(figsize=(9, 3))
axes = [plt.subplot(1, 3, i) for i in range(1, 4)]
set_fig_style(fig, font_size=20)
colors = sns.color_palette("Paired")

np.random.seed(0)
X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
X_train = X_train[y_train < 4]  # Keep first 3 classes
y_train = y_train[y_train < 4]  # Keep first 3 classes
list_Xis = []
for yi in range(1, 4):
    Xi = X_train[y_train == yi]
    np.random.shuffle(Xi)
    Xi = Xi[:15]  # Keep 15 samples per class at most
    list_Xis.append(Xi)
X_train = np.vstack(list_Xis)

dba_km = TimeSeriesKMeans(n_clusters=3,
                          metric="dtw",
                          verbose=False,
                          random_state=0)
y_pred = dba_km.fit_predict(X_train)

for yi in range(3):
    for xx in X_train[y_pred == yi]:
        axes[yi].plot(xx.ravel(), "k-", alpha=.2)
    axes[yi].plot(dba_km.cluster_centers_[yi].ravel(), color=colors[7], linestyle="-")
    axes[yi].set_ylim([-2.5, 4])
    axes[yi].set_xticks([])
    axes[yi].set_yticks([])
    axes[yi].text(x=0.55, y=0.85, s='Cluster %d' % (yi + 1),
             transform=axes[yi].transAxes)

plt.tight_layout()
plt.savefig("fig/kmeans_dtw.svg")
