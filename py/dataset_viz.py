import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from tslearn.datasets import CachedDatasets
from utils import set_fig_style, export_animation


def animate(i):
    for j in range(len(X_train)):
        if j < i:
            lines[j].set_alpha(lines[j].get_alpha() * .7)
        elif i == j:
            lines[j].set_alpha(1.)
        else:
            lines[j].set_alpha(0.)

    return lines

shift = 15
length = 60

fig, ax = plt.subplots()
set_fig_style(fig, font_size=14)
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

lines = []
for i in range(len(X_train)):
    line_ts, = ax.plot(X_train[i].ravel(), color=colors[7], linestyle='-', alpha=1. if i == 0 else 0.)
    lines.append(line_ts)
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylim([-2.5, 4])

plt.tight_layout()

ani = animation.FuncAnimation(fig, animate, interval=200, blit=True, save_count=len(X_train))
export_animation(ani, 'fig/dataset_viz')
