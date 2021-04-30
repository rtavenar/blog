import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tslearn.metrics import dtw_path
from utils import set_fig_style

f = np.zeros((12, ))
f[:4] = -1.
f[4:8] = 1.
f[8:] = -1.

length = 20

fig = plt.figure(figsize=(12, 4))
set_fig_style(fig, font_size=14)
ax2 = plt.subplot(1, 2, 1)
ax = plt.subplot(1, 2, 2)
for cur_ax in fig.axes:
    cur_ax.set_facecolor(fig.patch.get_facecolor())
colors = sns.color_palette("Paired")

x_ref = np.zeros((40, ))
x_ref[5:5+length] = np.sin(np.linspace(0, 2 * np.pi, num=length))

shift = 5
x = np.zeros((40, ))
x[5+shift:5+shift+length] = np.sin(np.linspace(0, 2 * np.pi, num=length))

path, _ = dtw_path(x_ref, x)

x -= 3

ax.plot(x_ref, color=colors[7], linestyle='-', marker='o', zorder=1)
ax.plot(x, color=colors[7], linestyle='-', marker='o', zorder=1)
for idx, (i, j) in enumerate(path):
    ax.plot([i, j], [x_ref[i], x[j]], color='k', alpha=.2, zorder=0)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Dynamic Time Warping")

ax2.plot(x_ref, color=colors[7], linestyle='-', marker='o', zorder=1)
ax2.plot(x, color=colors[7], linestyle='-', marker='o', zorder=1)
for idx in range(len(x_ref)):
    ax2.plot([idx, idx], [x_ref[idx], x[idx]], color='k', alpha=.2, zorder=0)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_title("Euclidean distance")

plt.tight_layout()
plt.savefig('fig/dtw_vs_euc.svg')