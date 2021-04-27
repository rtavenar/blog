from IPython.display import HTML
from celluloid import Camera

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from tslearn.metrics import dtw, soft_dtw
from tslearn.datasets import CachedDatasets
from utils import set_fig_style


def soft_dtw_div(x, y, gamma):
    return soft_dtw(x, y, gamma) - .5 * (soft_dtw(x, x, gamma) 
                                         + soft_dtw(y, y, gamma))


def animate(i):
    line_ts.set_ydata(list_x[i])
    scatter_dtw.set_xdata([list_values[i]])
    scatter_dtw.set_ydata([dtw_dists[i]])

    return line_ts, scatter_dtw

f = np.zeros((12, ))
f[:4] = -1.
f[4:8] = 1.
f[8:] = -1.

shift = 15
length = 60

fig = plt.figure()
set_fig_style(fig, font_size=14)
ax = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)
for cur_ax in fig.axes:
    cur_ax.set_facecolor(fig.patch.get_facecolor())
colors = sns.color_palette("Paired")

x_train = CachedDatasets().load_dataset("Trace")[0]
x_ref = x_train[3].ravel()
x = x_train[6].ravel()

pos = np.argmax(x)
list_values = x[pos] + np.linspace(-1., .1, num=50)

list_x = []
for mid_val in list_values:
    xi = x.copy()
    xi[pos] = mid_val
    list_x.append(xi)

dtw_dists = [dtw(x_ref, xi) for xi in list_x]

ax.plot(x_ref, color=colors[7], linestyle='dotted')
line_ts, = ax.plot(list_x[0], color=colors[7], linestyle='-')
ax.set_xticks([pos])
ax.set_xticklabels(["$\\tau$"], fontsize=14)
ax.set_title("Input time series")
ax.set_ylim([-2, 1.5])

ax2.plot(list_values, dtw_dists, color=colors[1], linestyle='-', label="DTW")
scatter_dtw, = ax2.plot([list_values[0]], [dtw_dists[0]], color=colors[1], marker='o')
ax2.set_xlabel("$x_{\\tau}$")
ax2.set_ylabel("DTW")

plt.tight_layout()

ani = animation.FuncAnimation(fig, animate, interval=50, blit=True, save_count=len(list_values))
ani.save(
    'fig/dtw_landscape.gif',
    dpi=100, savefig_kwargs={'pad_inches': 'tight'}
)