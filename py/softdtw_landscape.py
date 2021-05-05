import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from tslearn.metrics import dtw, soft_dtw
from tslearn.datasets import CachedDatasets
from utils import set_fig_style, export_animation


def soft_dtw_div(x, y, gamma):
    return soft_dtw(x, y, gamma) - .5 * (soft_dtw(x, x, gamma) 
                                         + soft_dtw(y, y, gamma))


def animate(i):
    line_ts.set_ydata(list_x[i])
    for gamma_idx, scatter_softdtw in enumerate(list_scatter_softdtw):
        scatter_softdtw.set_xdata([list_values[i]])
        scatter_softdtw.set_ydata([softdtw_dists[gamma_idx][i]])

    return [line_ts] + list_scatter_softdtw

shift = 15
length = 60

fig = plt.figure()
set_fig_style(fig, font_size=14)
ax = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)
for cur_ax in fig.axes:
    cur_ax.set_facecolor(fig.patch.get_facecolor())
colors = sns.color_palette("Paired")
colors_gradient = sns.color_palette("Blues")[2:]

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

gamma_values = [.01, .1]  # , .1, 1.]
softdtw_dists = [
    [soft_dtw_div(x_ref, xi, gamma) for xi in list_x]
    for gamma in gamma_values
]

ax.plot(x_ref, color=colors[7], linestyle='dotted')
line_ts, = ax.plot(list_x[0], color=colors[7], linestyle='-')
ax.set_xticks([pos])
ax.set_xticklabels(["$\\tau$"], fontsize=14)
ax.set_title("Input time series")
ax.set_ylim([-2, 1.5])

list_scatter_softdtw = [None] * len(gamma_values)
ax3 = ax2.twinx()
list_axes = [ax2, ax3]
for gamma_idx, (gamma, cur_ax) in enumerate(zip(gamma_values, list_axes)):
    cur_ax.plot(list_values, softdtw_dists[gamma_idx], color=colors_gradient[gamma_idx], linestyle='-', 
             label=f"$\\gamma={gamma}$")
    list_scatter_softdtw[gamma_idx], = cur_ax.plot([list_values[0]], [softdtw_dists[gamma_idx][0]], 
                                                color=colors_gradient[gamma_idx], marker='o')
ax2.set_xlabel("$x_{\\tau}$")
ax2.set_ylabel("softDTW divergence")
handles, labels = [(a + b) for a, b in zip(ax2.get_legend_handles_labels(), 
                                           ax3.get_legend_handles_labels())]
ax2.legend(handles, labels, loc="lower left", fontsize=12)
for cur_ax in list_axes:
    cur_ax.set_yticks([])

plt.tight_layout()

ani = animation.FuncAnimation(fig, animate, interval=50, blit=True, save_count=len(list_values))
export_animation(ani, 'fig/softdtw_landscape')
