from IPython.display import HTML
from celluloid import Camera

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from tslearn.metrics import soft_dtw
from utils import set_fig_style, export_animation


def soft_dtw_div(x, y, gamma):
    return soft_dtw(x, y, gamma) - .5 * (soft_dtw(x, x, gamma) 
                                         + soft_dtw(y, y, gamma))


def animate(i):
    line_ts.set_ydata(list_x[i])
    for gamma_idx, scatter_softdtw in enumerate(list_scatter_softdtw):
        scatter_softdtw.set_xdata([list_shifts[i]])
        scatter_softdtw.set_ydata([softdtw_dists[gamma_idx][i]])
    # scatter_euc.set_xdata([list_shifts[i]])
    # scatter_euc.set_ydata([euc_dists[i]])

    return [line_ts, ] + list_scatter_softdtw

length = 60

fig = plt.figure()
set_fig_style(fig, font_size=14)
ax = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)
for cur_ax in fig.axes:
    cur_ax.set_facecolor(fig.patch.get_facecolor())
colors = sns.color_palette("Paired")
colors_gradient = sns.color_palette("Blues")[2:]

x_ref = np.zeros((100, ))
x_ref[5:5+length] = np.sin(np.linspace(0, 2 * np.pi, num=length))

list_x = []
list_shifts = list(range(0, 30)) + list(range(30, 0, -1))
for shift in list_shifts:
    x = np.zeros((100, ))
    x[5+shift:5+shift+length] = np.sin(np.linspace(0, 2 * np.pi, num=length))
    list_x.append(x)

gamma_values = [.01, .1, 1.]
softdtw_dists = [
    [soft_dtw_div(x_ref, x, gamma=gamma) for x in list_x]
    for gamma in gamma_values
]
euc_dists = [np.linalg.norm(x_ref - x) for x in list_x]


ax.plot(x_ref, color=colors[7], linestyle='-')
line_ts, = ax.plot(list_x[0], color=colors[7], linestyle='dotted')
ax.set_title("Input time series")

# ax2.plot(list_shifts, euc_dists, color=colors[0], linestyle='-', label="Euclidean dist.")
# scatter_euc, = ax2.plot([list_shifts[0]], [euc_dists[0]], color=colors[0], marker='o')
list_scatter_softdtw = [None] * len(gamma_values)
for gamma_idx, gamma in enumerate(gamma_values):
    ax2.plot(list_shifts, softdtw_dists[gamma_idx], color=colors_gradient[gamma_idx], linestyle='-', 
             label=f"$\\gamma={gamma}$")
    list_scatter_softdtw[gamma_idx], = ax2.plot([list_shifts[0]], 
                                                [softdtw_dists[gamma_idx][0]], 
                                                color=colors_gradient[gamma_idx], marker='o')
ax2.set_xlabel("Temporal shift")
ax2.set_ylabel("softDTW divergence")
ax2.legend(loc="upper left", fontsize=12)

plt.tight_layout()

ani = animation.FuncAnimation(fig, animate, interval=100, blit=True, save_count=len(list_shifts))
export_animation(ani, 'fig/softdtw_shift')
