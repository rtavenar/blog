import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from tslearn.metrics import dtw
from utils import set_fig_style, export_animation


def animate(i):
    line_ts.set_ydata(list_x[i])
    scatter_dtw.set_xdata([list_shifts[i]])
    scatter_dtw.set_ydata([dtw_dists[i]])
    scatter_sakoe.set_xdata([list_shifts[i]])
    scatter_sakoe.set_ydata([sakoe_dists[i]])

    return line_ts, scatter_dtw, scatter_sakoe

f = np.zeros((12, ))
f[:4] = -1.
f[4:8] = 1.
f[8:] = -1.

length = 60
r = 15

fig = plt.figure()
set_fig_style(fig, font_size=14)
ax = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)
for cur_ax in fig.axes:
    cur_ax.set_facecolor(fig.patch.get_facecolor())
colors = sns.color_palette("Paired")

x_ref = np.zeros((100, ))
x_ref[5:5+length] = np.sin(np.linspace(0, 2 * np.pi, num=length))

list_x = []
list_shifts = list(range(0, 30)) + list(range(30, 0, -1))
for shift in list_shifts:
    x = np.zeros((100, ))
    x[5+shift:5+shift+length] = np.sin(np.linspace(0, 2 * np.pi, num=length))
    list_x.append(x)

dtw_dists = [dtw(x_ref, x) for x in list_x]
sakoe_dists = [dtw(x_ref, x, sakoe_chiba_radius=r) for x in list_x]

ax.plot(x_ref, color=colors[7], linestyle='-')
line_ts, = ax.plot(list_x[0], color=colors[7], linestyle='dotted')
ax.set_title("Input time series")

ax2.plot(list_shifts, dtw_dists, color=colors[0], linestyle='-', label="without global constraints")
scatter_dtw, = ax2.plot([list_shifts[0]], [dtw_dists[0]], color=colors[0], marker='o')
ax2.plot(list_shifts, sakoe_dists, color=colors[1], linestyle='-', label="with Sakoe-Chiba band\n of width $r$")
scatter_sakoe, = ax2.plot([list_shifts[0]], [sakoe_dists[0]], color=colors[1], marker='o')
ax2.axvline(x=r, linestyle='dashed', color='k')
ax2.set_xticks([r])
ax2.set_xticklabels(['$r$'])
ax2.set_xlabel("Temporal shift")
ax2.set_ylabel("DTW")
ax2.legend(loc="upper left", fontsize=12)

plt.tight_layout()

ani = animation.FuncAnimation(fig, animate, interval=100, blit=True, save_count=len(list_shifts))
export_animation(ani, 'fig/sakoe_shift')
