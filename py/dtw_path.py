import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from tslearn.metrics import dtw_path
from utils import set_fig_style


def animate(i):
    if i < len_pause or i >= len_anim + len_pause:
        return []
    
    t = (i - len_pause) / len_anim
    pos_x_ref = [
        (1 - t) * i + t * idx
        for idx, (i, j) in enumerate(path)
    ]
    pos_x = [
        (1 - t) * j + t * idx
        for idx, (i, j) in enumerate(path)
    ]
    line_x_ref.set_xdata([pos for idx, pos in enumerate(pos_x_ref) if not(x_ref_repeated[idx])])
    line_x.set_xdata([pos for idx, pos in enumerate(pos_x) if not(x_repeated[idx])])
    
    line_x_ref_dummy.set_xdata([pos for idx, pos in enumerate(pos_x_ref) if x_ref_repeated[idx]])
    line_x_dummy.set_xdata([pos for idx, pos in enumerate(pos_x) if x_repeated[idx]])
    
    for idx in range(len(path)):
        list_matches[idx].set_xdata([pos_x_ref[idx], pos_x[idx]])

    return [line_x, line_x_ref, line_x_dummy, line_x_ref_dummy] + list_matches

f = np.zeros((12, ))
f[:4] = -1.
f[4:8] = 1.
f[8:] = -1.

length = 20

fig, ax = plt.subplots()
set_fig_style(fig, font_size=14)
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
x_ref_resampled = [x_ref[i] for i, j in path]
x_resampled = [x[j] for i, j in path]
x_ref_repeated = [idx > 0 and path[idx][0] == path[idx - 1][0] for idx in range(len(path))]
x_repeated = [idx > 0 and path[idx][1] == path[idx - 1][1] for idx in range(len(path))]

line_x_ref, = ax.plot(x_ref, color=colors[7], linestyle='-', marker='o', zorder=1)
line_x, = ax.plot(x, color=colors[7], linestyle='-', marker='o', zorder=1)

line_x_ref_dummy, = ax.plot([i for idx, (i, j) in enumerate(path) if x_ref_repeated[idx]], 
                            [x_ref[i] for idx, (i, j) in enumerate(path) if x_ref_repeated[idx]], 
                            color=colors[1], linestyle='', marker='o', zorder=.5)
line_x_dummy, = ax.plot([j for idx, (i, j) in enumerate(path) if x_repeated[idx]], 
                        [x[j] for idx, (i, j) in enumerate(path) if x_repeated[idx]],
                        color=colors[5], linestyle='', marker='o', zorder=.5)


ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([-.5, len(path) - .5])

list_matches = [None] * len(path)
for idx, (i, j) in enumerate(path):
    list_matches[idx], = ax.plot([i, j], [x_ref[i], x[j]], color='k', alpha=.2, zorder=0)

plt.tight_layout()

len_anim = 40
len_pause = 5

ani = animation.FuncAnimation(fig, animate, interval=100, blit=True, save_count=len_anim + 2 * len_pause)
ani.save(
    'fig/dtw_path.gif',
    dpi=100, savefig_kwargs={'pad_inches': 'tight'}
)