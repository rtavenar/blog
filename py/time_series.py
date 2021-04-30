import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import matplotlib.animation as animation
import seaborn as sns
from utils import set_fig_style


def animate(i):
    marker.set_xdata([i])
    marker.set_ydata([x_ref[i]])
    text.set_x(i)
    text.set_y(x_ref[i] + .2)
    text.set_text("$x_{%d}$" % i)

    return [marker, text]

f = np.zeros((12, ))
f[:4] = -1.
f[4:8] = 1.
f[8:] = -1.

fig, ax = plt.subplots(figsize=(8, 4))
set_fig_style(fig, font_size=14)
for cur_ax in fig.axes:
    cur_ax.set_facecolor(fig.patch.get_facecolor())
colors = sns.color_palette("Paired")

x_ref = np.zeros((40, ))
length = 20
x_ref[5:5+length] = np.sin(np.linspace(0, 2 * np.pi, num=length))

ax.plot(x_ref, color=colors[7], linestyle='dashed')
ax.plot(x_ref, color=colors[6], linestyle='', marker='o')


marker, = ax.plot([0], [x_ref[0]], 
                  color=colors[7], linestyle='', marker='o')
text = ax.text(x=0, y=x_ref[0] + .2, s="$x_{{0}}$", fontsize=16)
con = ConnectionPatch(xyA=[-1, 0],
                      xyB=[len(x_ref) + 2, 0],
                      coordsA=ax.transData, coordsB=ax.transData,
                      color='k', linestyle='-', arrowstyle='->', 
                      zorder=-1)
ax.add_artist(con)
ax.set_xticks([])
ax.set_xlim([-1, len(x_ref) + 3])
ax.set_ylim([-1.1, 1.3])

plt.tight_layout()

ani = animation.FuncAnimation(fig, animate, blit=True, 
                              interval=250, save_count=len(x_ref))
ani.save(
    'fig/time_series.gif',
    dpi=100, savefig_kwargs={'pad_inches': 'tight'}
)