import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import ConnectionPatch
import matplotlib.animation as animation
import seaborn as sns
from tslearn.metrics import soft_dtw_alignment
from utils import set_fig_style, export_animation


def animate(i):
    line_text.set_text(f"$\gamma={gammas[i]}$")
    line_gram.set_data(a_gammas[i])
    
    return [line_text, line_gram]


fig = plt.figure(figsize=(6, 6))
set_fig_style(fig, font_size=26)
matplotlib.rcParams['lines.markersize'] = 4

# definitions for the axes
left, bottom = 0.01, 0.1
w_ts = 0.2
h_ts = w_ts
left_h = left + w_ts + 0.02
width = 0.6
height = width
bottom_h = bottom + height + 0.02

rect_s_y = [left, bottom, w_ts, height]
rect_gram = [left_h, bottom, width, height]
rect_s_x = [left_h, bottom_h, width, h_ts]

ax_gram = plt.axes(rect_gram)
ax_s_x = plt.axes(rect_s_x)
ax_s_y = plt.axes(rect_s_y)

for cur_ax in fig.axes:
    cur_ax.set_facecolor(fig.patch.get_facecolor())
colors = sns.color_palette("Paired")

sz = 30
length = 15

x_ref = np.zeros((sz, ))
x_ref[2:2+length] = np.sin(np.linspace(0, 2 * np.pi, num=length))

shift = 6
x = np.zeros((sz, ))
x[2+shift:2+shift+length] = np.sin(np.linspace(0, 2 * np.pi, num=length))

gammas = [10**i for i in range(-2, 3)]
sdtw_res = [soft_dtw_alignment(x, x_ref, gamma=gamma) for gamma in gammas]
a_gammas = [res[0] for res in sdtw_res]
dists = [res[1] for res in sdtw_res]

ax_gram.set_xticks([])
ax_gram.set_yticks([])
line_gram = ax_gram.imshow(a_gammas[0], origin="upper")
line_text = ax_gram.text(s=f"$\gamma={gammas[0]}$", x=.6*sz, y=.15*sz, color="w")

ax_s_x.plot(np.arange(sz), x_ref, color=colors[7], linestyle='-', marker='o')
con = ConnectionPatch(xyA=[-1, 0],
                      xyB=[sz + 2, 0],
                      coordsA=ax_s_x.transData, coordsB=ax_s_x.transData,
                      color='k', linestyle='-', arrowstyle='->')
ax_s_x.add_artist(con)
ax_s_x.axis("off")
ax_s_x.set_xlim([-.5, sz - .5])

ax_s_y.plot(- x, np.arange(sz)[::-1], color=colors[7], linestyle='-', marker='o')
con = ConnectionPatch(xyA=[0, sz],
                      xyB=[0, - 3],
                      coordsA=ax_s_y.transData, coordsB=ax_s_y.transData,
                      color='k', linestyle='-', arrowstyle='->')
ax_s_y.add_artist(con)
ax_s_y.axis("off")
ax_s_y.set_ylim([-.5, sz - .5])

ani = animation.FuncAnimation(fig, animate, interval=1000, blit=True, save_count=len(gammas))
export_animation(ani, 'fig/a_gamma_tweet', fps=1, ext='gif')
