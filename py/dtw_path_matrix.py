import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import ConnectionPatch
import seaborn as sns
from tslearn.metrics import dtw_path
from utils import set_fig_style


fig = plt.figure(figsize=(8, 4))
set_fig_style(fig, font_size=26)
matplotlib.rcParams['lines.markersize'] = 4

# definitions for the axes
left, bottom = 0.01, 0.1
w_ts = 0.1
h_ts = 2 * w_ts
left_h = left + w_ts + 0.02
width = 0.3
height = 2 * width
bottom_h = bottom + height + 0.02
h_align = 0.4
w_align = width + 0.1
bottom_align = .25

rect_s_y = [left, bottom, w_ts, height]
rect_gram = [left_h, bottom, width, height]
rect_s_x = [left_h, bottom_h, width, h_ts]
rect_align = [left_h + width + 0.1, bottom_align, w_align, h_align]

ax_gram = plt.axes(rect_gram)
ax_s_x = plt.axes(rect_s_x)
ax_s_y = plt.axes(rect_s_y)
ax_align = plt.axes(rect_align)

for cur_ax in fig.axes:
    cur_ax.set_facecolor(fig.patch.get_facecolor())
colors = sns.color_palette("Paired")
colors_grey = sns.color_palette("Greys")

sz = 20
length = 15

x_ref = np.zeros((sz, ))
x_ref[2:2+length] = np.sin(np.linspace(0, 2 * np.pi, num=length))

shift = 3
x = np.zeros((sz, ))
x[2+shift:2+shift+length] = np.sin(np.linspace(0, 2 * np.pi, num=length))

path, _ = dtw_path(x_ref, x)

ax_gram.set_xlim([-.5, sz - .5])
ax_gram.set_ylim([-.5, sz - .5])
ax_gram.set_xticks([])
ax_gram.set_yticks([])
# ax_gram.vlines(x=np.arange(sz) - .5, ymin=-.5, ymax=sz - .5, color='k', linewidth=.5)
# ax_gram.hlines(y=np.arange(sz) - .5, xmin=-.5, xmax=sz - .5, color='k', linewidth=.5)
ax_gram.plot([i for (i, j) in path], [sz - j - 1 for (i, j) in path], 
             color=colors_grey[2], marker='o', linestyle='-', markersize=5)
ax_gram.text(s="$A_\pi$", x=.75 * sz, y=.75 * sz)

idx_in_path = 9
subpath = path[idx_in_path:idx_in_path + 1]
ax_gram.plot([i for (i, j) in subpath], [sz - j - 1 for (i, j) in subpath], 
             color="k", marker='o', linestyle='-', markersize=5)
i, j = path[idx_in_path]
con = ConnectionPatch(xyA=[-x[j], np.arange(sz)[::-1][j]], 
                      xyB=[i, sz - j - 1], 
                      coordsA="data", coordsB="data",
                      axesA=ax_s_y, axesB=ax_gram,
                      color=colors_grey[-2], linestyle=(0, (4, 6)))
ax_gram.add_artist(con)
con = ConnectionPatch(xyA=[i, x_ref[i]],
                      xyB=[i, sz - j - 1],
                      coordsA="data", coordsB="data",
                      axesA=ax_s_x, axesB=ax_gram,
                      color=colors_grey[-2], linestyle=(0, (4, 6)))
ax_gram.add_artist(con)

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

x -= 3
ax_align.plot(x_ref, color=colors[7], linestyle='-', marker='o', zorder=1)
ax_align.plot(x, color=colors[7], linestyle='-', marker='o', zorder=1)
for idx, (i, j) in enumerate(path):
    ax_align.plot([i, j], [x_ref[i], x[j]], color=colors_grey[2] if idx != idx_in_path else 'k', 
                  zorder=0)
ax_align.set_xticks([])
ax_align.set_yticks([])

plt.savefig('fig/dtw_path_matrix.svg')