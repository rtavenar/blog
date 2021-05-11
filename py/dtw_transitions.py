import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import seaborn as sns
from utils import set_fig_style
import matplotlib

matplotlib.rcParams['lines.markersize'] = 30
matplotlib.rcParams['lines.linewidth'] = matplotlib.rcParams['axes.linewidth'] = 2

fig = plt.figure(figsize=(8, 8))
set_fig_style(fig, font_size=24)
ax = fig.gca()
for cur_ax in fig.axes:
    cur_ax.set_facecolor(fig.patch.get_facecolor())
colors = sns.color_palette("tab10")
grey = sns.color_palette("Greys")[2]

ax.set_xticks([])
ax.set_yticks([])
for pos in range(2):
    ax.axvline(x=pos + .5, linestyle='-', color='k', zorder=-1)
    ax.axhline(y=pos + .5, linestyle='-', color='k', zorder=-1)
ax.set_xlim([-.5, 2.5])
ax.set_ylim([-.5, 2.5])
ax.text(s="$i$", x=2, y=-.75, ha='center', color='k', zorder=-.25, fontsize=36)
ax.text(s="$i-1$", x=1, y=-.75, ha='center', color='k', zorder=-.25, fontsize=36)
ax.text(s="$i-2$", x=0, y=-.75, ha='center', color='k', zorder=-.25, fontsize=36)
ax.text(s="$j$", x=-.65, y=0, ha='right', va='center', color='k', zorder=-.25, fontsize=36)
ax.text(s="$j-1$", x=-.65, y=1, ha='right', va='center', color='k', zorder=-.25, fontsize=36)
ax.text(s="$j-2$", x=-.65, y=2, ha='right', va='center', color='k', zorder=-.25, fontsize=36)

ax.plot([0, 0, 0, 1, 2], [2, 1, 0, 2, 2], marker='o', color=grey, linestyle='', zorder=0)
for i_source in range(3):
    for j_source in range(3):
        if (i_source == 0) or (i_source == 1 and j_source < 2):
            ax.arrow(i_source, 2 - j_source, 0.9, 0, length_includes_head=True, head_width=.1, color=grey, linewidth=10, zorder=-.5)
        if (j_source == 0) or (j_source == 1 and i_source < 2):
            ax.arrow(i_source, 2 - j_source, 0, -0.9, length_includes_head=True, head_width=.1, color=grey, linewidth=10, zorder=-.5)
        if (i_source == 0 and j_source < 2) or (i_source == 1 and j_source == 0):
            ax.arrow(i_source, 2 - j_source, 0.9, -0.9, length_includes_head=True, head_width=.1, color=grey, linewidth=10, zorder=-.5)

ax.arrow(1, 1, 0.9, -.9, length_includes_head=True, head_width=.1, color=colors[2], linewidth=10, zorder=1)
ax.plot([1], [1], marker='o', color=colors[2], linestyle='', zorder=0)
# ax.text(s="$\gamma_{i-1,j-1}$", x=1, y=1.25, ha='center', color=colors[2], backgroundcolor=fig.patch.get_facecolor(), zorder=-.25)

ax.arrow(1, 0, 0.9, 0, length_includes_head=True, head_width=.1, color=colors[0], linewidth=10, zorder=1)
ax.plot([1], [0], marker='o', color=colors[0], linestyle='', zorder=0)
# ax.text(s="$\gamma_{i-1,j}$", x=1, y=-.25, ha='center', color=colors[0], backgroundcolor=fig.patch.get_facecolor(), zorder=-.25)

ax.arrow(2, 1, 0, -0.9, length_includes_head=True, head_width=.1, color=colors[3], linewidth=10, zorder=1)
ax.plot([2], [1], marker='o', color=colors[3], linestyle='', zorder=0)
# ax.text(s="$\gamma_{i,j-1}$", x=2, y=1.25, ha='center', color=colors[3], backgroundcolor=fig.patch.get_facecolor(), zorder=-.25)

ax.plot([2], [0], marker='o', color=colors[1], linestyle='', zorder=2)
ax.text(s="$\gamma_{i,j}$", x=2, y=-.25, ha='center', color=colors[1], backgroundcolor=fig.patch.get_facecolor(), zorder=-.25, fontsize=48)

plt.tight_layout()
plt.savefig('fig/dtw_transitions.svg')
