import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from tslearn.metrics import dtw, sakoe_chiba_mask, itakura_mask
from tslearn.datasets import CachedDatasets
from utils import set_fig_style


def mask2index_lists(mask):
    index_lists = []
    for i in range(mask.shape[0]):
        index_lists.append(np.flatnonzero(np.isfinite(mask[i])))
    return index_lists


def animate(i):
    list_artists = []
    for d in info:
        for idx_match, pos_match in enumerate(d["matches"][i]):
            d["artists_match"][idx_match].set_xdata([i, pos_match])
            d["artists_match"][idx_match].set_ydata([x_ref[i], x[pos_match]])
            d["artists_match"][idx_match].set_visible(True)
            list_artists.append(d["artists_match"][idx_match])
        for idx_match in range(len(d["matches"][i]), len(d["artists_match"])):
            d["artists_match"][idx_match].set_visible(False)
            list_artists.append(d["artists_match"][idx_match])
        d["artist_scatter"].set_xdata([i])
        d["artist_scatter"].set_ydata([x_ref[i]])
        
        list_artists.append(d["artist_scatter"])

    return list_artists

f = np.zeros((12, ))
f[:4] = -1.
f[4:8] = 1.
f[8:] = -1.

length = 30

fig = plt.figure(figsize=(6, 8))
set_fig_style(fig, font_size=14)
ax = plt.subplot(3, 1, 1)
ax2 = plt.subplot(3, 1, 2)
ax3 = plt.subplot(3, 1, 3)
for cur_ax in fig.axes:
    cur_ax.set_facecolor(fig.patch.get_facecolor())
colors = sns.color_palette("Paired")

x_ref = np.zeros((50, ))
x_ref[5:5+length] = np.sin(np.linspace(0, 2 * np.pi, num=length))

shift = 15
x = np.zeros((50, ))
x[5+shift:5+shift+length] = np.sin(np.linspace(0, 2 * np.pi, num=length))
x -= 3

info = [
    {
        "title": "No global constraint", 
        "axis": ax, 
        "matches": mask2index_lists(np.ones((len(x_ref), len(x)))),
        "artists_match": []
    },
    {
        "title": "Sakoe-Chiba band", 
        "axis": ax2, 
        "matches": mask2index_lists(sakoe_chiba_mask(len(x_ref), len(x), radius=5)),
        "artists_match": []
    },
    {
        "title": "Itakura parallelogram", 
        "axis": ax3, 
        "matches": mask2index_lists(itakura_mask(len(x_ref), len(x), max_slope=2.)),
        "artists_match": []
    }
]

for d in info:
    axis = d["axis"]
    
    axis.plot(x_ref, color=colors[7], linestyle='-')
    axis.plot(x, color=colors[7], linestyle='dotted')
    axis.set_title(d["title"])
    axis.set_yticks([])

    d["artists_match"] = [None] * max([len(l) for l in d["matches"]])
    for idx_match, pos_match in enumerate(d["matches"][0]):
        d["artists_match"][idx_match], = axis.plot([0, pos_match], [x_ref[0], x[pos_match]], 
                                                   color='k', alpha=.2)
    for idx_match in range(len(d["matches"][0]), len(d["artists_match"])):
        d["artists_match"][idx_match], = axis.plot([0, 0], [x_ref[0], x[0]], 
                                                   color='k', alpha=.2)
        d["artists_match"][idx_match].set_visible(False)
    d["artist_scatter"], = axis.plot([0], [x_ref[0]], color=colors[7], marker='o')


plt.tight_layout()

ani = animation.FuncAnimation(fig, animate, interval=90, blit=True, save_count=len(x_ref))
ani.save(
    'fig/dtw_global_constraints.gif',
    dpi=100, savefig_kwargs={'pad_inches': 'tight'}
)