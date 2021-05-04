import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import numpy as np
from tslearn.metrics import itakura_mask
from utils import set_fig_style


def get_color_matrix(m, fig):
    m[m == np.inf] = 1.
    np.fill_diagonal(m, .5)
    bg_color = fig.patch.get_facecolor()
    grays = sns.color_palette("Greys")

    mat_colors = np.zeros((sz, sz, 3))
    mat_colors[m == 1., :] = bg_color[:-1]
    mat_colors[m == 0., :] = grays[len(grays) // 2 - 2]
    mat_colors[m == .5, :] = grays[-1]
    
    return mat_colors[:, ::-1]


def clean_plot(ax, sz):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([-.7, sz - .4])
    ax.set_ylim([-.7, sz - .4])
    ax.axis('off')
    plt.tight_layout()
    
def animate(i):
    list_s_values = [1.5, 2., 3.]
    
    m = itakura_mask(sz, sz, max_slope=list_s_values[i])
    mat_colors = get_color_matrix(m, fig)
    ax_im.set_data(mat_colors)
    ax_text.set_text(f"$s={list_s_values[i]}$")

    return [ax_im, ax_text]

fig, ax = plt.subplots(figsize=(6, 6))
set_fig_style(fig, font_size=32)
bg_color = fig.patch.get_facecolor()
grays = colors = sns.color_palette("Greys")


sz = 20
m = itakura_mask(sz, sz, max_slope=2.)
mat_colors = get_color_matrix(m, fig)
ax_im = ax.imshow(mat_colors)
ax_text = ax.text(s="$s=2.$", x=.7 * sz, y=.9 * sz)
clean_plot(ax, sz)

ani = animation.FuncAnimation(fig, animate, interval=1000, blit=True, save_count=3)
ani.save(
    'fig/itakura_matrices.gif',
    dpi=100, savefig_kwargs={'pad_inches': 'tight'}
)