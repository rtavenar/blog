import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import numpy as np
from tslearn.metrics import sakoe_chiba_mask
from utils import set_fig_style, export_animation


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
    list_r_values = [3, 5, 10]
    
    m = sakoe_chiba_mask(sz, sz, radius=list_r_values[i])
    mat_colors = get_color_matrix(m, fig)
    ax_im.set_data(mat_colors)
    ax_text.set_text(f"$r={list_r_values[i]}$")

    return [ax_im, ax_text]

fig, ax = plt.subplots(figsize=(6, 6))
set_fig_style(fig, font_size=32)
bg_color = fig.patch.get_facecolor()
grays = colors = sns.color_palette("Greys")


sz = 20
m = sakoe_chiba_mask(sz, sz, radius=3)
mat_colors = get_color_matrix(m, fig)
ax_im = ax.imshow(mat_colors)
ax_text = ax.text(s="$r=3$", x=.7 * sz, y=.9 * sz)
clean_plot(ax, sz)

ani = animation.FuncAnimation(fig, animate, interval=1000, blit=True, save_count=3)
export_animation(ani, 'fig/sakoechiba_matrices', fps=1)
