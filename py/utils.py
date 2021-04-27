import os
from matplotlib import rc, rcParams, colors

def set_fig_style(fig, font_size=22):
    os.environ['PATH'] += ':/Library/TeX/texbin/'  # Path to your latex install
    rcParams['text.latex.preamble'] = r'\usepackage{newtxmath}'
    blog_color = colors.hex2color("#f9fafb")
    fig.patch.set_facecolor(blog_color)
    for ax in fig.axes:
        ax.set_facecolor(blog_color)
    rc('font', family="serif", serif="Times", size=font_size)
    rc('text', usetex=True)