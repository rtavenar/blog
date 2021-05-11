import os
from matplotlib import rc, rcParams, colors, animation


def set_fig_style(fig, font_size=22):
    os.environ['PATH'] += ':/Library/TeX/texbin/'  # Path to your latex install
    blog_color = colors.hex2color("#f9fafb")
    fig.patch.set_facecolor(blog_color)
    for ax in fig.axes:
        ax.set_facecolor(blog_color)
    rcParams.update({
        'text.latex.preamble': r'\usepackage{newtxmath}',
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "New Century Schoolbook", "Bookman", "Computer Modern Roman"],
        "font.size": font_size
    })
    rc('text', usetex=True)


def export_animation(anim, fname, ext=None, fps=5):
    if ext is None:
        # If ext is None, all GIF+video figures are generated
        for ext in ["gif", "mp4", "webm"]:
            export_animation(anim, fname, ext=ext)
    fname_with_ext = f"{fname}.{ext}"
    if ext == "gif":
        anim.save(fname_with_ext, dpi=100, savefig_kwargs={'pad_inches': 'tight'})
    elif ext == "html":
        rcParams["animation.frame_format"] = "svg"
        html_widget = anim.to_jshtml(default_mode="loop")
        open(fname_with_ext, "w").write(html_widget)  # Generated files are huge...
    elif ext in ["mp4", "webm"]:
        Writer = animation.writers['ffmpeg']
        if ext == "mp4":
            writer = Writer(fps=fps, codec="libx265")
        else:
            writer = Writer(fps=fps, codec='libvpx-vp9')
        anim.save(fname_with_ext, writer=writer)
    else:
        print(f"Unrecognized extension: {ext}")