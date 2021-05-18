import ternary
import matplotlib.pyplot as plt
import math
import numpy as np
import itertools
from utils import set_fig_style

def shannon_entropy(p):
    """Computes the Shannon Entropy at a distribution in the simplex."""
    s = 0.
    for i in range(len(p)):
        try:
            s += p[i] * math.log(p[i])
        except ValueError:
            continue
    return -1. * s


level = [0.25, 0.5, 0.75, 1.]       # values for contours

# === prepare coordinate list for contours
x_range = np.arange(0, 1.01, 0.01)   # ensure that grid spacing is small enough to get smooth contours
coordinate_list = np.asarray(list(itertools.product(x_range, repeat=2)))
coordinate_list = np.append(coordinate_list, (1 - coordinate_list[:, 0] - coordinate_list[:, 1]).reshape(-1, 1), axis=1)

# === calculate data with coordinate list
data_list = [shannon_entropy(point) for point in coordinate_list]
data_list = np.asarray(data_list)
data_list[np.sum(coordinate_list[:, 0:2], axis=1) > 1] = np.nan  # remove data outside triangle

# === reshape coordinates and data for use with pyplot contour function
x = coordinate_list[:, 0].reshape(x_range.shape[0], -1)
y = coordinate_list[:, 1].reshape(x_range.shape[0], -1)

h = data_list.reshape(x_range.shape[0], -1)

# === use pyplot to calculate contours
contours = plt.contour(x, y, h, level)  # this needs to be BEFORE figure definition
plt.clf()  # makes sure that contours are not plotted in carthesian plot


scale = 60
figure, tax = ternary.figure(scale=scale)
figure.set_size_inches(5, 3)
set_fig_style(figure, font_size=14, bg_color_html="#eeeeee")

# === plot contour lines
for ii, contour in enumerate(contours.allsegs):
    for jj, seg in enumerate(contour):
        tax.plot(seg[:, 0:2] * scale, color='k', linestyle='--')

tax.heatmapf(shannon_entropy, boundary=True, style="triangular")
tax.boundary(linewidth=2.0)

tax.get_axes().text(s="$p = (1, 0, 0)$", x=5, y=-1, va='top', ha='right')
tax.get_axes().text(s="$p = (0, 1, 0)$", x=55, y=-1, va='top', ha='left')
tax.get_axes().text(s="$p = (0, 0, 1)$", x=30, y=55, va='baseline', ha='center')

tax.clear_matplotlib_ticks()
tax.get_axes().set_xlim(-15, scale + 20)
tax.get_axes().set_ylim(-5, scale)
tax.get_axes().axis('off')

plt.tight_layout()
plt.savefig('fig/entropy.svg')
