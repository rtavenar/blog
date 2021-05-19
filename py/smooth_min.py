import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tslearn.metrics import dtw_path
from utils import set_fig_style

def softmin(a, gamma):
    exp_a_gamma = np.exp(-a / gamma)
    denom = np.sum(exp_a_gamma)
    return np.sum(a * exp_a_gamma) / denom


fig = plt.figure()
set_fig_style(fig, font_size=14)
ax = fig.gca()
for cur_ax in fig.axes:
    cur_ax.set_facecolor(fig.patch.get_facecolor())
colors = sns.color_palette("Blues")[1:]

a = np.linspace(-1.5, 1.5, 101)
gamma_values = [.1, .3, .6, 1.]

ax.axvline(x=0, linestyle='--', color='k', alpha=.5)
ax.axhline(y=0, linestyle='--', color='k', alpha=.5)
ax.plot(a, [np.min(np.array([-xi, xi])) for xi in a], color='k', alpha=.5, linewidth=3)
for i, gamma in enumerate(gamma_values):
    ax.plot(a, [softmin(np.array([-xi, xi]), gamma) for xi in a], color=colors[-i-1], label=f"$\gamma={gamma}$")
plt.legend(loc="upper right")
plt.ylabel("$\\text{smoothMin}^\gamma(-a, a)$", fontsize=16)
plt.xlabel("$a$", fontsize=16)
plt.tight_layout()
plt.savefig('fig/smooth_min.svg')
