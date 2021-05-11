import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tslearn.metrics import dtw_path
from utils import set_fig_style

def softmin(a, gamma):
    a_gamma = -a / gamma
    max_a_gamma = np.max(a_gamma)
    a_gamma -= max_a_gamma
    s = np.sum(np.exp(a_gamma))
    return - gamma * (np.log(s) + max_a_gamma)

fig, ax = plt.subplots()
set_fig_style(fig, font_size=14)
colors = sns.color_palette("Blues")[2:]

a = np.linspace(-1.5, 1.5, 100)
gamma_values = [.01, .1, .3, 1.]

ax.axvline(x=0, linestyle='--', color='k', alpha=.5)
ax.axhline(y=0, linestyle='--', color='k', alpha=.5)
for i, gamma in enumerate(gamma_values):
    ax.plot(a, [softmin(np.array([-xi, xi]), gamma) for xi in a], color=colors[i], label=f"$\gamma={gamma}$")
plt.legend(loc="upper right")
plt.ylabel("$min^\gamma(-a, a)$")
plt.xlabel("$a$")
plt.tight_layout()
plt.savefig('fig/soft_min.svg')
