import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from tslearn.datasets import CachedDatasets
from tslearn.metrics import soft_dtw, SoftDTW, SquaredEuclidean
from utils import set_fig_style, export_animation


def soft_dtw_grad(x, y, gamma):    
    D = SquaredEuclidean(x.reshape((-1, 1)), y.reshape((-1, 1)))
    sdtw = SoftDTW(D, gamma=gamma)
    prev_loss = sdtw.compute()
    A_gamma = sdtw.grad()
    grad = D.jacobian_product(A_gamma)
    return prev_loss, grad


def animate(i):
    line_x_t.set_ydata(list_x[i])
    line_x_t2.set_ydata(list_x2[i])
    # text1.set_text("$\gamma=0.1$, epoch %2d" % i)
    # text2.set_text("$\gamma=10$, epoch %2d" % i)
    scatter.set_xdata([i])
    scatter.set_ydata([losses[i]])
    scatter2.set_xdata([i])
    scatter2.set_ydata([losses2[i]])

    return [line_x_t, line_x_t2, scatter, scatter2]

fig = plt.figure(figsize=(6, 6))
set_fig_style(fig, font_size=14)
ax = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)

ax3 = fig.add_axes([.1, .287, 0.2, 0.15])
ax4 = fig.add_axes([.1, .77, 0.2, 0.15])
for cur_ax in fig.axes:
    cur_ax.set_facecolor(fig.patch.get_facecolor())
colors = sns.color_palette("Paired")

np.random.seed(0)
X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
x_ref = X_train[3]

eta = 1e-1
epochs = 7
gamma = .1
list_x = [x_ref]
losses = []
for e in range(epochs):
    x_t = list_x[-1]
    loss, grad = soft_dtw_grad(x_t, x_ref, gamma)
    list_x.append(
        x_t - eta * grad
    )
    losses.append(loss)
losses.append(soft_dtw(list_x[-1], x_ref, gamma))

gamma = 10.
losses2 = []
list_x2 = [x_ref]
for e in range(epochs):
    x_t = list_x2[-1]
    loss, grad = soft_dtw_grad(x_t, x_ref, gamma)
    list_x2.append(
        x_t - eta * grad
    )
    losses2.append(loss)
losses2.append(soft_dtw(list_x2[-1], x_ref, gamma))

line_x_ref, = ax.plot(x_ref.ravel(), color=colors[6], linestyle='-', linewidth=3, label="$x_\\text{ref}$")
line_x_t, = ax.plot(list_x[0].ravel(), color=colors[7], linestyle='-', linewidth=3, label="$\min_x \\text{soft-}DTW^\gamma (x, x_\\text{ref})$ estimate")
# text1 = ax.text(s="$\gamma=0.1$, epoch %2d" % 0, x=0, y=.8)
ax.set_xticks([])
ax.set_yticks([])
ax.legend(loc='lower right')
ax.set_title("$\gamma=0.1$")

line_x_ref2, = ax2.plot(x_ref.ravel(), color=colors[6], linestyle='-', linewidth=3, label="$x_\\text{ref}$")
line_x_t2, = ax2.plot(list_x2[0].ravel(), color=colors[7], linestyle='-', linewidth=3, label="$\min_x \\text{soft-}DTW^\gamma (x, x_\\text{ref})$ estimate")
# text2 = ax2.text(s="$\gamma=10$, epoch %2d" % 0, x=0, y=.8)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.legend(loc='lower right')
ax2.set_title("$\gamma=10$")

ax3.plot(np.arange(epochs), losses[:-1], color=colors[1], linestyle='-', linewidth=3)
scatter, = ax3.plot([0], losses[0], color=colors[1], linestyle='', marker='o')
ax3.set_xlabel("Epoch")
ax3.set_ylabel("soft-DTW")
ax3.set_yticks([])

ax4.plot(np.arange(epochs), losses2[:-1], color=colors[1], linestyle='-', linewidth=3)
scatter2, = ax4.plot([0], losses2[0], color=colors[1], linestyle='', marker='o')
ax4.set_xlabel("Epoch")
ax4.set_ylabel("soft-DTW")
ax4.set_yticks([])

plt.tight_layout()

ani = animation.FuncAnimation(fig, animate, interval=1000, blit=True, save_count=epochs)
export_animation(ani, 'fig/softdtw_denoising', fps=1)
