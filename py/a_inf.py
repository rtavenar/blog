import numpy
import matplotlib.pyplot as plt
from utils import set_fig_style

def delannoy(m, n):
  numbers = numpy.zeros((m + 1, n + 1), dtype=numpy.float64)
  numbers[0, 0] = 1
  for i in range(1, m + 1):
    for j in range(1, n + 1):
      numbers[i, j] = numbers[i - 1, j - 1] + numbers[i, j - 1] + numbers[i - 1, j]
  return numbers[1:, 1:]

m = n = 30
delannoy_numbers = delannoy(m, n)
weight_matrix = delannoy_numbers * delannoy_numbers[::-1, ::-1] / delannoy_numbers[-1, -1]

fig = plt.figure()
set_fig_style(fig, font_size=16)

plt.imshow(weight_matrix)
plt.colorbar()
# plt.title("$A_\infty$", fontsize=36)
plt.xticks([0, 10, 20])
plt.yticks([0, 10, 20])
plt.tight_layout()
plt.savefig("fig/a_inf.svg")