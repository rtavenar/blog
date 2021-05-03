---
title: An introduction to Dynamic Time Warping
language: en
author: Romain Tavenard
date: 2021/04/21
rights: Creative Commons CC BY-NC-SA
bibliography: dtw.bib
reference-section-title: References
link-citations: true
---

# Alignment-based metrics

This post is the first in a series about assessing similarity between time series.
More specifically, we will be interested in _alignment-based metrics_,<label for="sn-1" class="sidenote-toggle sidenote-number"></label>
<input type="checkbox" id="sn-1" class="sidenote-toggle" />
<span class="sidenote">Here we use the term "metrics" in a pretty unformal manner, that is an equivalent of "similarity measure".</span> 
that rely on a temporal alignment of the series in order to assess their similarity.

Before entering into more details about these metrics, let us define our base objects: time series.
In the following, a time series is a sequence of features: $x = \left(x_0, \dots, x_{n-1}\right)$.
All features from a time series lie in the same space $\mathbb{R}^p$.
Below is an example univariate<label for="sn-2" class="sidenote-toggle sidenote-number"></label>
<input type="checkbox" id="sn-2" class="sidenote-toggle" />
<span class="sidenote">A time series is said univariate if all its feature vectors are monodimensional ($p=1$).</span> time series:

<figure>
    <img src="fig/time_series.gif" alt="An example univariate time series" width="80%" />
    <figcaption> 
        Example univariate time series. 
        The horizontal axis is the time axis and the vertical one is dedicated to (univariate) feature values.
    </figcaption>
</figure>

Let us now illustrate the typical behavior of alignment-based metrics with an example.

<figure>
    <img src="fig/dtw_vs_euc.svg" alt="DTW vs Euclidean distance" width="100%" />
    <figcaption> 
        Comparison between DTW and Euclidean distance.
    </figcaption>
</figure>

Here, we are computing similarity between two time series using either Euclidean distance (left) or Dynamic Time Warping (DTW, right), which is an instance of alignment-based metric, that we will present in more details later in this post.
In both cases, the returned similarity is the sum of distances over all matches (represented by gray lines here).
Note how DTW matches distinctive patterns of the time series, which is likely to result in a more sound similarity assessment.

<figure>
    <img src="fig/kmeans.svg" alt="Euclidean k-means" width="100%" />
    <figcaption>
        $k$-means clustering with Euclidean distance. 
        Each subfigure represents series from a given cluster and their centroid (in red).
    </figcaption>
</figure>

The figure above is the result of a $k$-means clustering that uses Euclidean
distance as a base metric.
One issue with this metric is that it is not invariant to time shifts, while
the dataset at stake clearly holds such invariants.

When using a shift-invariant similarity measure (discussed in our
{ref}`sec:dtw` section) at the core of $k$-means, one gets:

<figure>
    <img src="fig/kmeans_dtw.svg" alt="Euclidean k-means" width="100%" />
    <figcaption>
        $k$-means clustering with Dynamic Time Warping. Each subfigure
        represents series from a given cluster and their centroid (in red).
    </figcaption>
</figure>

This part of the course tackles the definition of adequate similarity
measures for time series and their use at the core of machine learning methods.

# Dynamic Time Warping

Dynamic Time Warping (DTW) is a similarity measure between time series.
It has been introduced independently in the literature by [@vintsyuk1968speech] and [@sakoe1978dynamic],
in both cases for speech applications.<label for="sn-3" class="sidenote-toggle sidenote-number"></label>
<input type="checkbox" id="sn-3" class="sidenote-toggle" />
<span class="sidenote">Note that, in this series of posts, we will stick to the formalism from [@sakoe1978dynamic], which is more standard in the literature.</span>

Consider two time series $x$ and
${x}^\prime$ of respective lengths $n$ and
$m$.

Here, all elements $x_i$ and $x^\prime_j$ are assumed to lie in the same
$p$-dimensional space and the exact timestamps at which observations occur are
disregarded: only their ordering matters.

Dynamic Time Warping, in this context, is equivalent to minimizing Euclidean distance between aligned time
series under all admissible temporal alignments, as illustrated in the Figure below.

<figure>
    <img src="fig/dtw_path.gif" alt="DTW as minimum Euclidean distance up to a realignment" width="60%" />
    <figcaption> 
        Dynamic Time Warping seeks for the temporal alignment (gray lines) that minimizes Euclidean distance between
        aligned series.
        Red and blue dots correspond to repetitions of time series elements induced by the considered temporal alignment.
    </figcaption>
</figure>

## Problem formulation

More formally, the optimization problem writes:

\begin{equation}
DTW_q({x}, {x}^\prime) =
    \min_{\pi \in \mathcal{A}({x}, {x}^\prime)}
        \left( \sum_{(i, j) \in \pi} d(x_i, x^\prime_j)^q \right)^{\frac{1}{q}}
\label{eq:dtw}
\end{equation}

Here, an **alignment path** $\pi$ of length $K$ is a sequence of $K$ index pairs
$\left((i_0, j_0), \dots , (i_{K-1}, j_{K-1})\right)$ and $\mathcal{A}({x}, {x}^\prime)$ is the set of all admissible paths.
In order to be considered admissible, a path should satisfy the following conditions:

* Beginning (resp. end) of time series are matched together: 

  * $\pi_0 = (0, 0)$ 
  * $\pi_{K-1} = (n - 1, m - 1)$

* The sequence is monotonically increasing in both $i$ and $j$ and all time series indexes should appear at least once, which can be written:

  * $i_{k-1} \leq i_k \leq i_{k-1} + 1$
  * $j_{k-1} \leq j_k \leq j_{k-1} + 1$

### Dot product notation

Another way to represent a DTW path is to use a binary matrix whose non-zero entries are those corresponding to a 
matching between time series elements.
This representation is related to the index sequence representation used above through:

\begin{equation}
(A_\pi)_{i,j} = \left\{ \begin{array}{rl} 1 & \text{ if } (i, j) \in \pi \\
                                      0 & \text{ otherwise}
                        \end{array} \right. .
\end{equation}

This is illustrated in the Figure below where the binary matrix is represented as a grid on which the
DTW path $\pi$ is superimposed, and each dot on the grid corresponds to a non-zero entry in $A_\pi$:

<figure>
    <img src="fig/dtw_path_matrix.svg" alt="DTW path as a matrix" width="100%" />
    <figcaption> 
        Dynamic Time Warping path represented as a binary matrix. 
        Each dot on the path indicates the matching of an element in $x$ with an element in $x^\prime$.
    </figcaption>
</figure>

Using matrix notation, Dynamic Time Warping can be written as the minimization of a dot product between matrices:

\begin{equation*}
DTW_q(\mathbf{x}, \mathbf{x}^\prime) =
    \min_{\pi \in \mathcal{A}({x}, {x}^\prime)}
        \left\langle A_\pi,  D_q({x}, {x}^\prime) \right\rangle^{\frac{1}{q}}
\end{equation*}

where $D_q(\mathbf{x}, \mathbf{x}^\prime)$ stores distances
$d(x_i, x^\prime_j)$ at the power $q$.

## Algorithmic Solution

An exact solution to this optimization problem can be found using Dynamic Programming.
The essence of dynamic programming is to link the solution of a given problem to solutions of (easier) sub-problems.
Once this link is known, one can solve the original problem by recursively solving required sub-problems and storing their solutions.

In the case of DTW, we need to rely on the following quantity:

$$
    \gamma_{i,j} = DTW_q({x}_{\rightarrow i}, {x}^\prime_{\rightarrow j})^q
$$

where the notation ${x}_{\rightarrow i}$ denotes time series ${x}$ observed up to timestamp $i$ (included).
Then, we can observe that:

$$
\begin{aligned}
\gamma_{i,j} 
    &= \min_{\pi \in \mathcal{A}({x}_{\rightarrow i}, {x}^\prime_{\rightarrow j})}
        \sum_{(k, l) \in \pi} d(x_k, x^\prime_l)^q \\
    &\stackrel{*}{=} d(x_i, x^\prime_j)^q +
        \min_{\pi \in \mathcal{A}({x}_{\rightarrow i}, {x}^\prime_{\rightarrow j})}
            \sum_{(k, l) \in \pi[:-1]} d(x_k, x^\prime_l)^q \\
    &\stackrel{**}{=} d(x_i, x^\prime_j)^q +
        \min (\gamma_{i-1, j}, \gamma_{i, j-1}, \gamma_{i-1, j-1})
\end{aligned}
$$

Note that $(*)$ comes from the constraints on admissible paths $\pi$: the last element on an admissible path needs to match the last elements of the series;
Note also that $(**)$ comes from the contiguity conditions on the admissible paths.
Indeed, a path that would align time series ${x}_{\rightarrow i}$ and ${x}^\prime_{\rightarrow j}$ necessarily encapsulates either:

* a path that would align time series ${x}_{\rightarrow i - 1}$ and ${x}^\prime_{\rightarrow j}$, or
* a path that would align time series ${x}_{\rightarrow i}$ and ${x}^\prime_{\rightarrow j - 1}$, or
* a path that would align time series ${x}_{\rightarrow i - 1}$ and ${x}^\prime_{\rightarrow j - 1}$,

as illustrated in the Figure below:

<figure>
    <img src="fig/dtw_transitions.png" alt="DTW transitions" width="60%" />
    <figcaption> 
        Permitted DTW transitions.
    </figcaption>
</figure>

**TODO: fig DTW transitions**

This implies that filling a matrix that would store $\gamma_{i,j}$ terms row-by-row<label for="sn-row-wise" class="sidenote-toggle sidenote-number"></label>
<input type="checkbox" id="sn-row-wise" class="sidenote-toggle" />
<span class="sidenote">In practice, the matrix could be filled column-by-column too. The important part is that the terms `gamma[i-1, j]`, `gamma[i, j-1]` and `gamma[i-1, j-1]` are accessible when computing `gamma[i, j]`. When vectorizing code is of importance, an even better strategy is to compute the `gamma` terms one anti-diagonal at a time [@tralie2020exact].</span>
 is sufficient to retrieve 
$DTW_q({x}, {x}^\prime)$ as ${\gamma_{n-1, m-1}}^{\frac{1}{q}}$.

These observations result in the following $O(mn)$ algorithm to compute the exact optimum for DTW 
(assuming computation of $d(\cdot,\cdot)$ is $O(1)$):

<pre>
  <code class="language-python">
def dtw(x, x_prime, q=2):
  for i in range(len(x)):
    for j in range(len(x_prime)):
      gamma[i, j] = d(x[i], x_prime[j]) ** q
      if i > 0 or j > 0:
        gamma[i, j] += min(
          gamma[i-1, j  ] if i > 0             else inf,
          gamma[i  , j-1] if j > 0             else inf,
          gamma[i-1, j-1] if (i > 0 and j > 0) else inf
        )

  return gamma[-1, -1] ** (1. / q)
  </code>
</pre>

## Properties

Dynamic Time Warping holds the following properties:

* $\forall \mathbf{x}, \mathbf{x}^\prime, DTW_q(\mathbf{x}, \mathbf{x}^\prime) \geq 0$
* $\forall \mathbf{x}, DTW_q(\mathbf{x}, \mathbf{x}) = 0$
* Suppose $\mathbf{x}$ is a time series that is constant except for a motif that
occurs at some point in the series, and let us denote by $\mathbf{x}_{+k}$ a
copy of $\mathbf{x}$ in which the motif is temporally shifted by $k$ timestamps,
then $DTW_q(\mathbf{x}, \mathbf{x}_{+k}) = 0$.

<figure>
    <img src="fig/dtw_shift.gif" alt="Invariance to time shifts" width="80%" />
    <figcaption> 
        Contrary to Euclidean distance, DTW is invariant to time shifts between series.
    </figcaption>
</figure>

However, mathematically speaking, DTW is not a valid metric since it
satisfies neither the triangular inequality nor the identity of indiscernibles.

## Setting Additional Constraints

As we have seen, Dynamic Time Warping is invariant to time shifts, whatever their magnitude.
In order to allow invariances to local deformations only, one can impose additional constraints 
on the set of acceptable paths.
Such constraints typically translate into enforcing nonzero entries in admissible $A_\pi$ to stay 
close to the diagonal.

**TODO: visu path matrices**

The Sakoe-Chiba band [@sakoe1978dynamic] is parametrized by a radius $r$ (also called warping window size sometimes), while
the Itakura parallelogram [@itakura1975minimum] sets a maximum slope $s$ for alignment
paths, which leads to a parallelogram-shaped constraint.
As shown in the Figure below, setting global constraints on admissible DTW paths is equivalent to 
restricting the set of possible matches for each element in a time series.
The number of possible matches for an element is always $2r+1$ for Sakoe-Chiba constraints (except for border elements), 
while it varies depending on the time index for Itakura parallelograms.

<figure>
    <img src="fig/dtw_global_constraints.gif" alt="DTW Global constraints" width="80%" />
    <figcaption>
        Allowed matches for several global constraint schemes.
    </figcaption>
</figure>

As seen in the Figure below, DTW with a Sakoe-Chiba band constraint of radius $r$ is invariant to time shifts of magnitude up to $r$,
but is no longer invariant to longer time shifts.

<figure>
    <img src="fig/sakoe_shift.gif" alt="Invariance to time shifts using Sakoe-Chiba band" width="80%" />
    <figcaption>
        Impact of time shifts on a DTW constrained with a Sakoe-Chiba band of radius $r$.
    </figcaption>
</figure>

# Conclusion

We have seen in this post how alignment-based metrics can prove useful when dealing with temporally shifted time series.
We have presented in more details the most common of these metrics, which is Dynamic Time Warping (DTW).
If you enjoyed this post, stay tuned, a new one should be published soon on the specific topic of the differentiability of DTW.
