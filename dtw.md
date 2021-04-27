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

The definition of adequate metrics between objects to be compared is at the
core of many machine learning methods (_e.g._, nearest neighbors, kernel
machines, _etc._).
When complex objects (such as time series) are involved, such metrics have to
be carefully designed
in order to leverage on desired notions of similarity.

Let us illustrate our point with an example.

<figure>
    <img src="../fig/kmeans.svg" alt="Euclidean k-means" width="100%" />
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
    <img src="../fig/kmeans_dtw.svg" alt="Euclidean k-means" width="100%" />
    <figcaption>
        $k$-means clustering with Dynamic Time Warping. Each subfigure
        represents series from a given cluster and their centroid (in red).
    </figcaption>
</figure>

This part of the course tackles the definition of adequate similarity
measures for time series and their use at the core of machine learning methods.

# Dynamic Time Warping

This section covers works related to Dynamic Time Warping for time series.

Dynamic Time Warping (DTW) [@sakoe1978dynamic] is a similarity measure
between time series.
Consider two time series $\mathbf{x}$ and
$\mathbf{x}^\prime$ of respective lengths $n$ and
$m$.
Here, all elements $x_i$ and $x^\prime_j$ are assumed to lie in the same
$p$-dimensional space and the exact timestamps at which observations occur are
disregarded: only their ordering matters.

## Optimization Problem

In the following, a path $\pi$ of length $K$ is a
sequence of $K$ index pairs
$\left((i_0, j_0), \dots , (i_{K-1}, j_{K-1})\right)$.

DTW between $\mathbf{x}$ and $\mathbf{x}^\prime$ is formulated as the following
optimization problem:

\begin{equation}
DTW_q(\mathbf{x}, \mathbf{x}^\prime) =
    \min_{\pi \in \mathcal{A}(\mathbf{x}, \mathbf{x}^\prime)}
        \left( \sum_{(i, j) \in \pi} d(x_i, x^\prime_j)^q \right)^{\frac{1}{q}}
\label{eq:dtw}
\end{equation}

where $\mathcal{A}(\mathbf{x}, \mathbf{x}^\prime)$ is the set of all admissible
paths, _i.e._ the set of paths $\pi$ such that:

* $\pi$ is a sequence $[\pi_0, \dots , \pi_{K-1}]$ of index pairs
  $\pi_k = (i_k, j_k)$ with $0 \leq i_k < n$ and $0 \leq j_k < m$
* $\pi_0 = (0, 0)$ and $\pi_{K-1} = (n - 1, m - 1)$
* for all $k > 0$ , $\pi_k = (i_k, j_k)$ is related to
  $\pi_{k-1} = (i_{k-1}, j_{k-1})$ as follows:

  * $i_{k-1} \leq i_k \leq i_{k-1} + 1$
  * $j_{k-1} \leq j_k \leq j_{k-1} + 1$

In what follows, we will denote $DTW_2$ as DTW.
In this context, a path can be seen as a temporal alignment of time series and the optimal
path is such that
Euclidean distance between aligned (_ie._ resampled) time series is minimal.

The following image exhibits the DTW path (in white) for a given pair of time
series, on top of the cross-similarity matrix that stores $d(x_i, {x}^\prime_j)$
values:

## Algorithmic Solution

There exists an $O(mn)$ algorithm to compute the exact optimum for this
problem (assuming computation of $d(\cdot,\cdot)$ is $O(1)$):

<pre>
  <code class="language-python">
def dtw(x, x_prime, q=2):
  for i in 1..n:
    for j in 1..m:
      dist = d(x[i], x_prime[j]) ** q
      if i == 1 and j == 1:
        gamma[i, j] = dist
      else:
        gamma[i, j] = dist + min(gamma[i-1, j] if i > 1
                                               else inf,
                                 gamma[i, j-1] if j > 1
                                               else inf,
                                 gamma[i-1, j-1] if (i > 1 and j > 1)
                                                 else inf)

  return (gamma[n, m]) ** (1. / q)
  </code>
</pre>

The basic idea behind this algorithm is that there exists a recurrence relationship between partial DTW computations.
More precisely, if we denote by $\gamma_{i,j}$ the $DTW_q$ (at power $q$) similarity between sequences $\mathbf{x}_{\rightarrow i}$ and $\mathbf{x}^\prime_{\rightarrow j}$ (where the notation $\mathbf{x}_{\rightarrow i}$ denotes sequence $\mathbf{x}$ observed up to time $i$), then we have:

$$
\begin{aligned}
\gamma_{i,j} &=& DTW_q(\mathbf{x}_{\rightarrow i}, \mathbf{x}^\prime_{\rightarrow j})^q \\
&=&
    \min_{\pi \in \mathcal{A}(\mathbf{x}_{\rightarrow i}, \mathbf{x}^\prime_{\rightarrow j})}
        \sum_{(k, l) \in \pi} d(x_k, x^\prime_l)^q \\
&\stackrel{*}{=}& d(x_i, x^\prime_j)^q +
    \min_{\pi \in \mathcal{A}(\mathbf{x}_{\rightarrow i}, \mathbf{x}^\prime_{\rightarrow j})}
        \sum_{(k, l) \in \pi[:-1]} d(x_k, x^\prime_l)^q \\
&\stackrel{**}{=}& d(x_i, x^\prime_j)^q +
    \min (\gamma_{i-1, j}, \gamma_{i, j-1}, \gamma_{i-1, j-1})
\end{aligned}
$$

and $DTW_q(\mathbf{x}, \mathbf{x}^\prime)$ is then $(\gamma_{n, m})^{\frac{1}{q}}$.
In more details:

* $(*)$ comes from the constraints on admissible paths $\pi$: the last element on an admissible path needs to match the last elements of the series;
* $(**)$ comes from the contiguity conditions on the admissible paths: all admissible paths that match $x_i$ with $x^\prime_j$ need to go through one of these 3 possible ancestors: $(i-1, j)$, $(i, j-1)$ or $(i-1, j-1)$.

The dynamic programming algorithm presented above relies on this recurrence formula and stores intermediate computations for efficiency.

<label for="sn-anything" class="sidenote-toggle">⊕</label>
<input type="checkbox" id="sn-anything" class="sidenote-toggle" />
<span class="sidenote">
    **Dot product notation**

    Dynamic Time Warping can also be formalized using the following
notation:

\begin{equation*}
DTW_q(\mathbf{x}, \mathbf{x}^\prime) =
    \min_{\pi \in \mathcal{A}(\mathbf{x}, \mathbf{x}^\prime)}
        \langle A_\pi,  D_q(\mathbf{x}, \mathbf{x}^\prime) \rangle^{\frac{1}{q}}
\end{equation*}

where $D_q(\mathbf{x}, \mathbf{x}^\prime)$ stores distances
$d(x_i, x^\prime_j)$ at the power $q$ and
\begin{equation}
(A_\pi)_{i,j} = \left\{ \begin{array}{rl} 1 & \text{ if } (i, j) \in \pi \\
                                      0 & \text{ otherwise}
                        \end{array} \right. .
\end{equation}
</span>

## Properties

Dynamic Time Warping holds the following properties:

* $\forall \mathbf{x}, \mathbf{x}^\prime, DTW_q(\mathbf{x}, \mathbf{x}^\prime) \geq 0$
* $\forall \mathbf{x}, DTW_q(\mathbf{x}, \mathbf{x}) = 0$
* Suppose $\mathbf{x}$ is a time series that is constant except for a motif that
occurs at some point in the series, and let us denote by $\mathbf{x}_{+k}$ a
copy of $\mathbf{x}$ in which the motif is temporally shifted by $k$ timestamps,
then $DTW_q(\mathbf{x}, \mathbf{x}_{+k}) = 0$.

<figure>
    <img src="../fig/dtw_shift.gif" alt="Invariance to time shifts" width="80%" />
    <figcaption>
        Contrary to Euclidean distance, DTW is invariant to time shifts between series.
    </figcaption>
</figure>

However, mathematically speaking, DTW is not a valid metric since it
satisfies neither the triangular inequality nor the identity of indiscernibles.

## Setting Additional Constraints

The set of temporal deformations to which DTW is invariant can be reduced by
imposing additional constraints on the set of acceptable paths.
Such constraints typically consist in forcing paths to stay close to the
diagonal.

The Sakoe-Chiba band is parametrized by a radius $r$ (number of
off-diagonal elements to consider, also called warping window size sometimes),
as illustrated below:

The Itakura parallelogram sets a maximum slope $s$ for alignment
paths, which leads to a parallelogram-shaped constraint:
