---
title: Differentiability of DTW and the case of softDTW
language: en
author: Romain Tavenard
date: 2021/04/29
rights: Creative Commons CC BY-NC-SA
bibliography: dtw.bib
reference-section-title: References
link-citations: true
---

# Preamble: Differentiability of DTW
* envelope theorem for semi-discrete problems
* visualization of a (1d?2d?) loss landscape [done]

We have the following theorem from [@bonnans1998optimization]:

<div class="theorem">
Let $X$ be a metric space, $U$ be a normed space, and
$\Pi$ be a compact subset of $X$.
Let us define the optimal value function $v$ as: 

$$
  v(u) = \inf_{\pi \in \Pi} f(\pi ; u) \, .
$$

Suppose that:

1. for all $\pi \in X$,  the function $f(\pi ;\cdot )$ is differentiable;
2. $f(\pi ; u)$ and $D_u f(\pi ; u)$ the derivative of $f(\pi ;\cdot)$ are continuous on $X \times U$.

Then if, for $u^0 \in U$, $f( \cdot; u^0)$ has a unique minimizer $\pi^0$ over $\Pi$ then $v(u)$ is differentiable at $u^0$ and $Dv(u^0) = D_u f(x^0 ; u^0)$.
</div>

Let us come back to Dynamic Time Warping, and suppose we are given a reference time series $x_\text{ref}$.
We would like to study the differentiability of

$$
\begin{aligned}
  v(x) &= DTW(x, x_\text{ref}) \\
       &= \min_{\pi \in \mathcal{A}(x, x_\text{ref})} \left\langle A_\pi , D_2(x, x_\text{ref}) \right\rangle^{\frac{1}{2}}
\end{aligned}
$$

then the previous Theorem tells us that $v$ is differentiable everywhere except when:

* $DTW(x, x_\text{ref}) = 0$ since, in this case, the non-differentiability of the square root function breaks condition 1. of the Theorem above;
* there exist several optimal paths for the DTW problem.

This second condition is illustrated in the Figure below in which we vary the value a single element in one of the time series (for visualization purposes)
and study the evolution of $DTW(x, x_\text{ref})$ as a function of this value:

<figure>
    <img src="fig/dtw_landscape.gif" alt="DTW landscape" width="80%" />
    <figcaption> 
        (Non-)differentiability of Dynamic Time Warping.
    </figcaption>
</figure>

Note the sudden change in slope at the position marked by a vertical dashed line, which corresponds to a case where two distinct optimal alignment paths coexist.

# softDTW and variants
* soft alignment path
* properties 
    * $\gamma t^2$ impact of a $t$-offset (visu) [done]
    * interpretation as the expectation across all alignments
* denoising effect
* Visualization of a loss landscape for softDTW as a function of $\gamma$ [done]
* sharp softDTW, divergences...