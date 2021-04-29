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

# Introduction

We have seen in a [previous blog post](dtw.html) how one can use Dynamic Time Warping (DTW) as a shift-invariant similarity measure between time series.
In this new post, we will study some aspects related to the differentiability of DTW.
The reason why we focus on differentiability is that this property is key in modern machine learning approaches.

[@cuturi2017soft] provide a nice example setting in which differentiability is desirable:
Suppose we are given a forecasting task<label for="sn-1" class="sidenote-toggle sidenote-number"></label>
<input type="checkbox" id="sn-1" class="sidenote-toggle" />
<span class="sidenote">A forecasting task is a task in which we are given the beginning of a time series and the goal is to predict the future behavior of the series.</span>in which the exact temporal localization of the 
temporal motifs to be predicted are less important than their overall shapes.
In such a setting, it would make sense to use a shift-invariant similarity measure in order to assess whether a prediction made by the model is close enough from the ground-truth.
Hence, a rather reasonable approach could be to tune the parameters of a neural network in order to minimize such a loss.
Since optimization for this family of models heavily relies on gradient descent, having access to a differentiable shift-invariant similarity measure between time series is a key ingredient of this approach.

# Differentiability of DTW

Let us start by having a look at the differentiability of Dynamic Time Warping.
To do so, we will rely on the following theorem from [@bonnans1998optimization]:

<div class="theorem">
Let $\Phi$ be a metric space, $X$ be a normed space, and
$\Pi$ be a compact subset of $\Phi$.
Let us define the optimal value function $v$ as: 

$$
  v(x) = \inf_{\pi \in \Pi} f(x ; \pi) \, .
$$

Suppose that:

1. for all $\pi \in \Phi$,  the function $x \mapsto f( x ; \pi )$ is differentiable;
2. $f(x ; \pi)$ and $D_x f(x ; \pi)$ the derivative of $x \mapsto f( x ; \pi )$ are continuous on $X \times \Phi$.

If, for $x^0 \in X$, $\pi \mapsto f(x^0 ; \pi )$ has a unique minimizer $\pi^0$ over $\Pi$ then $v$ is differentiable at $x^0$ and $Dv(x^0) = D_x f(x^0 ; \pi^0)$.
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

Note the sudden change in slope at the position marked by a vertical dashed line, which corresponds to a case where (at least) two distinct optimal alignment paths coexist.

# softDTW and variants
* soft alignment path
* properties 
    * $\gamma t^2$ impact of a $t$-offset (visu) [done]
    * interpretation as the expectation across all alignments
* denoising effect
* Visualization of a loss landscape for softDTW as a function of $\gamma$ [done]
* sharp softDTW, divergences...