---
title: Differentiability of DTW and the case of soft-DTW
language: en
author: Romain Tavenard
date: 2021/06/22
year: 2021
draft: false
rights: Creative Commons CC BY-NC-SA
shortname: softdtw
bibliography: dtw.bib
---

# Introduction

We have seen in a [previous blog post](dtw.html) how one can use Dynamic Time Warping (DTW) as a shift-invariant similarity measure between time series.
In this new post, we will study some aspects related to the differentiability of DTW.
One of the reasons why we focus on differentiability is that this property is key in modern machine learning approaches.

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
  v(x) &= DTW_2(x, x_\text{ref}) \\
       &= \min_{\pi \in \mathcal{A}(x, x_\text{ref})} \left\langle A_\pi , D_2(x, x_\text{ref}) \right\rangle^{\frac{1}{2}}
\end{aligned}
$$

then the previous Theorem tells us that $v$ is differentiable everywhere except when:

* $DTW_2(x, x_\text{ref}) = 0$ since, in this case, the non-differentiability of the square root function breaks condition 1 of the Theorem above;
* there exist several optimal paths for the DTW problem.

This second condition is illustrated in the Figure below in which we vary the value of a single element in one of the time series (for visualization purposes)
and study the evolution of $DTW_2(x, x_\text{ref})$ as a function of this value:

<figure>
    <video playsinline muted autoplay controls loop width="80%">
        <source src="fig/dtw_landscape.webm" type="video/webm" />
        <source src="fig/dtw_landscape.mp4" type="video/mp4" />
        <img src="fig/dtw_landscape.gif" alt="DTW landscape" />
    </video>
    <figcaption> 
        (Non-)differentiability of Dynamic Time Warping.
        A single element $x_\tau$ is changed in $x$ (top) and $DTW(x, x^\prime)$ is reported as a function of $x_\tau$ (bottom).
    </figcaption>
</figure>

Note the sudden change in slope at the position marked by a vertical dashed line, which corresponds to a case where (at least) two distinct optimal alignment paths coexist.

# Soft-DTW and variants

Soft-DTW [@cuturi2017soft] has been introduced as a way to mitigate this
limitation.
The formal definition for soft-DTW is the following:

\begin{equation}
\text{soft-}DTW^{\gamma}(x, x^\prime) =
    \min_{\pi \in \mathcal{A}(x, x^\prime)}{}^\gamma
        \sum_{(i, j) \in \pi} d(x_i, x^\prime_j)^2
\label{eq:softdtw}
\end{equation}

where $\min{}^\gamma$ is the soft-min operator parametrized by a smoothing
factor $\gamma$.

## A note on soft-min

The soft-min operator $\min{}^\gamma$ is defined as:

\begin{equation}
    \min{}^\gamma(a_1, \dots, a_n) = - \gamma \log \sum_i e^{-a_i / \gamma}
\end{equation}

Note that when gamma tends to $0^+$, the term corresponding to the lower $a_i$
value will dominate other terms in the sum, and the soft-min then tends to the
hard minimum, as illustrated below:

<figure>
    <img src="fig/soft_min.svg" alt="soft-min function" width="80%" />
    <figcaption> 
        The soft-min function $\min^\gamma$ applied to the pair $(-a, a)$ for various values of $\gamma$.
        The solid gray line corresponds to the hard min function.<label for="sn-softmin" class="sidenote-toggle sidenote-number"></label>
        <input type="checkbox" id="sn-softmin" class="sidenote-toggle" />
        <span class="sidenote">This Figure is inspired from [the dedicated wikipedia page](https://en.wikipedia.org/wiki/Smooth_maximum).</span>
    </figcaption>
</figure>

As a consequence, we have:

$$
    \text{soft-}DTW^{\gamma}(x, x^\prime)
    \xrightarrow{\gamma \to 0^+} DTW_2(x, x^\prime)^2 \, .
$$

However, contrary to DTW, soft-DTW is differentiable everywhere for strictly positive $\gamma$ even if, for small $\gamma$ values, sudden changes can still occur in the loss landscape, as seen in the Figure below:

<figure>
    <video playsinline muted autoplay controls loop width="80%">
        <source src="fig/softdtw_landscape.webm" type="video/webm" />
        <source src="fig/softdtw_landscape.mp4" type="video/mp4" />
        <img src="fig/softdtw_landscape.gif" alt="softDTW landscape" />
    </video>
    <figcaption> 
        Differentiability of soft-DTW.
        For the sake of visualization, soft-DTW divergence, which is a normalized version of soft-DTW [discussed below](#related-similarity-measures), is reported in place of soft-DTW.
    </figcaption>
</figure>

Note that the recurrence relation we had in Equation [(2)](dtw.html#eq:rec) of the post on DTW still holds with this $\min^\gamma$ formulation:

$$
    R_{i,j} = d(x_i, x^\prime_j)^2 +
        \min{}^\gamma ({\color{MidnightBlue}R_{i-1, j}}, {\color{Red}R_{i, j-1}}, {\color{ForestGreen}R_{i-1, j-1}}) \, ,
$$

where $\text{soft-}DTW^{\gamma}(x, x^\prime) = R_{n-1, m-1}$.
As a consequence, the $O(mn)$ DTW algorithm is still valid here.

## Soft-Alignment Path

It is shown in [@mensch2018] that soft-DTW can be re-written:

<div class="scroll-wrapper" id="eq:entropy_softdtw">
$$
    \text{soft-}DTW^{\gamma}(x, x^\prime) =
        \min_{p \in \Sigma^{|\mathcal{A}(x, x^\prime)|}} \left\langle \sum_{\pi \in \mathcal{A}(x, x^\prime)} p(\pi) A_\pi , D_2(x, x^\prime) \right\rangle - \gamma H(p)
$$
</div>

where $\Sigma^{|\mathcal{A}(x, x^\prime)|}$ is the set of probability distributions over paths and $H(p)$ is the entropy
of a given probability distribution $p$.

<div>
<details>
<summary>A (very short) note on entropy</summary> 

For a discrete probability distribution $p$, entropy (also known as Shannon entropy) is defined as 
$$
    H(p) = -\sum_i p_i \log (p_i)
$$ 

and is maximized by the uniform distribution, as seen below:

<figure>
    <img src="fig/entropy.svg" alt="Entropy" width="70%" />
    <figcaption> 
        Entropy of probability distributions over $\Sigma^3$.
        Each barycentric position $(p_0, p_1, p_2)$ in the triangle above maps to a probability distribution $p = (p_0, p_1, p_2)$ whose entropy is color-coded.
        Note how higher entropy values are associated to distributions lying close to the center of the triangle (that are hence closer to the uniform distribution).
    </figcaption>
</figure>
</details>
</div>

For strictly positive $\gamma$, the [entropy-regularized problem](#eq:entropy_softdtw) above has a closed-form solution that is:

$$
    p^\star_\gamma(\pi) = \frac{e^{-\langle A_\pi, D_2(x, x^\prime)\rangle / \gamma}}{k_{\mathrm{GA}}^{\gamma}(x, x^\prime)}
$$

where $k_{\mathrm{GA}}^{\gamma}(x, x^\prime)$ is the Global Alignment kernel [@cuturi2007kernel] 
that acts as a normalization factor here.

This formulation leads to the following definition for the soft-alignment matrix $A_\gamma$

$$
    A_\gamma = \sum_{\pi \in \mathcal{A}(x, x^\prime)} p^\star_\gamma(\pi) A_\pi \, .
$$ {#eq:a_gamma}

$A_\gamma$ is a matrix that informs, for each pair
$(i, j)$, how much it will be taken into account in the matching.

<figure>
    <video playsinline muted autoplay controls loop width="60%">
        <source src="fig/a_gamma.webm" type="video/webm" />
        <source src="fig/a_gamma.mp4" type="video/mp4" />
        <img src="fig/a_gamma.gif" alt="$A_\gamma$ matrix" />
    </video>
    <figcaption> 
        $A_\gamma$ matrix. Note how the matrix blurs out when $\gamma$ grows.
    </figcaption>
</figure>
Note that when $\gamma$ tends toward $+\infty$, $p^\star_\gamma$ weights tend to the uniform distribution, hence the averaging operates over all alignments with equal weights, and the corresponding $A_\gamma$ matrix tends to favor diagonal matches, regardless
of the content of the series $x$ and $x^\prime$.

<!-- <figure>
    <img src="fig/a_inf.svg" alt="$A_\infty$ matrix" width="60%" />
    <figcaption> 
        $A_\infty$ matrix for time series of length 30.
    </figcaption>
</figure> -->

However, the sum in {@eq:a_gamma} is intractable due to the very large number of paths in $\mathcal{A}(x, x^\prime)$.
Fortunately, once soft-DTW has been computed, $A_\gamma$ can be obtained through a backward dynamic programming pass 
with complexity $O(mn)$ (see more details in [@cuturi2017soft]).

Computing this $A_\gamma$ matrix is especially useful since it is directly related to the gradients of the soft-DTW similarity measure:

\begin{equation}
\nabla_{x} \text{soft-}DTW^{\gamma}(x, x^\prime) =
    \left(\frac{\partial D_2(x, x^\prime)}{\partial x} \right)^T A_\gamma \, .
\end{equation}

## Properties

As discussed in [@janati2020spatio], soft-DTW is not invariant to time
shifts, as DTW is.
Suppose $x$ is a time series that is constant except for a motif that
occurs at some point in the series, and let us denote by $x_{+k}$ a
copy of $x$ in which the motif is temporally shifted by $k$ timestamps.
Then the quantity

<div class="scroll-wrapper">
\begin{equation*}
\Delta^\gamma(x, x_{+k}) = \left| \text{soft-}DTW^{\gamma}(x, x_{+k}) - \text{soft-}DTW^{\gamma}(x, x) \right|
\end{equation*}
</div>

grows linearly with $\gamma k^2$:

<figure>
    <video playsinline muted autoplay controls loop width="80%">
        <source src="fig/softdtw_shift.webm" type="video/webm" />
        <source src="fig/softdtw_shift.mp4" type="video/mp4" />
        <img src="fig/softdtw_shift.gif" alt="Impact of time shifts on soft-DTW" />
    </video>
    <figcaption> 
        Impact of time shifts on soft-DTW.
    </figcaption>
</figure>

The reason behind this sensibility to time shifts is that soft-DTW provides a
weighted average similarity score across all alignment paths (where stronger
weights are assigned to better paths), instead of focusing on the single best
alignment as done in DTW.

Another important property of soft-DTW is its "denoising effect", in
the sense that, for a given time series $x_\text{ref}$, the minimizer of
$\text{soft-}DTW^{\gamma}(x, x_\text{ref})$ is not $x_\text{ref}$
itself but rather a smoothed version of it:

<figure>
    <video playsinline muted autoplay controls loop width="80%">
        <source src="fig/softdtw_denoising.webm" type="video/webm" />
        <source src="fig/softdtw_denoising.mp4" type="video/mp4" />
        <img src="fig/softdtw_denoising.gif" alt="Denoising effect of soft-DTW" />
    </video>
    <figcaption> 
        Denoising effect of soft-DTW.
        Here, we perform a gradient descent on $x$, initialized at $x^{(0)} = x_\text{ref}$.
        Note how using a larger $\gamma$ value tends to smooth out more details.<label for="sn-denoising" class="sidenote-toggle sidenote-number"></label>
        <input type="checkbox" id="sn-denoising" class="sidenote-toggle" />
        <span class="sidenote">This Figure is widely inspired from Figure 2 in [@blondelmensch2020].</span>
    </figcaption>
</figure>

Finally, as seen in Figure 2, $\min^\gamma$ lower bounds the min operator.
As a result, soft-DTW lower bounds DTW.
Another way to see it is by taking a closer look at the [entropy-regularized formulation](#eq:entropy_softdtw) for soft-DTW and observing that a distribution that would have a probability of 1 for the best path and 0 for all other paths is an element of $\Sigma^{|\mathcal{A}(x, x^\prime|}$ whose cost is equal to $DTW(x, x^\prime)$.
Since soft-DTW is a minimum over all probability distributions in $\Sigma^{|\mathcal{A}(x, x^\prime|}$, it hence has to be lower or equal to $DTW(x, x^\prime)$.
Contrary to DTW, soft-DTW is not bounded below by zero, and we even have:

$$
    \text{soft-}DTW^\gamma (x, x^\prime) \xrightarrow{\gamma \to +\infty} -\infty \, .
$$

## Related Similarity Measures

In [@blondelmensch2020], new similarity measures are defined, that rely on
soft-DTW.
In particular, *soft-DTW divergence* is introduced to counteract the non-positivity of soft-DTW:

<div class="scroll-wrapper">
\begin{equation}
    D^\gamma (x, x^\prime) =
        \text{soft-}DTW^{\gamma}(x, x^\prime)
        - \frac{1}{2} \left(
                \text{soft-}DTW^{\gamma}(x, x) +
                \text{soft-}DTW^{\gamma}(x^\prime, x^\prime)
            \right) \, .
\end{equation}
</div>

This divergence has the advantage of being minimized for
$x = x^\prime$ and being exactly 0 in that case.

<!-- Second, another interesting similarity measure introduced in the same paper is
the **sharp soft-DTW** which is:

\begin{equation}
    \text{sharp-soft-}DTW^{\gamma} (x, x^\prime) =
        \langle A_\gamma,  D_2(x, x^\prime) \rangle
\end{equation}

Note that a **sharp soft-DTW divergence** can be derived from this
(with a similar approach as for $D^\gamma$), which has the extra benefit
(over the sharp soft-DTW) of
being minimized at $x = x^\prime$.

-->

Also, in [@hadji2020], a variant of $\min^\gamma$, called $\text{smoothMin}^\gamma$ is used in the recurrence formula.
Contrary to $\min^\gamma$, $\text{smoothMin}^\gamma$ upper bounds the min operator:

<figure>
    <img src="fig/smooth_min.svg" alt="smooth-min function" width="80%" />
    <figcaption> 
        The $\text{smoothMin}^\gamma$ function applied to the pair $(-a, a)$ for various values of $\gamma$.
        The solid gray line corresponds to the hard min function.
    </figcaption>
</figure>

As a consequence, the resulting similarity measure upper bounds DTW.
Note also that [@hadji2020] suggest that the DTW variants presented in these posts are not fully suited for representation learning and additional contrastive losses should be used to help learn useful representations. 

# Conclusion

We have seen in this post that DTW is not differentiable everywhere, and that there exists alternatives that basically change the min operator into a differentiable alternative in order to get a differentiable similarity measure that can later be used as a loss in gradient-based optimization.

<!-- The next post in this series will be dedicated to drawing links between optimal transport and dynamic time warping. -->
