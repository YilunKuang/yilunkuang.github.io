---
layout: distill
title: Rectified LpJEPA
description: Joint-Embedding Predictive Architectures with Sparse and Maximum-Entropy Representations
tags: distill formatting
giscus_comments: true
date: 2026-01-30
thumbnail: assets/img/final_teasor.png

authors:
  - name: Yilun Kuang
    url: "https://yilunkuang.github.io/"
    affiliations:
      name: New York University

bibliography: 2026-01-30-rectified-lp-jepa.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Self-Supervised Learning
  - name: Isotropic Gaussian Regularization
  - name: Product Laplace Regularization
  - name: Rectified Generalized Gaussian
  - name: Rectified LpJEPA

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }

---

## Outline


gaussian. why gaussian, because it's maximum-entropy, and btw the angular distribution coincide with uniform distributions on the sphere, which is the ideal distribution. 


Why Gaussian features are dense by construction

Laplace distributions and the geometry of sparsity

From special cases to a continuous spectrum

Why rectification changes everything
- intuition about rectifications

Designing representation geometry via distribution matching

## Self-Supervised Learning

Consider an input vector $\mathbf{x}$, such as an image, an audio clip, or a video frame. We would like to learn a neural network representation

$$
\begin{align}
\mathbf{z}=f_{\boldsymbol{\theta}}(\mathbf{x})\in\mathbb{R}^d
\end{align}
$$

in the absence of any labeling information. Self-supervised learning makes this possible by creating supervision from the data itself. Given $\mathbf{x}$, we can generate another view $\mathbf{x}\'$ of the input $\mathbf{x}$ that preserves the same semantic content.
- For images, $\mathbf{x}\'$ might be a cropped, rotated, or even corrupted version of $\mathbf{x}$
- For audio or video, $\mathbf{x}\'$ can be a nearby time segment or adjacent frame.

Although $\mathbf{x}$ and $\mathbf{x}\'$ may look different, they are assumed to be semantically related. Thus we can learn a neural network representation by relying on the following principle: **representations of different views of the same input should be similar**. Translating to mathematical language, this means we can minimize the $\ell_2$ distance between the embeddings of different views

$$
\begin{align}
\min_{\boldsymbol{\theta}}\|\mathbf{z}-\mathbf{z}'\|_2
\tag{2}
\end{align}
$$

where $$\mathbf{z}' = f_{\boldsymbol{\theta}}(\mathbf{x}')$$. By enforcing agreement across many randomly generated views, the network learns features that are invariant to nuisance transformations and capture meaningful structure in the data.

## Isotropic Gaussian Regularization

Simply minimizing the $$\ell_2$$ distance (Eq. (2)) between views, however, can lead to the problem of **feature collapse**. In the extreme case, the network can map every input to the same vector, resulting in **complete collapse**. Eq. (2) is perfectly minimized, but the representation is useless—it cannot distinguish between different inputs. More subtle forms of collapse also occur, such as **dimensional collapse**, where different feature dimensions encode redundant information <d-cite key="jing2022understandingdimensionalcollapsecontrastive"></d-cite>.

The goal of self-supervised learning is therefore to enforce invariance across views while maximally spreading feature vectors in the ambient space. One effective way to do this is to align the feature distributions towards an **isotropic Gaussian** $$\mathcal{N}(\mathbf{0},\mathbf{I}_{d})$$, in addition to minimizing Eq. (2) <d-cite key="kuang2025radialvcreg"></d-cite> <d-cite key="balestriero2025lejepaprovablescalableselfsupervised"></d-cite>. 

<!-- The probability density function of an isotropic Gaussian random vector $$\mathbf{z}\sim\mathcal{N}(\mathbf{0}, \mathbf{I}_{d})$$ is  -->


> ##### Probability Density Functions of Isotropic Gaussian
>
> $$p(\mathbf{z})=\frac{1}{(2\pi)^{d/2}}\exp\bigg(-\frac{1}{2}\|\mathbf{z}\|_2^2\bigg)$$
{: .block-tip }

The choice of Gaussian as the target distribution is not an arbitrary one. Geometrically, a $d$-dimensional isotropic Gaussian random vector $\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I}_{d})$ concentrates on a thin shell of radius $\sqrt{d}$ with an $O(1)$ width. Under polar decompositions, the radius $\|\|\mathbf{z}\|\|_2\sim\chi(d)$ follows the Chi distribution while the angular direction $\mathbf{z}/\|\|\mathbf{z}\|\|_2$ is uniformly distributed over the unit $\ell_2$ sphere with respect to the surface measure. This uniform angular distribution coincides exactly with the uniformity condition identified as optimal for contrastive learning <d-cite key="wang2022understandingcontrastiverepresentationlearning"></d-cite>. As a result, isotropic Gaussian features are maximally spread out in direction, providing a principled way to prevent feature collapse.

<div class="row justify-content-center" style="margin-top: 0.5rem;">
  <div class="col-sm-6">
    {% include figure.html path="assets/img/rectified_lp_jepa/L2_sphere_full.png"
       class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption" style="margin-top: 0.15rem; margin-bottom: 0.5rem;">
  Unit $\ell_2$ Sphere
</div>

This geometric behavior has a direct information-theoretic correspondence. Among all possible probability distributions with a fixed expected $\ell_2$-norm (i.e., fixed average energy), the isotropic Gaussian maximizes entropy. In other words, if we constrain only how much energy the features carry and impose no further structure, the Gaussian is the most "spread out" distribution possible.

## Product Laplace Regularization



## Generalized Gaussian Distributions

<!-- Beyond its geometric appeal, the isotropic Gaussian also has a fundamental information-theoretic property: among all probability distributions with a fixed expected $\ell_2$-norm (i.e., fixed average energy), it maximizes entropy. In other words, if we only control the overall energy of the features and impose no other structure, the Gaussian is the most "spread out" and least structured choice possible. -->


<!-- A neural network encoder $f_{\boldsymbol{\theta}}$ maps this input to a representation -->
<!-- \begin{align}
\mathbf{z}=f_{\boldsymbol{\theta}}(\mathbf{x})\in\mathbb{R}^d
\end{align} -->

<!-- and we denote its neural network embedding as $\mathbf{z}=f_{\theta}(\mathbf{x})\in\mathbb{R}^d$.  -->




## Rectified Generalized Gaussian

<!-- <div class="l-page">
  <div style="
    transform: scale(0.6);
    transform-origin: top center;
    width: 100%;
  ">
    <iframe
      src="{{ '/assets/plotly/vary_mu_p2_sigma1.html' | relative_url }}"
      frameborder="0"
      scrolling="no"
      height="800"
      width="100%"
      style="border: 1px dashed grey;"
    ></iframe>
  </div>
</div> -->

<div class="l-page">
  <div style="
    width: 100%;
    overflow: hidden;
    height: 480px;              /* 800 * 0.6 = 480 */
  ">
    <iframe
      src="{{ '/assets/plotly/vary_mu_p2_sigma1.html' | relative_url }}"
      frameborder="0"
      scrolling="no"
      width="100%"
      height="800"
      style="
        border: 1px dashed grey;
        display: block;
        transform: scale(0.6);
        transform-origin: top center;
      "
    ></iframe>
  </div>
</div>



## RDMReg

## Rectified LpJEPA


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/final_teasor.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Rectified LpJEPA
</div>


## additional helper

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.html
       path="assets/img/rectified_lp_jepa/L2_sphere_full.png"
       class="img-fluid rounded z-depth-1"
       zoomable=true %}
    <div class="caption">
    Unit $\ell_2$ Sphere
    </div>
  </div>

  <div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.html
       path="assets/img/rectified_lp_jepa/L1_sphere_full.png"
       class="img-fluid rounded z-depth-1"
       zoomable=true %}
    <div class="caption">
    Unit $\ell_1$ Sphere
    </div>
  </div>
</div>


You just need to surround your math expression with `$$`, like `$$ E = mc^2 $$`.


Note that MathJax 3 is [a major re-write of MathJax](https://docs.mathjax.org/en/latest/upgrading/whats-new-3.0.html)


<d-code block language="javascript">
  var x = 25;
  function(x) {
    return x * x;
  }
</d-code>

{% highlight javascript %}
var x = 25;
function(x) {
  return x * x;
}
{% endhighlight %}


{% details Click here to know more %}
Additional details, where math $$ 2x - 1 $$ and `code` is rendered correctly.
{% enddetails %}

Colons can be used to align columns.

| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |

There must be at least 3 dashes separating each header cell.
The outer pipes (|) are optional, and you don't need to make the
raw Markdown line up prettily. You can also use inline Markdown.

Markdown | Less | Pretty
--- | --- | ---
*Still* | `renders` | **nicely**
1 | 2 | 3

> Blockquotes are very handy in email to emulate reply text.
> This line is part of the same quote.

Quote break.

> This is a very long line that will still be quoted properly when it wraps. Oh boy let's keep writing to make sure this is long enough to actually wrap for everyone. Oh, you can *put* **Markdown** into a blockquote.


<!-- Self-supervised learning achieves this by creating another view $\mathbf{x}\'$ of the input $\mathbf{x}$. For images, this could be a cropped, rotated, or even corrupted version of the original image $\mathbf{x}$ which still preserves some information about what $\mathbf{x}$ is. For audio and video, this can just be temporally adjacent clips or frames which are slowly varying and hence semantically related to each other.  -->





<!-- Once we have both $\mathbf{x}$ and $\mathbf{x}\'$, we can simply learn a neural network such that the distance between these two vectors in the latent space are close, i.e.
\begin{align}
\min_{\boldsymbol{\theta}}\|\|\mathbf{z}-\mathbf{z}\'\|\|_2
\end{align}
This way, for any given $\mathbf{x}$, we can find the corresponding $\mathbf{x}\'$ and the network can rely on the similarity signal for learning. The simple procedure of learning invariances across different views of the input provides us ample supply of learning signal and thus enables scalable learning. -->

