---
layout: post
---
[back](./)

# Paper log

### Influence functions

[Influence Functions in Deep Learning Are Fragile ](https://arxiv.org/pdf/2006.14651.pdf)



#### [Understanding Black-box Predictions via Influence Functions](https://arxiv.org/pdf/1703.04730.pdf)

Pang Wei Koh, Percy Liang

**Overview**

- Introduces method for approximating the _influence_ of removing an individual point during training:
- Define:
  - $$\hat{\theta} = \arg\min_\theta \frac{1}{n}\sum_{i=1}^n L(z_i, \theta)$$
  - $$\hat\theta_{-z} = \arg\min_\theta \frac{1}{n}\sum_{z_i\neq z} L(z_i, \theta)$$

- The change in the parameters from removing $z$ is clearly just $$\hat{\theta}_{-z}-\hat{\theta}$$. The authors approximate this change by computing the 'influence' of $z$ on the parameters:
  - $$\hat{\theta}_{\epsilon,z} = \arg\min_\theta \frac{1}{n}\sum_{i=1}^n L(z_i,\theta) + \epsilon L(z,\theta)$$ (error upweighting $$z$$ by $$\epsilon$$)
  - $$\mathcal{I}_{up,params}(z) = \frac{d\hat{\theta}_{\epsilon,z}}{d\epsilon}\Big|_{\epsilon=0} = -H_\theta^{-1}\nabla L(z,\hat{\theta})$$ (take a newton step starting at $\hat{\theta}$ and measure the change)
- Removing the point $$z$$ is equivalent to setting $$\epsilon = -\frac{1}{n}$$, and so we can approximate $$\hat{\theta}_{-z}-\hat{\theta} \approx -\frac{1}{n}\mathcal{I}_{up,params}(z)$$.
- Using the same idea + the chain rule, we can also calculate how much removing $z$ changes functions of $$\hat{\theta}$$. In particular, we consider the function $$L(z_{test}, \hat{\theta})$ for some testing point $z_{test}$$:
  - $$\mathcal{I}_{up,loss}(z,z_{test}) = -\nabla L(z_{test},\hat{\theta})^\top H_\theta^{-1}\nabla L(z,\hat{\theta})$$
- This gives us a metric for finding the point "nearest" to a given testing point $$z_{test}$$, relative to a particular task, i.e. we can just look at $$z_{nearest} = \arg\min_{z\in S_{train}} \mathcal{I}_{up,loss}(z,z_{test})$$.
  - Can give explicit form for this in the case of logistic regression

**Methodology + experiments**



### Classification

#### [Classification vs regression in overparameterized regimes: Does the loss function matter?](https://arxiv.org/pdf/2005.08054.pdf)

#### [On the proliferation of support vectors in high dimensions ](https://arxiv.org/pdf/2009.10670.pdf)

#### [The generalization error of max-margin linear classifiers: High-dimensional asymptotics in the overparametrized regime ](https://arxiv.org/pdf/1911.01544.pdf)

#### [The Phase Transition for the Existence of the Maximum Likelihood Estimate in High-dimensional Logistic Regression](https://arxiv.org/pdf/1804.09753.pdf)

#### [**LIVING ON THE EDGE: PHASE TRANSITIONS IN CONVEX PROGRAMS WITH RANDOM DATA**](https://arxiv.org/pdf/1303.6672.pdf)

[A Study in Rashomon Curves and Volumes:
 A New Perspective on Generalization and Model Simplicity in Machine Learning ](https://arxiv.org/pdf/1908.01755.pdf)

[Stochastic Gradient Descent as Approximate Bayesian Inference ](https://arxiv.org/pdf/1704.04289.pdf)



### NTK

#### NTK Review

- Let's be naive and Taylor expand a neural network
  $$
  f(x, W(t)) \approx f(x, W(0)) + (W(0)-W(t))^\top\nabla f(x, W(0))
  $$

- For a width $m$ of this network, we can define a kernel
  $$
  K_m(x,x') = \nabla f(x, W(0))^\top \nabla f(x', W(0))
  $$

- It turns out that if the weights are initialized with variance $$O(1/m)$$, then a law of large number argument shows that
  $$
  K_m \xrightarrow{m\rightarrow\infty} K
  $$
  where $$K$$ is a fixed kernel, independent of the initialization.

- To show that inference with $K$ is an accurate depiction of neural networks in the large width limit, we need to show that the first order Taylor expansion is accurate.
  - This can be done by analyzing continuous-time gradient descent, which shows $$W(t)$$ is close to $$W(0)$$. For example the trajectory-based analysis of Du et. al., which looks at $$y - \hat{y}(t)$$, and shows that this evolves like a kernel method (specifically, $$K_m(t) \approx K_m(0) \approx K$$).
  - Liu et. al. shows that this isn't actually necessary: the first order Taylor expansion is accurate regardless because (for many architectures) we have contral on the Hessian norm $$\|H\| = \|\nabla^2 f(x, W(0))\| = O(\frac{1}{\sqrt{m}})$$.
- Simon Du talk: https://www.youtube.com/watch?v=HvEGJUwQEO8
- Lecture from UMD: https://www.youtube.com/watch?v=DObobAnELkU

#### [On Exact Computation with an Infinitely Wide Neural Net](https://arxiv.org/pdf/1904.11955.pdf)

#### [Beyond Linearization: On Quadratic and Higher-Order Approximation of Wide Neural Networks](https://arxiv.org/pdf/1910.01619.pdf)

#### [Learning Overparameterized Neural Networks via Stochastic Gradient Descent on Structured Data](https://papers.nips.cc/paper/8038-learning-overparameterized-neural-networks-via-stochastic-gradient-descent-on-structured-data.pdf)

#### [What Can ResNet Learn Efficiently, Going Beyond Kernels?](https://arxiv.org/pdf/1905.10337.pdf)

#### [On the linearity of large non-linear models: when and why the tangent kernel is constant](https://arxiv.org/pdf/2010.01092.pdf)

#### [Towards Explaining the Regularization Effect of Initial Large Learning Rate in Training Neural Networks ](https://arxiv.org/pdf/1907.04595.pdf)

Chaoyue Liu, Libin Zhu, Mikhail Belkin

**Overview**

- Typically, the linearity of large neural networks is shown by demonstrating that $W(t)$ is close to $W(0)$ (called the 'lazy training' regime)

  - This necessarily involves a loss function + optimization procedure (usually gradient flow)

- This paper shows that linearity holds for wide neural networks regardless of the training procedure, by showing that the Hessian norm $$\|H\| = O(\frac{1}{\sqrt{m}})$$, and hence that the Taylor expansion
  $$
  f(x, W(t)) \approx f(x, W(0)) + (W(0)-W(t))^\top\nabla f(x, W(0))
  $$
  is always accurate in the wide-width limit.
