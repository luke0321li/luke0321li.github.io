---
title: Questions regarding Lasso and Ridge Regression 
---
![placeholder](https://www.taftmuseum.org/wp-content/uploads/1976-41-Johnson_Cowboy-with-Lasso-on-Horse-957x1200.jpg)
*Cowboy with his lasso and presumably on a ridge, by Frank Tenney Johnson*

This article addresses some frequently asked questions about the two regularized forms of linear regression, namely Lasso and ridge regresion. 

## How are they different from regular linear regression?
Lasso and ridge regression impose $$L^{1}$$ and $$L^{2}$$ penalties on the weight, as a form of regularization. 

Linear regression has the form $$\hat{y} = \boldsymbol{w}^{T}\boldsymbol{x}$$ where $$\hat{y}, \boldsymbol{w}, \boldsymbol{x}$$ are the predicted label, weight vector and the feature vector respectively (assume the feature vector has a dummy entry of 1 to account for the bias term). The root mean square cost function is $$J(\boldsymbol{w}) = \frac{1}{n}\sum_{i = 1}^{n}(y_{i} - \boldsymbol{w}^{T}\boldsymbol{x}_{i})^{2}$$. 

Now, for Lasso, 

$$J(\boldsymbol{w}) = \frac{1}{n}\sum_{i = 1}^{n}(y_{i} - \boldsymbol{w}^{T}\boldsymbol{x}_{i})^{2} + \lambda\|\boldsymbol{w}\|_{1}$$

(note: $$\|\boldsymbol{v}\|_{p} = (\sum_{i = 1}^{m}\lvert v_{i}\rvert^{p})^{1/p}$$ is the $$L^{p}$$ norm of vector $$\boldsymbol{v} \in \mathbb{R}^{m}$$); <br/>

For ridge regression, 

$$J(\boldsymbol{w}) = \frac{1}{n}\sum_{i = 1}^{n}(y_{i} - \boldsymbol{w}^{T}\boldsymbol{x}_{i})^{2} + \lambda\|\boldsymbol{w}\|_{2}^{2}$$

It can be seen that for both Lasso and ridge regression, the cost function increases with the norm (can be thought of as a measurement of "size") of the weight vector. With larger weights, the weight vector will have greater norm, resulting in higher cost. A positive hyperparameter, $$\lambda$$, controlls the extent to which the norm affects the cost (its actual mechanism of action will be described later). 

## Are the cost functions still convex?

Yes. The original cost function is convex, the $$L^{p}$$ norm is always convex, and the sum of convex functions is still convex.

## What is the gradient of Lasso's cost function?

$$\lvert x \rvert$$ is non-differentiable. However, there exist some generalized gradient descent algorithms for non-differentiable functions. 

## What are the closed-form solutions for Lasso and ridge regression?

Since Lasso has a non-differentiable gradient, it does not have a closed-form solution. On the other hand, ridge regression does. Let $$\boldsymbol{y}$$ be the vector of labels where each element corresponds to the label of a single sample; let $$\mathbf{X}$$  be the matrix such that each row corresponds to a single sample's feature vector. Assume both $$\boldsymbol{y}$$ and $$\boldsymbol{w}$$ are column vectors, then rewrite the cost function:

$$J(\boldsymbol{w}) = (\boldsymbol{y} - \mathbf{X}\boldsymbol{w})^{T}(\boldsymbol{y} - \mathbf{X}\boldsymbol{w}) + \lambda\boldsymbol{w}^{T}\boldsymbol{w}$$

As mentioned this function is convex, so a global minimum is guaranteed to be found by setting its derivative to zero.

$$\frac{\partial J(\boldsymbol{w})}{\partial \boldsymbol{w}} = -2(\boldsymbol{y} - \mathbf{X}\boldsymbol{w})^{T}\mathbf{X} + 2\lambda\boldsymbol{w}^{T} = 0$$

Take the transpose of both sides:

$$\mathbf{X}^{T}(\boldsymbol{y} - \mathbf{X}\boldsymbol{w}) - \lambda\boldsymbol{w} = 0$$

$$\mathbf{X}^{T}\boldsymbol{y} - (\mathbf{X}^{T}\mathbf{X} + \lambda\mathbf{I})\boldsymbol{w} = 0$$

Which gives the closed form solution:

$$\boldsymbol{w} = (\mathbf{X}^{T}\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^{T}\boldsymbol{y}$$

## How does Lasso "select features"?

When being asked this question many people pull up this picture:
![placeholder](https://www.mlalgorithms.org/static/figure.l1_l2_regression.1-0af871ca059475802831110c99694ba6-16418.png)
*A graph seen by many but understood by few*

Here is a thorough explanation of the picture. Lasso attempts to solve the following convex optimization problem:

$$\min_{\boldsymbol{w}}\frac{1}{n}\sum_{i = 1}^{n}(y_{i} - \boldsymbol{w}^{T}\boldsymbol{x}_{i})^{2} + \lambda\|\boldsymbol{w}\|_{1}$$

In fact, the above is the Lagrangian of the following constrained problem:

$$\min_{\boldsymbol{w}}\frac{1}{n}\sum_{i = 1}^{n}(y_{i} - \boldsymbol{w}^{T}\boldsymbol{x}_{i})^{2}$$

$$s.t. \|\boldsymbol{w}\|_{1} \leq t$$

where $$t$$ is some constant. 

This picture describes the simple case where $$\boldsymbol{w}$$ only has 2 dimensions. The cocentric ellipses are the contour lines of $$\frac{1}{n}\sum_{i = 1}^{n}(y_{i} - \boldsymbol{w}^{T}\boldsymbol{x}_{i})^{2}$$, with its global minimum at the red dot (outer contours represent larger function values). The purple diamond represents the bounded region where $$\|\boldsymbol{w}\|_{1} \leq t$$. 

Now, solving this constrained optimization problem requires finding a point within the diamond such that it lies on a contour line with a small function value as possible. It is very likely that this point occurs at a vertex of the diamond, in which case one entry of the weight vector must be zero. 
As a result, one of the features will have a zero coefficient (since linear regression takes the dot product between the weight and feature vectors) and has no impact on $$y$$.

Still, Lasso does not always yield zero entries for the weight vector - but it happens often. According to *Elements of Statistical Learning*, with high dimensional data there are many more opportunities for the estimated weights to be zero because the purple diamond will become a rhomboid with a lot of vertices.

## What are the probabilistic interpretations of Lasso and ridge regression?

**Basically, Lasso and ridge regression impose different constraints on the probability distribution of the weight**. Recall that linear regression attempts to model a relationship of the form

$$y = \boldsymbol{w}^{T}\boldsymbol{x} + \epsilon$$

where $$\epsilon = y - \boldsymbol{w}^{T}\boldsymbol{x} = y - \hat{y} \sim N(0, \sigma^{2})$$ is a random noise variable that represents the model's perturbations from reality. Given some training data $$(\boldsymbol{x}_{1}, y_{1}),... (\boldsymbol{x}_{n}, y_{n})$$, one may fit a model by finding a maximum-likelihood estimation of its parameter $$\boldsymbol{w}$$. The likelihood is the probability of observing the training labels $$\boldsymbol{y}$$, given the
training samples $$\mathbf{X}$$ and the parameter $$\boldsymbol{w}$$:

$$L(\boldsymbol{w}) = P(\boldsymbol{y}\vert\mathbf{X}; \boldsymbol{w}) = \prod_{i = 1}^{n}P(y_{i} \vert \boldsymbol{x}_{i}; \boldsymbol{w})$$

Since $$\epsilon \sim N(0, \sigma^{2})$$, $$y\vert\boldsymbol{x}_{i} \sim N(\boldsymbol{w}^{T}\boldsymbol{x}_{i}, \sigma^{2})$$. Therefore the above equation becomes:

$$L(\boldsymbol{w}) = \prod_{i = 1}^{n}\frac{1}{\sqrt{2\pi\sigma^{2}}}e^{-\frac{(y_{i} - \boldsymbol{w}^{T}\boldsymbol{x}_{i})^{2}}{2\sigma^{2}}}$$

Taking its logarithm gives the log-likelihood form:

$$l(\boldsymbol{w}) = \sum_{i = 1}^{n}\log(\frac{1}{\sqrt{2\pi\sigma^{2}}})-\sum_{i = 1}^{n}\frac{(y_{i} -\boldsymbol{w}^{T}\boldsymbol{x}_{i})^{2}}{2\sigma^{2}}$$

To find a best $$\boldsymbol{w}$$ we attempt to maximize this log-likelihood. Notice that:

$$\underset{\boldsymbol{w}}{\arg\max}\;l(\boldsymbol{w}) = \underset{\boldsymbol{w}}{\arg\max}-\sum_{i = 1}^{n}(y_{i} -\boldsymbol{w}^{T}\boldsymbol{x}_{i})^{2} = \underset{\boldsymbol{w}}{\arg\min}\;J(\boldsymbol{w})$$

In other words, minimizing the cost is the same problem as maximizing the log likelihood, giving the optimization problem a graceful probabilistic interpretation.

With Lasso, each term in the weight vector $$\boldsymbol{w}$$ is forced to have a Laplacian prior distribution with mean 0 and some scale $$b$$:

$$w_{i} \sim Laplace(0, b)$$

$$P(\boldsymbol{w})= \prod_{i = 1}^{m}\frac{1}{2b}e^{\frac{-\vert w_{i}\vert}{b}}$$

The likelihood is multiplied by the prior to obtain the maximum a posteriori (MAP) objective:

$$P(\boldsymbol{y}\vert\mathbf{X}; \boldsymbol{w})P(\boldsymbol{w}) = \prod_{i = 1}^{n}\frac{1}{\sqrt{2\pi\sigma^{2}}}e^{-\frac{(y_{i} - \boldsymbol{w}^{T}\boldsymbol{x}_{i})^{2}}{2\sigma^{2}}} \prod_{i = 1}^{m}\frac{1}{2b}e^{\frac{-\vert w_{i}\vert}{b}}$$

Taking the logarithm:

$$\sum_{i = 1}^{n}\log(\frac{1}{\sqrt{2\pi\sigma^{2}}})-\sum_{i = 1}^{n}\frac{(y_{i} -\boldsymbol{w}^{T}\boldsymbol{x}_{i})^{2}}{2\sigma^{2}} + \sum_{i = 1}^{m}\log(\frac{1}{2b}) - \sum_{i = 1}^{m}\frac{\vert w_{i}\vert}{b}$$

After taking the argmax and removing irrelevant terms:

$$\underset{\boldsymbol{w}}{\arg\max}\;l(\boldsymbol{w}) = \underset{\boldsymbol{w}}{\arg\min}\sum_{i = 1}^{n}\frac{(y_{i} -\boldsymbol{w}^{T}\boldsymbol{x}_{i})^{2}}{2\sigma^{2}} + \sum_{i = 1}^{m}\frac{\vert w_{i}\vert}{b}$$

$$= \underset{\boldsymbol{w}}{\arg\min}\sum_{i = 1}^{n}(y_{i} -\boldsymbol{w}^{T}\boldsymbol{x}_{i})^{2} + \frac{2\sigma^{2}}{b}\|\boldsymbol{w}\|_{1}$$

Now this clearly resembles the Lasso cost function described previously, except that in the latter, the term $$2\sigma^{2}/b$$ is represented by a single parameter $$\lambda$$. Since $$\sigma$$ is fixed for a given training set, $$\lambda$$ is actually inversely proportional to the scale parameter $$b$$ of the distribution of weights. 

With ridge regression, $$\boldsymbol{w}$$ has a prior in the form of a multivariate normal distribution:

$$\boldsymbol{w} \sim N(0, a^{2}\mathbf{I})$$

In other words:

$$w_{i} \sim N(0, a^{2})$$

$$P(\boldsymbol{w}) = \prod_{i = 1}^{m}\frac{1}{\sqrt{2\pi a^{2}}}e^{-\frac{w_{i}^{2}}{2a^{2}}}$$

where $$a$$ is some constant. In a similar fashion as above, minimizing the cost function of ridge regression can be shown as equivalent to maximizing $$P(\boldsymbol{y}\vert\mathbf{X}; \boldsymbol{w})P(\boldsymbol{w})$$. 

## What is a compromise between Lasso and ridge regression?
The elasic net regression takes the linear combination of $$L^{1}$$ and $$L^{2}$$ penalties as the regularization term:

$$J(\boldsymbol{w}) = \frac{1}{n}\sum_{i = 1}^{n}(y_{i} - \boldsymbol{w}^{T}\boldsymbol{x}_{i})^{2} + \lambda_{1}\|\boldsymbol{w}\|_{1} + \lambda_{2}\|\boldsymbol{w}\|_{2}$$



