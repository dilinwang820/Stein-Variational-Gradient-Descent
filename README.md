# Variational Gradient Descent (VGD)
VGD is a general purpose variational inference algorithm that forms a natural counterpart of gradient descent for optimization. VGD iteratively transports a set of particles to match with the target distribution, by applying a form of functional gradient descent that minimizes the KL divergence.

The package contains implementation of VGD in both Matlab and Python language. Demos are also provided to reproduce the results in our paper. The Bayesian neural network example is based on Theano 0.8.2.

For more information, please visit our project website. [VGD](http://www.cs.dartmouth.edu/~dartml/project.html)

## Toy example on 1D Gaussian Mixture
Toy example with 1D Gaussian mixture. The red dashed lines are the target density function and the solid green lines are the densities of the particles at different iterations of our algorithm (estimated using kernel density estimator).

<img src="http://cs.dartmouth.edu/~dartml/1dgmm.gif" width="50%">

## Basic Usage
```python
'''
  x0: initial particles
  dlnprob: returns first order derivative of log probability
  n_iter: number of iterations
  stepsize: initial learning rate 
'''
theta = VGD().update(x0, dlnprob, n_iter, stepsize)
```

## Citation
//TO APPEAR

## Feedback
Feedback is greatly appreciated. If you have any questions, comments, issues or anything else really, [shoot me an email](mailto:dilin.wang.gr@dartmouth.edu).


@Copyright (c) 2016,  Qiang Liu & Dilin Wang.
All rights reserved.
