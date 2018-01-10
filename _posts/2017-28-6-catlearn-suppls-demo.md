Background
==========

This post is intended to help people get started with their own cognitive modelling or machine learning projects in the `catlearn` environment with the help of `catlearn.suppls`, the [Learning and Representation in Cognition Laboratory](http://kurtzlab.psychology.binghamton.edu/)'s suite of helper functions for modeling in `catlearn`.

The motivation behind `catlearn.suppls` was to create a suite of functions that could be used to rapidly prototype new machine learning architectures that correspond to the cognitive modeling efforts of the LaRC Lab. This includes functions to set up the *state* information of a model and quickly take rectangular data and format it for the `catlearn` design pattern---the stateful list processor.

Getting Started
===============

If you haven't already done so, download the `catlearn` and `catlearn.suppls` packages.

``` r
install.packages(c("devtools", "catlearn"))
devtools::install_github("ghonk/catlearn.suppls")
```

Choosing some data and setting up the model
===========================================

We're going to use the classic `iris` dataset for this demonstration. We need two objects to run a model in `catlearn`, the model's `state` and the training matrix, which we will call `tr`.
