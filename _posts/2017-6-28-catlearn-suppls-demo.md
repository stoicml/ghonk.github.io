---
title: Getting Started with Catlearn and Catlearn Supplementals
excerpt: Brief introduction to machine learning and cognitive modeling with Catlearn and Catlearn Supplementals
tags: machine learning, cognitive modeling, catlearn
season: summer 2017
type: blog
layout: post
---



# background

This post is intended to help people get started with their own cognitive modelling or machine learning projects in the `catlearn` environment with the help of `catlearn.suppls`, the [Learning and Representation in Cognition Laboratory](http://kurtzlab.psychology.binghamton.edu/)'s suite of helper functions for modeling in `catlearn`.

The motivation behind `catlearn.suppls` was to create a suite of functions that could be used to rapidly prototype new machine learning architectures that correspond to the cognitive modeling efforts of the LaRC Lab. This includes functions to set up the *state* information of a model and quickly take rectangular data and format it for the `catlearn` design pattern---the stateful list processor.

# getting Started

If you haven't already done so, download the `catlearn` and `catlearn.suppls` packages.


{% highlight r %}
install.packages(c("devtools", "catlearn"))
devtools::install_github("ghonk/catlearn.suppls")
{% endhighlight %}

# choosing some data and setting up the model

We're going to use the classic `iris` dataset and DIVA (the DIVergent Autoencoder) for this demonstration. We need two objects to run a model in `catlearn`, the model's `state` and the training matrix, which we will call `tr`.


{% highlight r %}
# # # load the libraries
library(catlearn)
library(catlearn.suppls)
{% endhighlight %}

*First, we will construct the model's state.* For this demonstration we'll set the hyper-parameters to values that we know---a priori---will play nice with our dataset. For real-world problems, you will likely want optimize these values (see future post on grid search and Bayesian optimization options with `catlearn.suppls`). Detailed description of model hyper-parameters is available in the normal places (e.g., `?slpDIVA`).


{% highlight r %}
# # # setup the inputs, class labels and model state
# check out our data
str(iris)
{% endhighlight %}



{% highlight text %}
## 'data.frame':	150 obs. of  5 variables:
##  $ Sepal.Length: num  5.1 4.9 4.7 4.6 5 5.4 4.6 5 4.4 4.9 ...
##  $ Sepal.Width : num  3.5 3 3.2 3.1 3.6 3.9 3.4 3.4 2.9 3.1 ...
##  $ Petal.Length: num  1.4 1.4 1.3 1.5 1.4 1.7 1.4 1.5 1.4 1.5 ...
##  $ Petal.Width : num  0.2 0.2 0.2 0.2 0.2 0.4 0.3 0.2 0.2 0.1 ...
##  $ Species     : Factor w/ 3 levels "setosa","versicolor",..: 1 1 1 1 1 1 1 1 1 1 ...
{% endhighlight %}



{% highlight r %}
# find the inputs minus the labels
ins <- iris[,colnames(iris) != 'Species']

# create a separate vector for the labels (labels must be numeric)
labs <- as.numeric(iris[,'Species'])

# get number of categories and features
nfeats <- dim(ins)[2]
ncats <- length(unique(labs))

# construct a state list
st <- list(learning_rate = 0.15, num_feats = nfeats, num_hids = 6, num_cats = ncats,
  beta_val = 0, phi = 1, continuous = TRUE, in_wts = NULL, out_wts = NULL, wts_range = 1,
  wts_center = 0, colskip = 4)
{% endhighlight %}

We can then use `catleanr.suppls` to create our training matrix


{% highlight r %}
# tr_init initializes an empty training matrix
tr <- tr_init(nfeats, ncats)

# tr_add fills in the data and procedure (i.e., training, test, model reset)
tr <- tr_add(inputs = ins, tr = tr, labels = labs, blocks = 12, ctrl = 0, 
  shuffle = TRUE, reset = TRUE)
{% endhighlight %}

Here's what our training matrix looks like after the setup procedure:

{% highlight r %}
head(tr)
{% endhighlight %}



{% highlight text %}
##      ctrl trial blk example  x1  x2  x3  x4 t1 t2 t3
## [1,]    1     1   1     118 7.7 3.8 6.7 2.2 -1 -1  1
## [2,]    0     2   1      64 6.1 2.9 4.7 1.4 -1  1 -1
## [3,]    0     3   1      23 4.6 3.6 1.0 0.2  1 -1 -1
## [4,]    0     4   1      87 6.7 3.1 4.7 1.5 -1  1 -1
## [5,]    0     5   1      99 5.1 2.5 3.0 1.1 -1  1 -1
## [6,]    0     6   1      92 6.1 3.0 4.6 1.4 -1  1 -1
{% endhighlight %}

Finally, we run the model with our state list `st` and training matrix `tr`


{% highlight r %}
diva_model <- slpDIVA(st, tr)
{% endhighlight %}

We can examine performance of the model easily. 


{% highlight r %}
# # # use response_probs to extract the response probabilities 
# # # for the target categories (for every training step (trial) 
# # # or averaged across blocks)

response_probs(tr, diva_model$out, blocks = TRUE)
{% endhighlight %}



{% highlight text %}
##  [1] 0.7852448 0.8431547 0.8103888 0.8067568 0.8087171 0.8102305 0.8135465
##  [8] 0.8028162 0.8466575 0.8375568 0.8025279 0.8350697
{% endhighlight %}

`plot_training` is a simple function used to plot the learning of one or more models.


{% highlight r %}
plot_training(list(response_probs(tr, diva_model$out, blocks = TRUE)))
{% endhighlight %}

![plot of chunk unnamed-chunk-9](/assets/rfigs/unnamed-chunk-9-1.svg)

So with no optimization, we can see that DIVA learns about as much as it is going to learn after one pass through our 150 item training set (obviously absent any cross-validation). Where does it go wrong? You might like to examine which items are not being correctly classified---you can do so by combining the classification probabilities with the original training matrix.


{% highlight r %}
# # # if we want to look at individual classication decisions, 
# # # we can do so by combining the model's output with the 
# # # original training matrix

trn_result <- cbind(tr, round(diva_model$out, 4))
tail(trn_result)
{% endhighlight %}



{% highlight text %}
##         ctrl trial blk example  x1  x2  x3  x4 t1 t2 t3                     
## [1795,]    0  1795  12     123 7.7 2.8 6.7 2.0 -1 -1  1 0.0005 0.0006 0.9989
## [1796,]    0  1796  12      82 5.5 2.4 3.7 1.0 -1  1 -1 0.0000 0.9999 0.0000
## [1797,]    0  1797  12     125 6.7 3.3 5.7 2.1 -1 -1  1 0.1204 0.1765 0.7030
## [1798,]    0  1798  12       1 5.1 3.5 1.4 0.2  1 -1 -1 1.0000 0.0000 0.0000
## [1799,]    0  1799  12     144 6.8 3.2 5.9 2.3 -1 -1  1 0.0000 0.0000 1.0000
## [1800,]    0  1800  12     129 6.4 2.8 5.6 2.1 -1 -1  1 0.0001 0.0002 0.9996
{% endhighlight %}

You might also like to see how a number of initializations do on the problem. It's good practice to average over a series of initializations---something we did not do in this demonstration.


{% highlight r %}
# # # Run 5 models with the same params on the same training set
model_inits <- lapply(1:5, function(x) slpDIVA(st, tr))

# # # determine the response probability for the correct class
model_resp_probs <- 
  lapply(model_inits, function(x) {
    response_probs(tr, x$out, blocks = TRUE)})

# # # plot the leanrning curves
plot_training(model_resp_probs)
{% endhighlight %}

![plot of chunk unnamed-chunk-11](/assets/rfigs/unnamed-chunk-11-1.svg)

Here we see that there is a fair amount of variation across initializations. This suggests it would be smart to follow the typical procedure of averaging across a series of models to accurately represent the response probabilities. It also suggests that our approach would likely benefit from some optimization and validation. 


Future demos will explore the tools within `catlearn.suppls` used to optimize hyper-parameters and examine the hidden unit representation space toward the goal of uncovering new insight about the problem. 







