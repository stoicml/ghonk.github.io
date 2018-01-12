---
title: Grid Search with Catlearn and Catlearn Supplementals
excerpt: Quick tutorial on grid search with Catlearn and Catlearn Supplementals
tags: ml, datascience, DIVA
season: summer 2017
type: blog
layout: post
---


## load libraries 


{% highlight r %}
library(catlearn)
library(catlearn.suppls)
{% endhighlight %}

## *initialize variables.* Create a named list for each hyperparameter you plan to test in the grid search, specify how many random model initializations you'd like to average across to calculate response probabilities for each parameter combination.  


{% highlight r %}
# # parameter list
short_param_list <- list(beta_val = c(0, 1, 2, 3),
                    learning_rate = c(.05, .10, .15),
                         num_hids = c(4, 5, 6))

# long_param_list <- list(beta_val = c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
#                    learning_rate = c(.05, .10, .15, .20, .25, .35),
#                         num_hids = c(4, 5, 6, 10, 20))

# # # number of initializations
num_inits = 4

# # # data
# input_list <- get_test_inputs('type1')
input_list <- get_test_inputs('type4')

# # # fit type
fit_type <- 'bestacc'

# # # fit vector
crit_fit_vector <- NULL
{% endhighlight %}

## run it


{% highlight r %}
# # # single core
system.time(gs_output <- diva_grid_search(short_param_list, num_inits, input_list))
{% endhighlight %}



{% highlight text %}
##    user  system elapsed 
##    3.95    0.00    3.99
{% endhighlight %}



{% highlight r %}
# # # parallelized
system.time(gs_output <- diva_grid_search_par(short_param_list, num_inits, input_list))
{% endhighlight %}



{% highlight text %}
##    user  system elapsed 
##    0.03    0.00    3.04
{% endhighlight %}

## examine the results with `plot_training`


{% highlight r %}
plot_training(lapply(gs_output, function(x) x$resp_probs))
{% endhighlight %}

![plot of chunk unnamed-chunk-16](/assets/rfigs/unnamed-chunk-16-1.svg)

## examine the detailed results of a grid search run


{% highlight r %}
# # # how many paramter settings did we have?
(n_models <- length(gs_output))
{% endhighlight %}



{% highlight text %}
## [1] 36
{% endhighlight %}



{% highlight r %}
# # # what was the accuracy distribution?
final_accuracy <- lapply(gs_output, function(x) {x$resp_probs[12]})
plot(1:n_models, final_accuracy)
{% endhighlight %}

![plot of chunk unnamed-chunk-17](/assets/rfigs/unnamed-chunk-17-1.svg)

{% highlight r %}
# # # what parameter setting had the best performance?
gs_output[[which.max(final_accuracy)]]$params
{% endhighlight %}



{% highlight text %}
##    beta_val learning_rate num_hids
## 33        0          0.15        6
{% endhighlight %}



{% highlight r %}
# # # what parameter setting had the worst perfomance?
gs_output[[which.min(final_accuracy)]]$params
{% endhighlight %}



{% highlight text %}
##   beta_val learning_rate num_hids
## 7        2           0.1        4
{% endhighlight %}



{% highlight r %}
# # # plot em
plot_training(list(gs_output[[which.max(final_accuracy)]]$resp_probs, 
  gs_output[[which.min(final_accuracy)]]$resp_probs))
{% endhighlight %}

![plot of chunk unnamed-chunk-17](/assets/rfigs/unnamed-chunk-17-2.svg)

{% highlight r %}
# # # what comes as output for each parameter setting?
names(gs_output[[which.max(final_accuracy)]])
{% endhighlight %}



{% highlight text %}
## [1] "resp_probs" "params"     "st"
{% endhighlight %}



{% highlight r %}
# # # plot the training curves for a parameter subset (hid units = 5)
hidunit5_respprobs <- list()
for (i in 1:length(gs_output)) {
  if (gs_output[[i]]$params$num_hids == 5){
    hidunit5_respprobs[[paste0(i)]] <- gs_output[[i]]$resp_probs
  } 
}

# # # how many?
(n_models <- length(hidunit5_respprobs))
{% endhighlight %}



{% highlight text %}
## [1] 12
{% endhighlight %}



{% highlight r %}
# # # accuracy?
plot(1:n_models, unlist(lapply(hidunit5_respprobs, function(x) x[[12]])))
{% endhighlight %}

![plot of chunk unnamed-chunk-17](/assets/rfigs/unnamed-chunk-17-3.svg)

{% highlight r %}
plot_training(hidunit5_respprobs)
{% endhighlight %}

![plot of chunk unnamed-chunk-17](/assets/rfigs/unnamed-chunk-17-4.svg)







