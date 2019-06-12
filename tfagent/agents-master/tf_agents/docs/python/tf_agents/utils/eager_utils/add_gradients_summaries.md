<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.eager_utils.add_gradients_summaries" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.eager_utils.add_gradients_summaries

Add summaries to gradients.

``` python
tf_agents.utils.eager_utils.add_gradients_summaries(
    grads_and_vars,
    step
)
```



Defined in [`utils/eager_utils.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/eager_utils.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`grads_and_vars`</b>: A list of gradient to variable pairs (tuples).
* <b>`step`</b>: Variable to use for summaries.