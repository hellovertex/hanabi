<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.eager_utils.create_train_step" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.eager_utils.create_train_step

Creates a train_step that evaluates the gradients and returns the loss.

``` python
tf_agents.utils.eager_utils.create_train_step(
    loss,
    optimizer,
    global_step=_USE_GLOBAL_STEP,
    total_loss_fn=None,
    update_ops=None,
    variables_to_train=None,
    transform_grads_fn=None,
    summarize_gradients=False,
    gate_gradients=tf.compat.v1.train.Optimizer.GATE_OP,
    aggregation_method=None,
    check_numerics=True
)
```



Defined in [`utils/eager_utils.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/eager_utils.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`loss`</b>: A (possibly nested tuple of) `Tensor` or function representing
    the loss.
* <b>`optimizer`</b>: A tf.Optimizer to use for computing the gradients.
* <b>`global_step`</b>: A `Tensor` representing the global step variable. If left as
    `_USE_GLOBAL_STEP`, then tf.train.get_or_create_global_step() is used.
* <b>`total_loss_fn`</b>: Function to call on loss value to access the final
   item to minimize.
* <b>`update_ops`</b>: An optional list of updates to execute. If `update_ops` is
    `None`, then the update ops are set to the contents of the
    `tf.GraphKeys.UPDATE_OPS` collection. If `update_ops` is not `None`, but
    it doesn't contain all of the update ops in `tf.GraphKeys.UPDATE_OPS`,
    a warning will be displayed.
* <b>`variables_to_train`</b>: an optional list of variables to train. If None, it will
    default to all tf.trainable_variables().
* <b>`transform_grads_fn`</b>: A function which takes a single argument, a list of
    gradient to variable pairs (tuples), performs any requested gradient
    updates, such as gradient clipping or multipliers, and returns the updated
    list.
* <b>`summarize_gradients`</b>: Whether or not add summaries for each gradient.
* <b>`gate_gradients`</b>: How to gate the computation of gradients. See tf.Optimizer.
* <b>`aggregation_method`</b>: Specifies the method used to combine gradient terms.
    Valid values are defined in the class `AggregationMethod`.
* <b>`check_numerics`</b>: Whether or not we apply check_numerics.


#### Returns:

In graph mode: A (possibly nested tuple of) `Tensor` that when evaluated,
  calculates the current loss, computes the gradients, applies the
  optimizer, and returns the current loss.
In eager mode: A lambda function that when is called, calculates the loss,
  then computes and applies the gradients and returns the original
  loss values.

#### Raises:

* <b>`ValueError`</b>: if loss is not callable.
* <b>`RuntimeError`</b>: if resource variables are not enabled.