<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.metrics.tf_metric.TFStepMetric" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="name_scope"/>
<meta itemprop="property" content="submodules"/>
<meta itemprop="property" content="trainable_variables"/>
<meta itemprop="property" content="variables"/>
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__delattr__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__setattr__"/>
<meta itemprop="property" content="call"/>
<meta itemprop="property" content="init_variables"/>
<meta itemprop="property" content="result"/>
<meta itemprop="property" content="tf_summaries"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# tf_agents.metrics.tf_metric.TFStepMetric

## Class `TFStepMetric`

Defines the interface for TF metrics.





Defined in [`metrics/tf_metric.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/metrics/tf_metric.py).

<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    name,
    prefix='Metrics'
)
```





## Properties

<h3 id="name"><code>name</code></h3>

Returns the name of this module as passed or determined in the ctor.

NOTE: This is not the same as the `self.name_scope.name` which includes
parent module names.

<h3 id="name_scope"><code>name_scope</code></h3>

Returns a `tf.name_scope` instance for this class.

<h3 id="submodules"><code>submodules</code></h3>

Sequence of all sub-modules.

Submodules are modules which are properties of this module, or found as
properties of modules which are properties of this module (and so on).

>>> a = tf.Module()
>>> b = tf.Module()
>>> c = tf.Module()
>>> a.b = b
>>> b.c = c
>>> assert list(a.submodules) == [b, c]
>>> assert list(b.submodules) == [c]
>>> assert list(c.submodules) == []

#### Returns:

A sequence of all submodules.

<h3 id="trainable_variables"><code>trainable_variables</code></h3>

Sequence of variables owned by this module and it's submodules.

Note: this method uses reflection to find variables on the current instance
and submodules. For performance reasons you may wish to cache the result
of calling this method if you don't expect the return value to change.

#### Returns:

A sequence of variables for the current module (sorted by attribute
name) followed by variables from all submodules recursively (breadth
first).

<h3 id="variables"><code>variables</code></h3>

Sequence of variables owned by this module and it's submodules.

Note: this method uses reflection to find variables on the current instance
and submodules. For performance reasons you may wish to cache the result
of calling this method if you don't expect the return value to change.

#### Returns:

A sequence of variables for the current module (sorted by attribute
name) followed by variables from all submodules recursively (breadth
first).



## Methods

<h3 id="__call__"><code>__call__</code></h3>

``` python
__call__(
    *args,
    **kwargs
)
```

Returns op to execute to update this metric for these inputs.

Returns None if eager execution is enabled.
Returns a graph-mode function if graph execution is enabled.

#### Args:

* <b>`*args`</b>: * <b>`**kwargs`</b>: A mini-batch of inputs to the Metric, passed on to `call()`.

<h3 id="__delattr__"><code>__delattr__</code></h3>

``` python
__delattr__(name)
```



<h3 id="__setattr__"><code>__setattr__</code></h3>

``` python
__setattr__(
    name,
    value
)
```

Support self.foo = trackable syntax.

<h3 id="call"><code>call</code></h3>

``` python
call(
    *args,
    **kwargs
)
```

Accumulates statistics for the metric. Users should use __call__ instead.

Note: This function is executed as a graph function in graph mode.
This means:
a) Operations on the same resource are executed in textual order.
   This should make it easier to do things like add the updated
   value of a variable to another, for example.
b) You don't need to worry about collecting the update ops to execute.
   All update ops added to the graph by this function will be executed.
As a result, code should generally work the same way with graph or
eager execution.

#### Args:

* <b>`*args`</b>: * <b>`**kwargs`</b>: A mini-batch of inputs to the Metric, as passed to
    `__call__()`.

<h3 id="init_variables"><code>init_variables</code></h3>

``` python
init_variables()
```

Initializes this Metric's variables.

Should be called after variables are created in the first execution
of `__call__()`. If using graph execution, the return value should be
`run()` in a session before running the op returned by `__call__()`.
(See example above.)

#### Returns:

If using graph execution, this returns an op to perform the
initialization. Under eager execution, the variables are reset to their
initial values as a side effect and this function returns None.

<h3 id="result"><code>result</code></h3>

``` python
result()
```

Computes and returns a final value for the metric.

<h3 id="tf_summaries"><code>tf_summaries</code></h3>

``` python
tf_summaries(
    train_step=None,
    step_metrics=()
)
```

Generates summaries against train_step and all step_metrics.

#### Args:

* <b>`train_step`</b>: (Optional) Step counter for training iterations. If None, no
    metric is generated against the global step.
* <b>`step_metrics`</b>: (Optional) Iterable of step metrics to generate summaries
    against.


#### Returns:

A list of summaries.

<h3 id="with_name_scope"><code>with_name_scope</code></h3>

``` python
with_name_scope(
    cls,
    method
)
```

Decorator to automatically enter the module name scope.

>>> class MyModule(tf.Module):
...   @tf.Module.with_name_scope
...   def __call__(self, x):
...     if not hasattr(self, 'w'):
...       self.w = tf.Variable(tf.random.normal([x.shape[1], 64]))
...     return tf.matmul(x, self.w)

Using the above module would produce `tf.Variable`s and `tf.Tensor`s whose
names included the module name:

>>> mod = MyModule()
>>> mod(tf.ones([8, 32]))
<tf.Tensor: ...>
>>> mod.w
<tf.Variable ...'my_module/w:0'>

#### Args:

* <b>`method`</b>: The method to wrap.


#### Returns:

The original method wrapped such that it enters the module's name scope.



