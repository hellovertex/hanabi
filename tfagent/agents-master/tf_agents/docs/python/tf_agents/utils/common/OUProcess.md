<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.common.OUProcess" />
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
<meta itemprop="property" content="with_name_scope"/>
</div>

# tf_agents.utils.common.OUProcess

## Class `OUProcess`

A zero-mean Ornstein-Uhlenbeck process.





Defined in [`utils/common.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/common.py).

<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    initial_value,
    damping=0.15,
    stddev=0.2,
    seed=None,
    scope='ornstein_uhlenbeck_noise'
)
```

A Class for generating noise from a zero-mean Ornstein-Uhlenbeck process.

The Ornstein-Uhlenbeck process is a process that generates temporally
correlated noise via a random walk with damping. This process describes
the velocity of a particle undergoing brownian motion in the presence of
friction. This can be useful for exploration in continuous action
environments with momentum.

The temporal update equation is:
`x_next = (1 - damping) * x + N(0, std_dev)`

#### Args:

* <b>`initial_value`</b>: Initial value of the process.
* <b>`damping`</b>: The rate at which the noise trajectory is damped towards the
    mean. We must have 0 <= damping <= 1, where a value of 0 gives an
    undamped random walk and a value of 1 gives uncorrelated Gaussian noise.
    Hence in most applications a small non-zero value is appropriate.
* <b>`stddev`</b>: Standard deviation of the Gaussian component.
* <b>`seed`</b>: Seed for random number generation.
* <b>`scope`</b>: Scope of the variables.



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
__call__()
```



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



