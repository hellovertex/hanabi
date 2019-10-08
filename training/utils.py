import tensorflow as tf

def has_eager_been_enabled():
  """Returns true iff in TF2 or in TF1 with eager execution enabled."""
  with tf.init_scope():
    return tf.executing_eagerly()

def create_variable(name,
                    initial_value=0,
                    shape=(),
                    dtype=tf.int64,
                    use_local_variable=False,
                    trainable=False,
                    initializer=None,
                    unique_name=True):
  """Create a variable."""
  if has_eager_been_enabled():
    if initializer is None:
      if shape:
        initial_value = tf.constant(initial_value, shape=shape, dtype=dtype)
      else:
        initial_value = tf.convert_to_tensor(initial_value, dtype=dtype)
    else:
      if callable(initializer):
        initial_value = lambda: initializer(shape)
      else:
        initial_value = initializer
    return tf.compat.v2.Variable(initial_value,
                                 trainable=trainable,
                                 dtype=dtype,
                                 name=name)
  collections = [tf.compat.v1.GraphKeys.GLOBAL_VARIABLES]
  if use_local_variable:
    collections = [tf.compat.v1.GraphKeys.LOCAL_VARIABLES]
  if initializer is None:
    initializer = tf.compat.v1.initializers.constant(initial_value, dtype=dtype)
  if unique_name:
    name = tf.compat.v1.get_default_graph().unique_name(name)
  return tf.compat.v1.get_variable(
      name=name,
      shape=shape,
      dtype=dtype,
      initializer=initializer,
      collections=collections,
      use_resource=True,
      trainable=trainable)