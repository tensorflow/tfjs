# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unit tests for converting JAX to TensorFlow.js web format."""
import functools

from flax import linen as nn
from jax import random
import jax.numpy as jnp
import tensorflow as tf
from tensorflowjs.converters import jax_conversion


class FlaxModule(nn.Module):
  """A simple Flax Module containing a few Dense layers and ReLUs."""

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(features=20)(x)
    x = nn.relu(x)
    for _ in range(5):
      x = nn.Dense(features=10)(x)
      x = nn.relu(x)

    x = nn.Dense(features=2)(x)
    x = nn.sigmoid(x)
    return x


class FlaxModuleBatchNorm(nn.Module):
  """A simple CNN-like Flax model with BatchNorm."""

  @nn.compact
  def __call__(self, x, *, training=True):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.BatchNorm(use_running_average=not training)(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.BatchNorm(use_running_average=not training)(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    return x


class JaxConversionTest(tf.test.TestCase):

  def test_convert_simple(self):
    apply_fn = lambda params, x: jnp.sum(x) * params['w']
    jax_conversion.convert_jax(
        apply_fn,
        {'w': 0.5},
        input_signatures=[tf.TensorSpec((2, 3), tf.float32)],
        model_dir=self.get_temp_dir())

  def test_convert_poly(self):
    apply_fn = lambda params, x: jnp.sum(x) * params['w']
    jax_conversion.convert_jax(
        apply_fn,
        {'w': 0.5},
        input_signatures=[tf.TensorSpec((None, 3), tf.float32)],
        polymorphic_shapes=['(b, 3)'],
        model_dir=self.get_temp_dir())

  def test_convert_tf_poly_mismatch_raises(self):
    apply_fn = lambda params, x: jnp.sum(x) * params['w']
    with self.assertRaisesRegex(
        ValueError, 'polymorphic shape.* must match .* for argument shape'):
      jax_conversion.convert_jax(
          apply_fn,
          {'w': 0.5},
          input_signatures=[tf.TensorSpec((None, 3), tf.float32)],
          polymorphic_shapes=['(b, 4)'],
          model_dir=self.get_temp_dir())

  def test_convert_multiargs(self):
    apply_fn = lambda params, x, y: jnp.sum(x) * jnp.sum(y) * params['w']
    jax_conversion.convert_jax(
        apply_fn,
        {'w': 0.5},
        input_signatures=[tf.TensorSpec((2, 3), tf.float32),
                          tf.TensorSpec((5, 6), tf.float32)],
        model_dir=self.get_temp_dir())

  def test_convert_multiarg_poly(self):
    apply_fn = lambda params, x, y: jnp.sum(x) * jnp.sum(y) * params['w']
    jax_conversion.convert_jax(
        apply_fn,
        {'w': 0.5},
        input_signatures=[tf.TensorSpec((None, 3), tf.float32),
                          tf.TensorSpec((None, 6), tf.float32)],
        polymorphic_shapes=['(b, 3)', '(b, 6)'],
        model_dir=self.get_temp_dir())

  def test_convert_flax(self):
    m, x = FlaxModule(), jnp.zeros((3, 4))
    variables = m.init(random.PRNGKey(0), x)
    jax_conversion.convert_jax(
        m.apply,
        variables,
        input_signatures=[tf.TensorSpec((3, 4), tf.float32)],
        model_dir=self.get_temp_dir())

  def test_convert_flax_poly(self):
    m, x = FlaxModule(), jnp.zeros((3, 4))
    variables = m.init(random.PRNGKey(0), x)
    jax_conversion.convert_jax(
        m.apply,
        variables,
        input_signatures=[tf.TensorSpec((None, 4), tf.float32)],
        polymorphic_shapes=['(b, 4)'],
        model_dir=self.get_temp_dir())

  # TODO(marcvanzee): This test currently fails due to
  # https://github.com/google/jax/issues/11804.
  # This issue is fixed in JAX, but only will be part of jax>0.3.16. Once JAX
  # releases a new version we can re-enable this test. If you install JAX from
  # Github this will work fine.
  # def test_convert_flax_bn(self):
  #   m, x = FlaxModuleBatchNorm(), jnp.zeros((1, 32, 32, 3))
  #   variables = m.init(random.PRNGKey(0), x)
  #   # Note: if we don't pass training=False here, we will get an error during
  #   # conversion since `batch_stats` is mutated while it is not passed as a
  #   # mutable variable collections (we currently do not support this).
  #   apply_fn = functools.partial(m.apply, training=False)
  #   jax_conversion.convert_jax(
  #       apply_fn,
  #       variables,
  #       input_signatures=[tf.TensorSpec((1, 32, 32, 3), tf.float32)],
  #       model_dir=self.get_temp_dir())


if __name__ == '__main__':
  tf.test.main()
