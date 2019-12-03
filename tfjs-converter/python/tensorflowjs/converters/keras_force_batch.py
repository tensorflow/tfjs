import contextlib
import copy

from tensorflow.python.keras.saving import model_config


class BatchSizeEnforcer(object):
  """Patches Keras to force unit batch sizes to help shape calculation."""

  def __init__(self):
    self._original_model_from_config = None

  def enable(self):
    assert self._original_model_from_config is None
    self._original_model_from_config = model_config.model_from_config
    model_config.model_from_config = self._patched_model_from_config

  def disable(self):
    assert self._original_model_from_config
    model_config.model_from_config = self._original_model_from_config
    self._original_model_from_config = None

  def _patched_model_from_config(self, config, *args, **kwargs):
    """Replacement for keras.engine.saving.model_from_config."""
    patched_config = self._patch_config(config)
    return self._original_model_from_config(patched_config, *args, **kwargs)

  def _patch_config(self, config):
    """Force unknown batch sizes of inputs equal 1 in a Keras model config.

    Args:
      config: The initial model configuration (nested plain Python dicts/lists).

    Returns:
      The model configuration with unknown batch sizes of inputs forced to be 1.
    """
    patched_config = copy.deepcopy(config)
    layers = patched_config['config']['layers']
    for layer in layers:
      layer_config = layer['config']
      try:
        batch_input_shape = layer_config['batch_input_shape']
      except KeyError:
        pass
      else:
        if batch_input_shape[0] is None:
          batch_input_shape = [1] + batch_input_shape[1:]
          layer_config['batch_input_shape'] = batch_input_shape
    return patched_config


@contextlib.contextmanager
def fixed_batch_size():
  batch_size_enforcer = BatchSizeEnforcer()
  batch_size_enforcer.enable()
  try:
    yield
  finally:
    batch_size_enforcer.disable()