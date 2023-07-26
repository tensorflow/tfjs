# Copyright 2023 Google LLC
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
import tensorflow.keras as keras
import inspect

from tensorflow.python.util import tf_export


TFCLASS_MODULE_MAP = {}
MODULE = keras

def _build_class_module_map(keras_module):
    """Build the map between TFJS classes and corresponding module path in TF.

    Args:
      keras_module: keras module used to go through all the classes
    """
    for name, obj in inspect.getmembers(keras_module):
        if inspect.isclass(obj):
            # Retrive the module path from tensorflow.
            parts = str(tf_export.get_canonical_name_for_symbol(obj, api_name='keras')).split(".")
            # Map the class name with module path exclude the class name.
            TFCLASS_MODULE_MAP[name] = ".".join(parts[:-1])

        elif inspect.ismodule(obj):
            _build_class_module_map(obj)

def get_module_path(key):
    """Get the module path base on input key

    Args:
      key: The name of the class we want to get module path.
    Return:
      RESULT_MAP[key]: the corresponding module path in TF.
    """
    if not TFCLASS_MODULE_MAP:
        _build_class_module_map(MODULE)
    if key not in TFCLASS_MODULE_MAP:
        raise KeyError(f"Cannot find the module path for {key} class.")
    return TFCLASS_MODULE_MAP[key]
