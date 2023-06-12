import tensorflow.keras as keras
import inspect
import os
import re

from tensorflow.python.util import tf_export


TFCLASS_MODULE_MAP = {}
RESULT_MAP = {}
MODULE = keras

def _build_ts_class_module_map(folder_path):
    """Build the map between TFJS classes and corresponding module path in TF.

    Args:
      folder_path: folder path of tfjs-layers
    """
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".ts"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    file_contents = f.read()
                    matches = re.findall(r"class\s+(\w+)", file_contents)
                    for cls in matches:
                        if cls in TFCLASS_MODULE_MAP:
                            RESULT_MAP[cls] = TFCLASS_MODULE_MAP[cls]

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

def build_map():
  # Build the module Map
  _build_class_module_map(MODULE)
  abs_path = os.path.abspath(__file__)
  root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(abs_path)))))
  path = os.path.join(root_path, 'tfjs-layers')
  _build_ts_class_module_map(path)

def get_module_path(key):
    """Get the module path base on input key

    Args:
      key: The name of the class we want to get module path.
    Return:
      RESULT_MAP[key]: the corresponding module path in TF.
    """
    # if not RESULT_MAP:
    #     raise Exception("Cannot find mapping, please build the map first.")
    if key not in RESULT_MAP:
        raise KeyError(f"Cannot find the module path for {key} class.")
    return RESULT_MAP[key]








