import glob
# import rt
# import keras
import tensorflow.keras as keras
from tensorflow.python.util import tf_export
import pkgutil
import inspect
import os
import re

MAP = {}
RESULT_MAP = {}
module = keras

def find_ts_class(folder_path):
    class_names = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".ts"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    file_contents = f.read()
                    matches = re.findall(r"class\s+(\w+)", file_contents)
                    for cls in matches:
                        if cls in MAP:
                            RESULT_MAP[cls] = MAP[cls]
                    class_names.extend(matches)
    return class_names

def build_class_module_map(module):
    classes = []
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            parts = str(tf_export.get_canonical_name_for_symbol(obj, api_name='keras')).split(".")
            MAP[name] = ".".join(parts[:-1])
            classes.append(name)
        elif inspect.ismodule(obj):
            classes.extend(build_class_module_map(obj))
    return classes


classes = build_class_module_map(module)
abs_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(os.path.dirname(abs_path)))
path = os.path.join(root_path, 'tfjs-layers')
c = find_ts_class(path)

for key, value in RESULT_MAP.items():
    print('Key: ', key, ' Value: ', value)




