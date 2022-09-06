    # Copyright 2020 Google LLC
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
"""Python script for creating Tensorflow SavedModel with UINT8 input."""

import os

import tensorflow as tf
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import variables
from tensorflow.python.training.tracking import tracking
from tensorflow.python.saved_model.save import save

"""Test a basic model with functions to make sure functions are inlined."""
input_data = constant_op.constant(1, shape=[1], dtype=tf.uint8)
root = tracking.AutoTrackable()
root.v1 = variables.Variable(3)
root.v2 = variables.Variable(2)
root.f = def_function.function(lambda x: root.v1 * root.v2 * tf.cast(x, tf.int32))
to_save = root.f.get_concrete_function(input_data)

save_dir = os.path.join('..', 'test_objects', 'saved_model', 'uint8_multiply')
save(root, save_dir, to_save)
