# Copyright 2017 Google Inc. All Rights Reserved.
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

"""This script defines TensorflowCheckpointDumper class.

This class takes a tensorflow checkpoint file and writes all of the variables in the
checkpoint to a directory which deeplearnjs can take as input.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six import iteritems

import argparse
import json
import os
import re

import tensorflow as tf

from checkpoint_dumper import CheckpointDumper

class TensorflowCheckpointDumper(CheckpointDumper):

  """Class for dumping Tensorflow Checkpoints.

  Attributes
  ----------
  reader : NewCheckpointReader
      Reader for given tensorflow checkpoint
  """

  def __init__(self, checkpoint_file, output_dir, remove_variables_regex):
    """Constructs object for Tensorflow Checkpoint Dumper.

    Parameters
    ----------
    checkpoint_file : str
        Path to the model checkpoint
    output_dir : str
        Output directory path
    remove_variables_regex : str
        Regex expression for variables to be ignored
    """
    super(TensorflowCheckpointDumper, self).__init__(
      checkpoint_file, output_dir, remove_variables_regex)

    self.reader = tf.train.NewCheckpointReader(self.checkpoint_file)

  def var_name_to_filename(self, var_name):
    """Converts variable names to standard file names.

    Parameters
    ----------
    var_name : str
        Variable name to be converted

    Returns
    -------
    str
        Standardized file name
    """
    chars = []

    for c in var_name:
      if c in CheckpointDumper.FILENAME_CHARS:
        chars.append(c)
      elif c == '/':
        chars.append('_')

    return ''.join(chars)

  def build_and_dump_vars(self):
    """Builds and dumps variables and a manifest file.
    """
    var_to_shape_map = self.reader.get_variable_to_shape_map()

    for (var_name, var_shape) in iteritems(var_to_shape_map):
      if self.should_ignore(var_name) or var_name == 'global_step':
        print('Ignoring ' + var_name)
        continue

      var_filename = self.var_name_to_filename(var_name)
      self.manifest[var_name] = {'filename': var_filename, 'shape': var_shape}

      tensor = self.reader.get_tensor(var_name)
      self.dump_weights(var_name, var_filename, var_shape, tensor)

    self.dump_manifest()
