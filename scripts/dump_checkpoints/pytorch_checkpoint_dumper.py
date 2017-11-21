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

"""This script defines PytorchCheckpointDumper class.

This class takes a pytorch checkpoint file and writes all of the variables in the
checkpoint to a directory which deeplearnjs can take as input.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six import iteritems

import argparse
import os
import re
import json
import string

import numpy as np

import torch

from checkpoint_dumper import CheckpointDumper

class PytorchCheckpointDumper(CheckpointDumper):

  """Class for dumping Pytorch Checkpoints.

  Attributes
  ----------
  state_dictionary : dict
      Dictionary defining checkpoint variables and weights
  """

  def __init__(self, checkpoint_file, output_dir, remove_variables_regex):
    """Constructs object for Pytorch Checkpoint Dumper.

    Parameters
    ----------
    checkpoint_file : str
        Path to the model checkpoint
    output_dir : str
        Output directory path
    remove_variables_regex : str
        Regex expression for variables to be ignored
    """
    super(PytorchCheckpointDumper, self).__init__(
      checkpoint_file, output_dir, remove_variables_regex)

    self.state_dictionary = torch.load(self.checkpoint_file)

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
      elif c == '.':
        chars.append('_')

    return ''.join(chars)

  def build_and_dump_vars(self):
    """Builds and dumps variables and a manifest file.
    """
    for (var_name, var_weights) in iteritems(self.state_dictionary):
      if (self.should_ignore(var_name)):
        print('Ignoring ' + var_name)
        continue

      var_filename = self.var_name_to_filename(var_name)
      var_shape = list(map(int, list(var_weights.size())))
      tensor = var_weights.cpu().numpy()

      self.dump_weights(var_name, var_filename, var_shape, tensor)

    self.dump_manifest()
