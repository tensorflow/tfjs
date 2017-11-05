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

"""This script defines CheckpointDumper class.

This class serves as a base class for other deeplearning checkpoint dumper
classes and defines common methods, attributes etc.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import re
import string

class CheckpointDumper(object):

  """Base Checkpoint Dumper class.

  Attributes
  ----------
  checkpoint_file : str
      Path to the model checkpoint
  FILENAME_CHARS : str
      Allowed file char names
  manifest : dict
      Manifest file defining variables
  output_dir : str
      Output directory path
  remove_variables_regex : str
      Regex expression for variables to be ignored
  remove_variables_regex_re : sre.SRE_Pattern
      Compiled `remove variable` regex
  """
  
  FILENAME_CHARS = string.ascii_letters + string.digits + '_'

  def __init__(self, checkpoint_file, output_dir, remove_variables_regex):
    """Constructs object for Checkpoint Dumper.

    Parameters
    ----------
    checkpoint_file : str
        Path to the model checkpoint
    output_dir : str
        Output directory path
    remove_variables_regex : str
        Regex expression for variables to be ignored
    """
    self.checkpoint_file = os.path.expanduser(checkpoint_file)
    self.output_dir = os.path.expanduser(output_dir)
    self.remove_variables_regex = remove_variables_regex

    self.manifest = {}
    self.remove_variables_regex_re = re.compile(self.remove_variables_regex)

    self.make_dir(self.output_dir)


  @staticmethod
  def make_dir(directory):
    """Makes directory if not existing.
    
    Parameters
    ----------
    directory : str
        Path to directory
    """
    if not os.path.exists(directory):
      os.makedirs(directory)


  def should_ignore(self, name):
    """Checks whether name should be ignored or not.

    Parameters
    ----------
    name : str
        Name to be checked

    Returns
    -------
    bool
        Whether to ignore the name or not
    """
    return self.remove_variables_regex and re.match(self.remove_variables_regex_re, name)


  def dump_weights(self, variable_name, filename, shape, weights):
    """Creates a file with given name and dumps byte weights in it.

    Parameters
    ----------
    variable_name : str
        Name of given variable
    filename : str
        File name for given variable
    shape : list
        Shape of given variable
    weights : ndarray
        Weights for given variable
    """
    self.manifest[variable_name] = {'filename': filename, 'shape': shape}

    print('Writing variable ' + variable_name + '...')
    with open(os.path.join(self.output_dir, filename), 'wb') as f:
      f.write(weights.tobytes())


  def dump_manifest(self, filename='manifest.json'):
    """Creates a manifest file with given name and dumps meta information
    related to model.

    Parameters
    ----------
    filename : str, optional
        Manifest file name
    """
    manifest_fpath = os.path.join(self.output_dir, filename)

    print('Writing manifest to ' + manifest_fpath)
    with open(manifest_fpath, 'w') as f:
      f.write(json.dumps(self.manifest, indent=2, sort_keys=True))
