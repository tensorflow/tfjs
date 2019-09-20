# Copyright 2019 Google LLC
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
"""Resource management library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


def open_file(path):
  """Opens the file at given path, where path is relative to tensorflowjs/.

  Args:
    path: a string resource path relative to tensorflowjs/.

  Returns:
    An open file of that resource.

  Raises:
    IOError: If the path is not found, or the resource can't be opened.
  """
  path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
  return open(path)


def list_dir(path):
  """List the files inside a dir where path is relative to tensorflowjs/.

  Args:
    path: a string path to a resource directory relative to tensorflowjs/.

  Returns:
    A list of files inside that directory.

  Raises:
    IOError: If the path is not found, or the resource can't be read.
  """
  path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
  return os.listdir(path)
