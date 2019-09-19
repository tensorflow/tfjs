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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import shutil
import unittest

import tempfile

import numpy as np

from tensorflowjs import resource_loader

class ResourceLoaderTest(unittest.TestCase):

  def testListingFilesInOpList(self):
    files = resource_loader.list_dir('op_list')
    self.assertTrue(len(files) > 0)
    for file in files:
      self.assertTrue(file.endswith('.json'))

  def testReadingFileInOpList(self):
    with resource_loader.open_file('op_list/arithmetic.json') as f:
      data = json.load(f)
      first_op = data[0]
      self.assertTrue('tfOpName' in first_op)
      self.assertTrue('category' in first_op)

if __name__ == '__main__':
  unittest.main()
