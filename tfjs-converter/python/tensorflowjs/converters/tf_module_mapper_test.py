# Copyright 2018 Google LLC
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
"""Unit tests for build_module_map."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflowjs.converters.tf_module_mapper as bm
import tensorflow.compat.v2 as tf

class TfModuleMapperTest(tf.test.TestCase):
  def setUp(self):
    bm.build_map()
    super(TfModuleMapperTest, self).setUp()

  def tearDown(self):
    super(TfModuleMapperTest, self).tearDown()

  def testUnsupportClassInMap(self):
    non_exist_class_name = 'FakeClass'

    with self.assertRaises(KeyError):
      bm.get_module_path(non_exist_class_name)

  def testDenseClassInMap(self):
    class_name = 'Dense'

    self.assertEqual('keras.layers', bm.get_module_path(class_name))

if __name__ == '__main__':
  tf.test.main()
