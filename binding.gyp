##
# @license
# Copyright 2018 Google Inc. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# Node.js TensorFlow Binding config:
{
  'variables' : {
    'tensorflow_include_dir' : '<(module_root_dir)/deps/include',
    'tensorflow_headers' : [
      '<@(tensorflow_include_dir)/tensorflow/c/c_api.h',
      '<@(tensorflow_include_dir)/tensorflow/c/eager/c_api.h',
    ],
  },
  'targets' : [{
    'target_name' : 'tfjs_binding',
    'sources' : [
      'binding/tfe_utils.cc',
      'binding/tfjs_backend.cc',
      'binding/tfjs_binding.cc'
    ],
    'include_dirs' : [ '..', '<(tensorflow_include_dir)' ],
    'conditions' : [
      [
        'OS=="linux"', {
          'libraries' : [
            '-Wl,-rpath,\$$ORIGIN',
            '-ltensorflow',
          ],
          'library_dirs' : ['<(PRODUCT_DIR)'],
          'variables': {
            'tensorflow-library-target': 'linux-cpu'
          }
        }
      ],
      [
        'OS=="mac"', {
          'libraries' : [
            '-Wl,-rpath,@loader_path',
            '-ltensorflow',
          ],
          'library_dirs' : ['<(PRODUCT_DIR)'],
          'variables': {
            'tensorflow-library-target': 'darwin'
          }
        }
      ],
    ],
    'actions': [
      {
        'action_name': 'get_libtensorflow',
        'inputs': [
          '<(module_root_dir)/scripts/get_libtensorflow.js'
        ],
        'outputs': [
          '<(PRODUCT_DIR)/libtensorflow.so',
        ],
        'action': [
          'node',
          '<@(_inputs)',
          '<(tensorflow-library-target)',
          '<(PRODUCT_DIR)',
        ]
      }
    ],
  }]
}
