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
    'tensorflow-library-action': 'symlink'
  },
  'targets' : [{
    'target_name' : 'tfjs_binding',
    'sources' : [
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
            '-ltensorflow_framework',
          ],
          'library_dirs' : ['<(PRODUCT_DIR)'],
          'actions': [
            {
              'action_name': 'deps-stage',
              'inputs': [
                '<(module_root_dir)/scripts/deps-stage.js'
              ],
              'outputs': [
                '<(PRODUCT_DIR)/libtensorflow.so',
              ],
              'action': [
                'node',
                '<@(_inputs)',
                '<@(tensorflow-library-action)',
                '<(PRODUCT_DIR)'
              ]
            }
          ],
        }
      ],
      [
        'OS=="mac"', {
          'libraries' : [
            '-Wl,-rpath,@loader_path',
            '-ltensorflow',
          ],
          'library_dirs' : ['<(PRODUCT_DIR)'],
          'actions': [
            {
              'action_name': 'deps-stage',
              'inputs': [
                '<(module_root_dir)/scripts/deps-stage.js'
              ],
              'outputs': [
                '<(PRODUCT_DIR)/libtensorflow.so',
              ],
              'action': [
                'node',
                '<@(_inputs)',
                '<@(tensorflow-library-action)',
                '<(PRODUCT_DIR)'
              ]
            }
          ],
        }
      ],
      [
        'OS=="win"', {
          'defines': ['COMPILER_MSVC'],
          'libraries': ['tensorflow'],
          'library_dirs' : ['<(INTERMEDIATE_DIR)'],
          'variables': {
            'tensorflow-library-target': 'windows'
          },
          'msvs_disabled_warnings': [
            # Warning	C4190: 'TF_NewWhile' has C-linkage specified, but returns
            # UDT 'TF_WhileParams' which is incompatible with C.
            # (in include/tensorflow/c/c_api.h)
            4190
          ],
          'actions': [
            {
              'action_name': 'deps-stage',
              'inputs': [
                '<(module_root_dir)/scripts/deps-stage.js'
              ],
              'outputs': [
                '<(PRODUCT_DIR)/tensorflow.dll',
              ],
              'action': [
                'node',
                '<@(_inputs)',
                '<@(tensorflow-library-action)',
                '<(PRODUCT_DIR)'
              ]
            },
            {
              'action_name': 'generate_def',
              'inputs': [
                '<(module_root_dir)/scripts/generate_defs.js',
                '<@(tensorflow_headers)',
                "<(PRODUCT_DIR)/tensorflow.dll"
              ],
              'outputs': [
                '<(INTERMEDIATE_DIR)/tensorflow.def'
              ],
              'action': [
                'cmd',
                '/c node <@(_inputs) > <@(_outputs)'
              ]
            },
            {
              'action_name': 'build-tensorflow-lib',
              'inputs': [
                '<(INTERMEDIATE_DIR)/tensorflow.def'
              ],
              'outputs': [
                '<(INTERMEDIATE_DIR)/tensorflow.lib'
              ],
              'action': [
                'lib',
                '/def:<@(_inputs)',
                '/out:<@(_outputs)',
                '/machine:<@(target_arch)'
              ]
            },
          ],
        },
      ]
    ],
  }]
}
