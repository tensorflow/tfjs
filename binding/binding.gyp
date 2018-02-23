#Binding config
{
  'variables' : {
    'tensorflow_include_dir' : '<(module_root_dir)/../deps/tensorflow/include',
    'tensorflow_lib_dir' : '<(module_root_dir)/../deps/tensorflow/lib',
  },
  'targets' : [{
    'target_name' : 'tfnodejs',
    'sources' : [
      'tensor_handle.cc',
      'tf_node_binding.cc',
      'tfe_context_env.cc',
      'tfe_execute.cc',
    ],
    'include_dirs' : [ '..', '<(tensorflow_include_dir)' ],
    'conditions' : [
      [
        'OS=="linux"', {
          'libraries' : [
            '-Wl,-rpath,<@(tensorflow_lib_dir)',
            '-ltensorflow',
            '-ltensorflow_framework',
          ],
          'library_dirs' : ['<(tensorflow_lib_dir)'],
          'variables': {
            'tensorflow-library-target': 'linux-cpu'
          }
        }
      ],
      [
        'OS=="mac"', {
          'libraries' : [
            '-Wl,-rpath,<@(tensorflow_lib_dir)',
            '-ltensorflow',
            '-ltensorflow_framework',
          ],
          'library_dirs' : ['<(tensorflow_lib_dir)'],
          'variables': {
            'tensorflow-library-target': 'darwin'
          }
        }
      ],
    ],
    'actions': [
      {
        'action_name': 'download_libtensorflow',
        'inputs': [
          '<(module_root_dir)/../scripts/download-libtensorflow.sh',
        ],
        'outputs': [
          '<(PRODUCT_DIR)/libtensorflow.so',
        ],
        'action': [
          'sh',
          '<@(_inputs)',
          '<(tensorflow-library-target)',
        ]
      }
    ],
  }]
}
