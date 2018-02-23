#Binding config
{
  "variables" : {
    "tensorflow_include_dir" : "<(module_root_dir)/../deps/tensorflow/include",
    "tensorflow_headers" : [
      "<@(tensorflow_include_dir)/tensorflow/c/c_api.h",
      "<@(tensorflow_include_dir)/tensorflow/c/eager/c_api.h",
    ],
    "tensorflow_lib_dir" : "<(module_root_dir)/../deps/tensorflow/lib",
  },
  "targets" : [{
    "target_name" : "tfnodejs",
    "sources" : [
      "tensor_handle.cc",
      "tf_node_binding.cc",
      "tfe_context_env.cc",
      "tfe_execute.cc",
    ],
    "include_dirs" : [ '..', "<(tensorflow_include_dir)" ],
    "conditions" : [
      [
        'OS=="linux"', {
          'libraries' : [
            '-Wl,-rpath,<@(tensorflow_lib_dir)',
            '-ltensorflow',
            '-ltensorflow_framework',
          ],
          'library_dirs' : ['<(tensorflow_lib_dir)'],
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
        }
      ],
    ],
  }]
}
