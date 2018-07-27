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
"""Convert Tensorflow SavedModel to TensorFlow.js web format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import numpy as np

import tensorflow as tf
from tensorflow.core.protobuf import device_properties_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import graph_util
from tensorflow.python.grappler import cluster as gcluster
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.lib.io import file_io
from tensorflow.python.tools import freeze_graph

import tensorflow_hub as hub

from tensorflowjs import write_weights

DEFAULT_MODEL_PB_FILENAME = 'tensorflowjs_model.pb'


def get_cluster():
  """Grappler optimization configuration for GPU."""
  named_device = device_properties_pb2.NamedDevice()
  named_device.name = '/GPU:0'
  named_device.properties.type = 'GPU'
  named_device.properties.environment['architecture'] = '4'
  cluster = gcluster.Cluster(devices=[named_device])
  return cluster


def load_graph(graph_filename, output_node_names):
  """Loads GraphDef. Returns Python Graph object.

  Args:
    graph_filename: string File name for the frozen graph.
  """
  with tf.gfile.Open(graph_filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  with tf.Graph().as_default() as graph:
    # Set name to empty to avoid using the default name 'import'.
    tf.import_graph_def(graph_def, name='')

  for node in output_node_names.split(','):
    graph.add_to_collection('train_op',
                            graph.get_operation_by_name(node.strip()))

  return graph


def validate(nodes, skip_op_check, strip_debug_ops):
  """Validate if the node's op is compatible with TensorFlow.js.

  Args:
    nodes: tf.NodeDef TensorFlow NodeDef objects from GraphDef.
    skip_op_check: Bool whether to skip the op check.
    strip_debug_ops: Bool whether to allow unsupported debug ops.
  """
  if skip_op_check:
    return set()
  ops = []
  op_list_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), '../op_list/')
  for filename in os.listdir(op_list_path):
    if os.path.splitext(filename)[1] == '.json':
      with open(os.path.join(op_list_path, filename)) as json_data:
        ops += json.load(json_data)

  names = set([x['tfOpName'] for x in ops])
  if strip_debug_ops:
    names = names.union(set(['Assert', 'CheckNumerics', 'Print']))
  not_supported = set(
      [x.op for x in [x for x in nodes if x.op not in names]])
  return not_supported


def optimize_graph(graph,
                   output_graph,
                   quantization_dtype=None,
                   skip_op_check=False,
                   strip_debug_ops=False):
  """Takes a Python Graph object and optimizes the graph.

  Args:
    graph: tf.Graph TensorFlow dataflow graph.
    strip_debug_ops: Bool whether to strip out debug ops.
  """
  unsupported = validate(graph.as_graph_def().node, skip_op_check,
                         strip_debug_ops)
  if unsupported:
    raise ValueError('Unsupported Ops in the model before optimization\n' +
                     ', '.join(unsupported))

  rewriter_config = rewriter_config_pb2.RewriterConfig()
  rewriter_config.optimizers[:] = [
      'pruning', 'constfold', 'arithmetic', 'dependency', 'pruning',
      'constfold', 'arithmetic', 'dependency'
  ]
  if strip_debug_ops:
    rewriter_config.optimizers.insert(0, 'debug_stripper')
  meta_graph = tf.train.export_meta_graph(
      graph_def=graph.as_graph_def(), graph=graph)
  optimized_graph = tf_optimizer.OptimizeGraph(
      rewriter_config, meta_graph, cluster=get_cluster())

  unsupported = validate(optimized_graph.node, skip_op_check,
                         strip_debug_ops)

  if unsupported:
    raise ValueError('Unsupported Ops in the model after optimization\n' +
                     ', '.join(unsupported))

  extract_weights(optimized_graph, output_graph, quantization_dtype)
  return optimize_graph


def extract_weights(graph_def,
                    output_graph,
                    quantization_dtype=None):
  """Takes a Python GraphDef object and extract the weights.

  Args:
    graph_def: tf.GraphDef TensorFlow GraphDef proto object, which represents
      the model topology.
    quantization_dtype: An optional numpy dtype to quantize weights to for
        compression. Only np.uint8 and np.uint16 are supported.
  """
  constants = [node for node in graph_def.node if node.op == 'Const']
  constInputs = {}
  # removed the conditional inputs for constants
  for const in constants:
    constInputs[const.name] = const.input[:]
    del const.input[:]

  print('Writing weight file ' + output_graph + '...')
  const_manifest = []
  path = os.path.dirname(output_graph)

  graph = tf.Graph()
  with tf.Session(graph=graph) as sess:
    tf.import_graph_def(graph_def, name='')
    for const in constants:
      tensor = graph.get_tensor_by_name(const.name + ':0')
      value = tensor.eval(session=sess)
      if not isinstance(value, np.ndarray):
        value = np.array(value)

      # Restore the conditional inputs
      const_manifest.append({'name': const.name, 'data': value})
      const.input[:] = constInputs[const.name]

      # Remove the binary array from tensor and save it to the external file.
      const.attr["value"].tensor.ClearField('tensor_content')

  write_weights.write_weights(
      [const_manifest], path, quantization_dtype=quantization_dtype)

  file_io.atomic_write_string_to_file(
      os.path.abspath(output_graph), graph_def.SerializeToString())


def convert_tf_session_bundle(session_bundle_dir,
                              output_node_names,
                              output_dir,
                              quantization_dtype=None,
                              skip_op_check=False,
                              strip_debug_ops=False):
  """Freeze the Session Bundle model and check the model compatibility with
  Tensorflow.js.

  Optimize and convert the model to Tensorflow.js format, when the model passes
  the compatiblity check.

  Args:
    session_bundle_dir: string The session bundle model directory.
    output_node_names: string The names of the output nodes, comma separated.
    output_dir: string The name of the output directory. The directory
      will consist of
      - a file named 'tensorflowjs_model.pb'
      - a JSON weights manifest file named 'weights_manifest.json'
      - possibly sharded binary weight files.
    quantization_dtype: An optional numpy dtype to quantize weights to for
      compression. Only np.uint8 and np.uint16 are supported.
    skip_op_check: Bool whether to skip the op check.
    strip_debug_ops: Bool whether to strip debug ops.
  """

  print("Tensorflow has deprecated the Session Bundle format, ",
        "please migrate to SavedModel.")
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  output_graph = os.path.join(output_dir, DEFAULT_MODEL_PB_FILENAME)

  checkpoint = tf.train.get_checkpoint_state(session_bundle_dir)
  input_checkpoint = checkpoint.model_checkpoint_path
  frozen_file = output_graph + '.frozen'
  freeze_graph.freeze_graph(
      '',
      '',
      True,
      input_checkpoint,
      output_node_names,
      '',
      '',
      frozen_file,
      True,
      '',
      input_meta_graph=input_checkpoint + '.meta')
  graph = load_graph(output_graph + '.frozen', output_node_names)

  try:
    optimize_graph(graph, output_graph, quantization_dtype=quantization_dtype,
                   skip_op_check=skip_op_check, strip_debug_ops=strip_debug_ops)
  finally:
    # Clean up the temp files.
    if os.path.exists(frozen_file):
      os.remove(frozen_file)


def convert_tf_saved_model(saved_model_dir, output_node_names,
                           output_dir, saved_model_tags='serve',
                           quantization_dtype=None,
                           skip_op_check=False,
                           strip_debug_ops=False):
  """Freeze the SavedModel and check the model compatibility with Tensorflow.js.

  Optimize and convert the model to Tensorflow.js format, when the model passes
  the compatiblity check.

  Args:
    saved_model_dir: string The saved model directory.
    output_node_names: string The names of the output nodes, comma separated.
    output_dir: string The name of the output directory. The directory
      will consist of
      - a file named 'tensorflowjs_model.pb'
      - a JSON weights manifest file named 'weights_manifest.json'
      - possibly sharded binary weight files.
    saved_model_tags: string Tagset of the MetaGraphDef to load, in comma
      separated string format. Defaulted to 'serve'
    quantization_dtype: An optional numpy dtype to quantize weights to for
      compression. Only np.uint8 and np.uint16 are supported.
    skip_op_check: Bool whether to skip the op check.
    strip_debug_ops: Bool whether to strip debug ops.
  """

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  output_graph = os.path.join(output_dir, DEFAULT_MODEL_PB_FILENAME)

  frozen_file = output_graph + '.frozen'
  freeze_graph.freeze_graph(
      '',
      '',
      True,
      '',
      output_node_names,
      '',
      '',
      frozen_file,
      True,
      '',
      saved_model_tags=saved_model_tags,
      input_saved_model_dir=saved_model_dir)

  graph = load_graph(output_graph + '.frozen', output_node_names)
  try:
    optimize_graph(graph, output_graph, quantization_dtype=quantization_dtype,
                   skip_op_check=skip_op_check, strip_debug_ops=strip_debug_ops)
  finally:
    # Clean up the temp files.
    if os.path.exists(frozen_file):
      os.remove(frozen_file)


def convert_tf_frozen_model(frozen_model_path, output_node_names,
                            output_dir, quantization_dtype=None,
                            skip_op_check=False,
                            strip_debug_ops=False):
  """Convert frozen model and check the model compatibility with Tensorflow.js.

  Optimize and convert the model to Tensorflow.js format, when the model passes
  the compatiblity check.

  Args:
    frozen_model_path: string The path to frozen model.
    output_node_names: string The names of the output nodes, comma separated.
    output_dir: string The name of the output directory. The directory
      will consist of
      - a file named 'tensorflowjs_model.pb'
      - a JSON weights manifest file named 'weights_manifest.json'
      - possibly sharded binary weight files.
    quantization_dtype: An optional numpy dtype to quantize weights to for
      compression. Only np.uint8 and np.uint16 are supported.
    skip_op_check: Bool whether to skip the op check.
    strip_debug_ops: Bool whether to strip debug ops.
  """

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  output_graph = os.path.join(output_dir, DEFAULT_MODEL_PB_FILENAME)

  graph = load_graph(frozen_model_path, output_node_names)
  optimize_graph(graph, output_graph, quantization_dtype=quantization_dtype,
                 skip_op_check=skip_op_check, strip_debug_ops=strip_debug_ops)


def load_and_initialize_hub_module(module_path, signature='default'):
  """Loads graph of a TF-Hub module and initializes it into a session.

  Args:
    module_path: string Path to TF-Hub module.
    signature: string Signature to use when creating the apply graph.

  Return:
    graph: tf.Graph Graph of the module.
    session: tf.Session Session with initialized variables and tables.
    inputs: dict Dictionary of input tensors.
    outputs: dict Dictionary of output tensors.

  Raises:
    ValueError: If signature contains a SparseTensor on input or output.
  """
  graph = tf.Graph()
  with graph.as_default():
    tf.logging.info('Importing %s', module_path)
    module = hub.Module(module_path)

    signature_inputs = module.get_input_info_dict(signature)
    signature_outputs = module.get_output_info_dict(signature)
    # First check there are no SparseTensors in input or output.
    for key, info in list(signature_inputs.items()) + list(
        signature_outputs.items()):
      if info.is_sparse:
        raise ValueError(
            'Signature "%s" has a SparseTensor on input/output "%s".'
            ' SparseTensors are not supported.' % (signature, key))

    # Create placeholders to represent the input of the provided signature.
    inputs = {}
    for input_key, input_info in signature_inputs.items():
      inputs[input_key] = tf.placeholder(
          shape=input_info.get_shape(), dtype=input_info.dtype, name=input_key)

    outputs = module(inputs=inputs, signature=signature, as_dict=True)

    session = tf.Session(graph=graph)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())

  return graph, session, inputs, outputs


def convert_tf_hub_module(module_path, output_dir,
                          signature='default', quantization_dtype=None,
                          skip_op_check=False, strip_debug_ops=False):
  """Freeze the TF-Hub module and check compatibility with Tensorflow.js.

  Optimize and convert the TF-Hub module to Tensorflow.js format, if it passes
  the compatiblity check.

  Args:
    module_path: string Path to the module.
    output_dir: string The name of the output directory. The directory
      will consist of
      - a file named 'tensorflowjs_model.pb'
      - a JSON weights manifest file named 'weights_manifest.json'
      - possibly sharded binary weight files.
    signature: string Signature to load.
    skip_op_check: Bool whether to skip the op check.
    strip_debug_ops: Bool whether to strip debug ops.
  """

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  graph, sess, inputs, outputs = load_and_initialize_hub_module(
      module_path, signature)

  input_node_names = []
  output_node_names = []

  for _, input_tensor in inputs.items():
    input_node_names.append(input_tensor.name.split(':')[0])
  for _, output_tensor in outputs.items():
    output_node_names.append(output_tensor.name.split(':')[0])

  print('Creating a model with inputs %s and outputs %s.' % (input_node_names,
                                                             output_node_names))

  frozen_graph_def = graph_util.convert_variables_to_constants(
      sess, graph.as_graph_def(), output_node_names)

  output_graph = os.path.join(output_dir, DEFAULT_MODEL_PB_FILENAME)
  frozen_file = output_graph + '.frozen'
  try:
    with tf.gfile.GFile(frozen_file, 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())

    graph = load_graph(frozen_file, ','.join(output_node_names))
    optimize_graph(graph, output_graph, quantization_dtype=quantization_dtype,
                   skip_op_check=skip_op_check, strip_debug_ops=strip_debug_ops)
  finally:
    # Clean up the temp files.
    if os.path.exists(frozen_file):
      os.remove(frozen_file)
