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

import os
import json
import numpy as np

import tensorflow as tf
from tensorflow.core.protobuf import device_properties_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.grappler import cluster as gcluster
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.lib.io import file_io
from tensorflow.python.tools import freeze_graph

from tensorflowjs import write_weights

DEFAULT_MODEL_PB_FILENAME = 'tensorflowjs_model.pb'


def get_cluster():
  """ Grappler optimization configuration for GPU."""
  named_device = device_properties_pb2.NamedDevice()
  named_device.name = '/GPU:0'
  named_device.properties.type = 'GPU'
  named_device.properties.environment['architecture'] = '4'
  cluster = gcluster.Cluster(devices=[named_device])
  return cluster


def load_graph(graph_filename, output_node_names):
  """Loads GraphDef. Returns Python Graph object.

  Args:
    graph_filename: string file name for the frozen graph
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


def validate(nodes):
  """Validate if the node's op is compatible with TensorFlow.js.

  Args:
    nodes: tf.NodeDef tensorflow NodeDef objects from GraphDef
  """
  ops = []
  op_list_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), '../op_list/')
  for filename in os.listdir(op_list_path):
    if os.path.splitext(filename)[1] == '.json':
      with open(os.path.join(op_list_path, filename)) as json_data:
        ops += json.load(json_data)

  names = set([x['tfOpName'] for x in ops])
  not_supported = set(
      [x.op for x in [x for x in nodes if x.op not in names]])
  return not_supported


def optimize_graph(graph, output_graph):
  """Takes a Python Graph object and optimizes the graph.

  Args:
    graph: tf.Graph tensorflow dataflow graph
  """
  rewriter_config = rewriter_config_pb2.RewriterConfig()
  rewriter_config.optimizers[:] = [
      'pruning', 'constfold', 'arithmetic', 'dependency', 'pruning',
      'constfold', 'arithmetic', 'dependency'
  ]
  meta_graph = tf.train.export_meta_graph(
      graph_def=graph.as_graph_def(), graph=graph)
  optimized_graph = tf_optimizer.OptimizeGraph(
      rewriter_config, meta_graph, cluster=get_cluster())

  extract_weights(optimized_graph, output_graph)
  return optimize_graph


def extract_weights(graph_def, output_graph):
  """Takes a Python GraphDef object and extract the weights.

  Args:
    graph_def: tf.GraphDef tensorflow GraphDef proto object, which represents
      the model topology
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

  write_weights.write_weights([const_manifest], path)

  file_io.atomic_write_string_to_file(
      os.path.abspath(output_graph), graph_def.SerializeToString())


def convert_tf_session_bundle(session_bundle_dir,
                              output_node_names,
                              output_dir):
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
  unsupported = validate(graph.as_graph_def().node)
  if unsupported:
    print('Unsupported Ops in the model\n' + ', '.join(unsupported))
  else:
    optimize_graph(graph, output_graph)

  # Clean up the temp files.
  if os.path.exists(frozen_file):
    os.remove(frozen_file)


def convert_tf_saved_model(saved_model_dir, output_node_names,
                           output_dir, saved_model_tags='serve'):
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
  unsupported = validate(graph.as_graph_def().node)
  if unsupported:
    print('Unsupported Ops in the model\n' + ', '.join(unsupported))
  else:
    optimize_graph(graph, output_graph)

  # Clean up the temp files.
  if os.path.exists(frozen_file):
    os.remove(frozen_file)
