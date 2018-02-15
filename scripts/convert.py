"""Run Grappler optimizers in the standalone mode.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys
import os

from google.protobuf import text_format
from absl import flags
from tensorflow.python.tools import freeze_graph
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf import device_properties_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.grappler import cluster as gcluster
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.lib.io import file_io

flags.DEFINE_string('saved_model_dir', '', 'The saved model directory.')
flags.DEFINE_string('output_node_names', '',
                    'The names of the output nodes, comma separated.')
flags.DEFINE_string('output_graph', '', 'The name of the output graph file')
flags.DEFINE_string(
    'saved_model_tags', 'serve',
    'Tags of the MetaGraphDef to load, in comma separated string format.'
)

FLAGS = flags.FLAGS

def get_cluster():
  named_device = device_properties_pb2.NamedDevice()
  named_device.name = '/GPU:0'
  named_device.properties.type = 'GPU'
  named_device.properties.environment['architecture'] = '4'
  cluster = gcluster.Cluster(devices=[named_device])
  return cluster


def load_graph(graph_filename):
  """Loads GraphDef. Returns Python Graph object."""
  with tf.gfile.Open(graph_filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  with tf.Graph().as_default() as graph:
    # Set name to empty to avoid using the default name 'import'.
    tf.import_graph_def(graph_def, name='')

  for node in FLAGS.output_node_names.split(','):
    graph.add_to_collection('train_op',graph.get_operation_by_name(node.strip()))

  return graph


def optimize_graph(graph):
  """Takes a Python Graph object and optimizes the graph."""
  rewriter_config = rewriter_config_pb2.RewriterConfig()
  rewriter_config.optimizers[:] = ['pruning', 'constfold', 'arithmetic',
                                'dependency', 'pruning',
                                'constfold', 'arithmetic','dependency']
  meta_graph = tf.train.export_meta_graph(
      graph_def=graph.as_graph_def(), graph=graph)
  optimized_graph = tf_optimizer.OptimizeGraph(
      rewriter_config, meta_graph, cluster=get_cluster())

  extract_weights(graph, optimized_graph)
  return optimize_graph

def extract_weights(graph, graph_def):
  """Takes a Python GraphDef object and extract the weights."""
  constants = [node for node in graph_def.node if node.op == 'Const']
  print('Writing weight file ' + FLAGS.output_graph + '...')
  index = 0
  with open(os.path.abspath(FLAGS.output_graph + '.weight'), 'wb') as f:
    with tf.Session(graph=graph) as sess:
      for const in constants:
        tensor = graph.get_tensor_by_name(const.name + ':0')
        """save the value of the tensor to the external file"""
        f.write(tensor.eval(session=sess).tobytes())

        """store the index and length of the tensor in the external file"""
        byte_length = tf.size(tensor).eval()
        const.attr["index"].CopyFrom(attr_value_pb2.AttrValue(i=index))
        const.attr["length"].CopyFrom(attr_value_pb2.AttrValue(i=byte_length))

        """Remove the binary array from tensor and save it to the external file."""
        const.attr["value"].tensor.ClearField('tensor_content')

        index += byte_length * tensor.dtype.size

  file_io.atomic_write_string_to_file(
    os.path.abspath(FLAGS.output_graph), graph_def.SerializeToString())

  file_io.atomic_write_string_to_file(
    os.path.abspath(FLAGS.output_graph+'txt'), text_format.MessageToString(graph_def))

def main(_):

# Freeze the graph
  freeze_graph.freeze_graph('', '', True, '',
                          FLAGS.output_node_names,
                          '', '',
                          FLAGS.output_graph + '.frozen', True, '',
                          saved_model_tags=FLAGS.saved_model_tags,
                          input_saved_model_dir=FLAGS.saved_model_dir)
  graph = load_graph(FLAGS.output_graph + '.frozen')
  optimize_graph(graph)


if __name__ == '__main__':
  FLAGS(sys.argv)
  tf.app.run(main)
