"""Run Grappler optimizers in the standalone mode.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys
import os

from absl import flags
from tensorflow.python.tools import freeze_graph
from tensorflow.core.protobuf import device_properties_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.grappler import cluster as gcluster
from tensorflow.python.grappler import tf_optimizer

flags.DEFINE_string('saved_model_dir', '', 'The saved model directory.')
flags.DEFINE_string('output_node_names', '',
                    'The names of the output nodes, comma separated.')
flags.DEFINE_string('output_graph', '', 'The name of the output graph file')
flags.DEFINE_string('input_checkpoint', '',
                    'TensorFlow variables file to load.')
flags.DEFINE_boolean("input_binary", True,
                    "Whether the input files are in binary format.")
flags.DEFINE_string('input_saver', '', 'TensorFlow saver file to load.')
flags.DEFINE_string('input_graph', '', 'TensorFlow GraphDef file to load.')
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

  if FLAGS.output_graph:
    head, tail = os.path.split(FLAGS.output_graph)
    tf.train.write_graph(
        optimized_graph, head, tail, as_text=False)


def main(_):

# Freeze the graph
  freeze_graph.freeze_graph(FLAGS.input_graph, FLAGS.input_saver,
                          FLAGS.input_binary, FLAGS.input_checkpoint,
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
