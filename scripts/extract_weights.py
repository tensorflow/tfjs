"""Run Grappler optimizers in the standalone mode.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys
import os
from tensorflow.core.framework import attr_value_pb2
from absl import flags

flags.DEFINE_string('input_frozen_graph', '', 'TensorFlow GraphDef file to load.')
flags.DEFINE_string('output_graph', '', 'The name of the output graph file')
FLAGS = flags.FLAGS

def load_graph(graph_filename):
  """Loads GraphDef. Returns Python Graph object."""
  with tf.gfile.Open(graph_filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  with tf.Graph().as_default() as graph:
    # Set name to empty to avoid using the default name 'import'.
    tf.import_graph_def(graph_def, name='')

  return graph


def extract_weights(graph):
  """Takes a Python Graph object and extract the weights."""
  graph_def = graph.as_graph_def()
  constants = [node for node in graph_def.node if node.op == 'Const']
  print('Writing weight file ' + FLAGS.output_graph + '...')
  print(constants[0])
  index = 0
  with open(os.path.abspath(FLAGS.output_graph), 'wb') as f:
    with tf.Session(graph=graph) as sess:
      for const in constants:
        tensor = graph.get_tensor_by_name(const.name + ':0')
        f.write(tensor.eval(session=sess).tobytes())
        const.attr["index"].CopyFrom(attr_value_pb2.AttrValue(i=index))
        del const.attr["value"]
        index += tf.size(tensor).eval()

  print(constants[0])

  constants = [node for node in graph_def.node if node.op == 'Const']
  print(constants[0])
  head, tail = os.path.split(FLAGS.output_graph)

  with open(os.path.abspath(FLAGS.output_graph+'.pb'), 'wb') as f:
    f.write(graph_def.SerializeToString())

def main(_):
  if FLAGS.input_frozen_graph:
    graph = load_graph(FLAGS.input_frozen_graph)
    if FLAGS.output_graph:
      extract_weights(graph)


if __name__ == '__main__':
  FLAGS(sys.argv)
  tf.app.run(main)
