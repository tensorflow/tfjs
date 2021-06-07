import argparse
import os
import tensorflow as tf
import numpy as np
import json
from tensorflowjs.converters import common
common.GRAPH_DEF_FILE_NAME = 'graph_def.pb'

UNCONVERTIBLE_TYPES = (tf.variant,)


def get_output_names(model_dir):
  graph_def_path = os.path.join(model_dir, common.GRAPH_DEF_FILE_NAME)

  with tf.compat.v1.io.gfile.GFile(graph_def_path, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

  with tf.compat.v1.Session() as sess:
    return [output.name for op in sess.graph.get_operations()
                        for output in op.outputs
                        if output.dtype not in UNCONVERTIBLE_TYPES]


def get_py_values(model_dir, input_values):
  model_path = os.path.join(model_dir, common.ARTIFACT_MODEL_JSON_FILE_NAME)

  with tf.compat.v1.io.gfile.GFile(model_path, 'rb') as f:
    signature = json.load(f)[common.SIGNATURE_KEY]

  input_names = [input['name'] for input in signature['inputs'].values()]
  input_names.sort()
  output_names = get_output_names(model_dir)

  if len(input_names) != len(input_values):
    raise ValueError('Incorrect number of input values to graph, inputs are:\n'+
                     ', '.join(input_names))

  input_values = map(json.loads, input_values)
  feed_dict = dict(zip(input_names, input_values))

  with tf.compat.v1.Session() as sess:
    py_values = sess.run(output_names, feed_dict=feed_dict)
    py_values = map(np.array, py_values)
    py_values = zip(output_names, py_values)
    return py_values


def get_tfjs_values(tfjs_json_path):
  with tf.compat.v1.io.gfile.GFile(tfjs_json_path, 'rb') as f:
    tfjs_values = json.load(f)

  for node_name, node_value in tfjs_values.items():
    if node_value is not None:
      array, shape = node_value
      array = np.array(array).reshape(shape)

      if np.issubdtype(array.dtype, np.unicode) or\
         np.issubdtype(array.dtype, np.string_):
        array = array.astype('bytes')

      tfjs_values[node_name] = array

  return tfjs_values

def compare_values(py_values, tfjs_values):
  def arrays_similar(py_value, tfjs_value):
    if py_value.shape != tfjs_value.shape:
      return False
    if np.array_equal(py_value.astype('bytes'), tfjs_value.astype('bytes')):
      return True
    try:
      if np.allclose(py_value, tfjs_value):
        return True
    except:
      pass

    return False

  differences = []

  for output_name, py_value in py_values:
    if output_name not in tfjs_values:
      print("TFJS graph does not contain the python node: ", output_name)
    else:
      tfjs_value = tfjs_values[output_name]
      if tfjs_value is None:
        print("TFJS had an error executing the node: ", output_name)
      else:
        if not arrays_similar(py_value, tfjs_value):
          print("Difference in node values between python and js at node: ", output_name)
          print("Python value: ", py_value)
          print("TFJS value: ", tfjs_value)
          differences.append((output_name, py_value, tfjs_value))

  return str(differences)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--model_dir", type=str, required=True, help="Path where TFJS converted model is found")
  parser.add_argument(
    "--tfjs_json_path", type=str, help="File where TFJS output JSON is found")
  parser.add_argument(
    "--input_values", nargs='*', help="Inputs to the graph as JSON values (given in sorted order of input name)")
  parser.add_argument(
      "--get_output_nodes",
      action='store_true',
      help="Generate list of output nodes")
  parser.add_argument(
    "--output_path",
    type=str,
    required=True,
    help="File to store output nodes or model differences")
  args = parser.parse_args()

  if args.get_output_nodes:
    output_names = get_output_names(args.model_dir)
    with tf.compat.v1.io.gfile.GFile(args.output_path, 'w') as f:
      f.write('\n'.join(output_names))
    return

  py_values = get_py_values(args.model_dir, args.input_values)
  tfjs_values = get_tfjs_values(args.tfjs_json_path)
  differences = compare_values(py_values, tfjs_values)

  with tf.compat.v1.io.gfile.GFile(args.output_path, 'w') as f:
    f.write(differences)


if __name__ == "__main__":
  main()
