# Copyright 2019 Google LLC
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
"""Interactive command line tool for tensorflow.js model conversion."""

from __future__ import print_function, unicode_literals

import json
import os
import re
import sys
import tempfile
import traceback

import PyInquirer
import h5py
import tensorflow.compat.v2 as tf
from tensorflow.core.framework import types_pb2
from tensorflow.python.saved_model import loader_impl
from tensorflowjs.converters import converter
from tensorflowjs.converters import common

# regex for recognizing valid url for TFHub module.
TFHUB_VALID_URL_REGEX = re.compile(
    # http:// or https://
    r'^(http)s?://', re.IGNORECASE)

# prompt style
prompt_style = PyInquirer.style_from_dict({
    PyInquirer.Token.Separator: '#6C6C6C',
    PyInquirer.Token.QuestionMark: '#FF9D00 bold',
    PyInquirer.Token.Selected: '#5F819D',
    PyInquirer.Token.Pointer: '#FF9D00 bold',
    PyInquirer.Token.Instruction: '',  # default
    PyInquirer.Token.Answer: '#5F819D bold',
    PyInquirer.Token.Question: '',
})


def value_in_list(answers, key, values):
  """Determine user's answer for the key is in the value list.
  Args:
    answer: Dict of user's answers to the questions.
    key: question key.
    values: List of values to check from.
  """
  try:
    value = answers[key]
    return value in values
  except KeyError:
    return False


def get_tfjs_model_type(model_file):
  with open(model_file) as f:
    data = json.load(f)
    print("====", data)
    if 'format' in data:
      return data['format']
    else: # Default to layers model
      return common.TFJS_LAYERS_MODEL_FORMAT


def detect_saved_model(input_path):
  if os.path.exists(os.path.join(input_path, 'assets', 'saved_model.json')):
    return common.KERAS_SAVED_MODEL
  saved_model = loader_impl.parse_saved_model(input_path)
  graph_def = saved_model.meta_graphs[0].object_graph_def
  if graph_def.nodes:
    if 'tf_keras' in graph_def.nodes[0].user_object.identifier:
      return common.KERAS_SAVED_MODEL
  return common.TF_SAVED_MODEL


def detect_input_format(input_path):
  """Determine the input format from model's input path or file.
  Args:
    input_path: string of the input model path
  returns:
    string: detected input format
    string: normalized input path
  """
  input_path = input_path.strip()
  detected_input_format = None
  if re.match(TFHUB_VALID_URL_REGEX, input_path):
    detected_input_format = common.TF_HUB_MODEL
  elif os.path.isdir(input_path):
    if (any(fname.lower().endswith('saved_model.pb')
            for fname in os.listdir(input_path))):
      detected_input_format = detect_saved_model(input_path)
    else:
      for fname in os.listdir(input_path):
        fname = fname.lower()
        if fname.endswith('model.json'):
          filename = os.path.join(input_path, fname)
          if get_tfjs_model_type(filename) == common.TFJS_LAYERS_MODEL_FORMAT:
            input_path = os.path.join(input_path, fname)
            detected_input_format = common.TFJS_LAYERS_MODEL
            break
  elif os.path.isfile(input_path):
    if h5py.is_hdf5(input_path):
      detected_input_format = common.KERAS_MODEL
    elif input_path.endswith('saved_model.pb'):
      input_path = os.path.dirname(input_path)
      detected_input_format = detect_saved_model(input_path)
    elif (input_path.endswith('model.json') and
          get_tfjs_model_type(input_path) == common.TFJS_LAYERS_MODEL_FORMAT):
      detected_input_format = common.TFJS_LAYERS_MODEL

  return detected_input_format, input_path


def input_path_message(answers):
  """Determine question for model's input path.
  Args:
    answer: Dict of user's answers to the questions
  """
  answer = answers[common.INPUT_FORMAT]
  message = 'The original path seems to be wrong, '
  if answer == common.KERAS_MODEL:
    return message + 'what is the path of input HDF5 file?'
  elif answer == common.TF_HUB_MODEL:
    return message + ("what is the TFHub module URL? \n"
                      "(i.e. https://tfhub.dev/google/imagenet/"
                      "mobilenet_v1_100_224/classification/1)")
  else:
    return message + 'what is the directory that contains the model?'


def validate_input_path(input_path, input_format):
  """Validate the input path for given input format.
  Args:
    input_path: input path of the model.
    input_format: model format string.
  """
  path = os.path.expanduser(input_path.strip())
  if not path:
    return 'Please enter a valid path'
  if input_format == common.TF_HUB_MODEL:
    if not re.match(TFHUB_VALID_URL_REGEX, path):
      return """This is not an valid URL for TFHub module: %s,
        We expect a URL that starts with http(s)://""" % path
  elif not os.path.exists(path):
    return 'Nonexistent path for the model: %s' % path
  if input_format in (common.KERAS_SAVED_MODEL, common.TF_SAVED_MODEL):
    is_dir = os.path.isdir(path)
    if not is_dir and not path.endswith('saved_model.pb'):
      return 'The path provided is not a directory or pb file: %s' % path
    if (is_dir and
        not any(f.endswith('saved_model.pb') for f in os.listdir(path))):
      return 'Did not find a .pb file inside the directory: %s' % path
    if input_format == common.KERAS_SAVED_MODEL:
      if detect_saved_model(input_path) != common.KERAS_SAVED_MODEL:
        return 'This is a saved model but not a keras saved model: %s' % path

  if input_format == common.TFJS_LAYERS_MODEL:
    is_dir = os.path.isdir(path)
    if not is_dir and not path.endswith('model.json'):
      return 'The path provided is not a directory or json file: %s' % path
    if is_dir and not any(f.endswith('model.json') for f in os.listdir(path)):
      return 'Did not find the model.json file inside the directory: %s' % path
  if input_format == common.KERAS_MODEL:
    if not h5py.is_hdf5(path):
      return 'The path provided is not a keras model file: %s' % path
  return True


def expand_input_path(input_path):
  """Expand the relative input path to absolute path, and add layers model file
  name to the end if input format is `tfjs_layers_model`.
  Args:
    input_path: input path of the model.
  Returns:
    string: return expanded input path.
  """
  input_path = os.path.expanduser(input_path.strip())
  is_dir = os.path.isdir(input_path)
  if is_dir:
    for fname in os.listdir(input_path):
      if fname.endswith('.json'):
        filename = os.path.join(input_path, fname)
        return filename
  return input_path


def output_path_exists(output_path):
  """Check the existence of the output path.
  Args:
    output_path: input path of the model.
  Returns:
    bool: return true when the output directory exists.
  """
  if os.path.exists(output_path):
    return True
  return False


def generate_arguments(params):
  """Generate the tensorflowjs command string for the selected params.
  Args:
    params: user selected parameters for the conversion.
  Returns:
    list: the argument list for converter.
  """
  args = []
  not_param_list = [common.INPUT_PATH, common.OUTPUT_PATH,
                    'overwrite_output_path', 'quantize']
  no_false_param = [common.SPLIT_WEIGHTS_BY_LAYER, common.SKIP_OP_CHECK]
  for key, value in sorted(params.items()):
    if key not in not_param_list and value is not None:
      if key in no_false_param:
        if value is True:
          args.append('--%s' % (key))
      else:
        args.append('--%s=%s' % (key, value))

  args.append(params[common.INPUT_PATH])
  args.append(params[common.OUTPUT_PATH])
  return args


def is_saved_model(input_format):
  """Check if the input path contains saved model.
  Args:
    input_format: input model format.
  Returns:
    bool: whether this is for a saved model conversion.
  """
  return input_format == common.TF_SAVED_MODEL

def available_output_formats(answers):
  """Generate the output formats for given input format.
  Args:
    ansowers: user selected parameter dict.
  """
  input_format = answers[common.INPUT_FORMAT]
  if input_format == common.KERAS_MODEL:
    return [{
        'key': 'g', # shortcut key for the option
        'name': 'Tensorflow.js Graph Model',
        'value': common.TFJS_GRAPH_MODEL,
    }, {
        'key': 'l',
        'name': 'TensoFlow.js Layers Model',
        'value': common.TFJS_LAYERS_MODEL,
    }]
  if input_format == common.TFJS_LAYERS_MODEL:
    return [{
        'key': 'k', # shortcut key for the option
        'name': 'Keras Model',
        'value': common.KERAS_MODEL,
    }, {
        'key': 's',
        'name': 'Keras Saved Model',
        'value': common.KERAS_SAVED_MODEL,
    }, {
        'key': 'l',
        'name': 'TensoFlow.js Layers Model',
        'value': common.TFJS_LAYERS_MODEL,
    }]
  return []


def available_tags(answers):
  """Generate the available saved model tags from the proto file.
  Args:
    ansowers: user selected parameter dict.
  """
  if is_saved_model(answers[common.INPUT_FORMAT]):
    saved_model = loader_impl.parse_saved_model(answers[common.INPUT_PATH])
    tags = []
    for meta_graph in saved_model.meta_graphs:
      tags.append(",".join(meta_graph.meta_info_def.tags))
    return tags
  return []


def available_signature_names(answers):
  """Generate the available saved model signatures from the proto file
    and selected tags.
  Args:
    ansowers: user selected parameter dict.
  """
  if (is_saved_model(answers[common.INPUT_FORMAT]) and
      common.SAVED_MODEL_TAGS in answers):
    path = answers[common.INPUT_PATH]
    tags = answers[common.SAVED_MODEL_TAGS]
    saved_model = loader_impl.parse_saved_model(path)
    for meta_graph in saved_model.meta_graphs:
      if tags == ",".join(meta_graph.meta_info_def.tags):
        signatures = []
        for key in meta_graph.signature_def:
          input_nodes = meta_graph.signature_def[key].inputs
          output_nodes = meta_graph.signature_def[key].outputs
          signatures.append(
              {'value': key,
               'name': format_signature(key, input_nodes, output_nodes)})
        return signatures
  return []


def format_signature(name, input_nodes, output_nodes):
  string = "signature name: %s\n" % name
  string += "        inputs: %s" % format_nodes(input_nodes)
  string += "        outputs: %s" % format_nodes(output_nodes)
  return string


def format_nodes(nodes):
  string = "%s of %s\n" % (3 if len(nodes) > 3 else len(nodes), len(nodes))
  count = 0
  for key in nodes:
    value = nodes[key]
    string += "              name: %s, " % value.name
    string += "dtype: %s, " % types_pb2.DataType.Name(value.dtype)
    if value.tensor_shape.unknown_rank:
      string += "shape: Unknown\n"
    else:
      string += "shape: %s\n" % [x.size for x in value.tensor_shape.dim]
    count += 1
    if count >= 3:
      break
  return string


def input_format_string(base, target_format, detected_format):
  if target_format == detected_format:
    return base + ' *'
  else:
    return base


def input_format_message(detected_input_format):
  message = 'What is your input model format? '
  if detected_input_format:
    message += '(auto-detected format is marked with *)'
  else:
    message += '(model format cannot be detected.) '
  return message

def update_output_path(output_path, params):
  output_path = os.path.expanduser(output_path.strip())
  if (common.OUTPUT_FORMAT in params and
      params[common.OUTPUT_FORMAT] == common.KERAS_MODEL):
    if os.path.isdir(output_path):
      output_path = os.path.join(output_path, 'model.H5')
  return output_path

def input_formats(detected_format):
  formats = [{
      'key': 'k',
      'name': input_format_string('Keras (HDF5)', common.KERAS_MODEL,
                                  detected_format),
      'value': common.KERAS_MODEL
  }, {
      'key': 'e',
      'name': input_format_string('Tensorflow Keras Saved Model',
                                  common.KERAS_SAVED_MODEL,
                                  detected_format),
      'value': common.KERAS_SAVED_MODEL,
  }, {
      'key': 's',
      'name': input_format_string('Tensorflow Saved Model',
                                  common.TF_SAVED_MODEL,
                                  detected_format),
      'value': common.TF_SAVED_MODEL,
  }, {
      'key': 'h',
      'name': input_format_string('TFHub Module',
                                  common.TF_HUB_MODEL,
                                  detected_format),
      'value': common.TF_HUB_MODEL,
  }, {
      'key': 'l',
      'name': input_format_string('TensoFlow.js Layers Model',
                                  common.TFJS_LAYERS_MODEL,
                                  detected_format),
      'value': common.TFJS_LAYERS_MODEL,
  }]
  formats.sort(key=lambda x: x['value'] != detected_format)
  return formats


def run(dryrun):
  print('Welcome to TensorFlow.js Converter.')
  input_path = [{
      'type': 'input',
      'name': common.INPUT_PATH,
      'message': 'Please provide the path of model file or '
                 'the directory that contains model files. \n'
                 'If you are converting TFHub module please provide the URL.',
      'filter': os.path.expanduser,
      'validate':
          lambda path: 'Please enter a valid path' if not path else True
  }]

  input_params = PyInquirer.prompt(input_path, style=prompt_style)
  detected_input_format, normalized_path = detect_input_format(
      input_params[common.INPUT_PATH])
  input_params[common.INPUT_PATH] = normalized_path

  formats = [
      {
          'type': 'list',
          'name': common.INPUT_FORMAT,
          'message': input_format_message(detected_input_format),
          'choices': input_formats(detected_input_format)
      }, {
          'type': 'list',
          'name': common.OUTPUT_FORMAT,
          'message': 'What is your output format?',
          'choices': available_output_formats,
          'when': lambda answers: value_in_list(answers, common.INPUT_FORMAT,
                                                (common.KERAS_MODEL,
                                                 common.TFJS_LAYERS_MODEL))
      }
  ]
  format_params = PyInquirer.prompt(formats, input_params, style=prompt_style)
  message = input_path_message(format_params)

  questions = [
      {
          'type': 'input',
          'name': common.INPUT_PATH,
          'message': message,
          'filter': expand_input_path,
          'validate': lambda value: validate_input_path(
              value, format_params[common.INPUT_FORMAT]),
          'when': lambda answers: (not detected_input_format)
      },
      {
          'type': 'list',
          'name': common.SAVED_MODEL_TAGS,
          'choices': available_tags,
          'message': 'What is tags for the saved model?',
          'when': lambda answers: (is_saved_model(answers[common.INPUT_FORMAT])
                                   and
                                   (common.OUTPUT_FORMAT not in format_params
                                    or format_params[common.OUTPUT_FORMAT] ==
                                    common.TFJS_GRAPH_MODEL))
      },
      {
          'type': 'list',
          'name': common.SIGNATURE_NAME,
          'message': 'What is signature name of the model?',
          'choices': available_signature_names,
          'when': lambda answers: (is_saved_model(answers[common.INPUT_FORMAT])
                                   and
                                   (common.OUTPUT_FORMAT not in format_params
                                    or format_params[common.OUTPUT_FORMAT] ==
                                    common.TFJS_GRAPH_MODEL))
      },
      {
          'type': 'list',
          'name': 'quantize',
          'message': 'Do you want to compress the model? '
                     '(this will decrease the model precision.)',
          'choices': [{
              'name': 'No compression (Higher accuracy)',
              'value': None
          }, {
              'name': 'float16 quantization '
                      '(2x smaller, Minimal accuracy loss)',
              'value': 'float16'
          }, {
              'name': 'uint16 affine quantization (2x smaller, Accuracy loss)',
              'value': 'uint16'
          }, {
              'name': 'uint8 affine quantization (4x smaller, Accuracy loss)',
              'value': 'uint8'
          }]
      },
      {
          'type': 'input',
          'name': common.QUANTIZATION_TYPE_FLOAT16,
          'message': 'Please enter the layers to apply float16 quantization '
                     '(2x smaller, minimal accuracy tradeoff).\n'
                     'Supports wildcard expansion with *, e.g., conv/*/weights',
          'default': '*',
          'when': lambda answers:
                  value_in_list(answers, 'quantize', ('float16'))
      },
      {
          'type': 'input',
          'name': common.QUANTIZATION_TYPE_UINT8,
          'message': 'Please enter the layers to apply affine 1-byte integer '
                     'quantization (4x smaller, accuracy tradeoff).\n'
                     'Supports wildcard expansion with *, e.g., conv/*/weights',
          'default': '*',
          'when': lambda answers:
                  value_in_list(answers, 'quantize', ('uint8'))
      },
      {
          'type': 'input',
          'name': common.QUANTIZATION_TYPE_UINT16,
          'message': 'Please enter the layers to apply affine 2-byte integer '
                     'quantization (2x smaller, accuracy tradeoff).\n'
                     'Supports wildcard expansion with *, e.g., conv/*/weights',
          'default': '*',
          'when': lambda answers:
                  value_in_list(answers, 'quantize', ('uint16'))
      },
      {
          'type': 'input',
          'name': common.WEIGHT_SHARD_SIZE_BYTES,
          'message': 'Please enter shard size (in bytes) of the weight files?',
          'default': str(4 * 1024 * 1024),
          'validate':
              lambda size: ('Please enter a positive integer' if not
                            (size.isdigit() and int(size) > 0) else True),
          'when': lambda answers: (value_in_list(answers, common.OUTPUT_FORMAT,
                                                 (common.TFJS_LAYERS_MODEL,
                                                  common.TFJS_GRAPH_MODEL)) or
                                   value_in_list(answers, common.INPUT_FORMAT,
                                                 (common.TF_SAVED_MODEL,
                                                  common.TF_HUB_MODEL)))
      },
      {
          'type': 'confirm',
          'name': common.SPLIT_WEIGHTS_BY_LAYER,
          'message': 'Do you want to split weights by layers?',
          'default': False,
          'when': lambda answers: (value_in_list(answers, common.OUTPUT_FORMAT,
                                                 (common.TFJS_LAYERS_MODEL)) and
                                   value_in_list(answers, common.INPUT_FORMAT,
                                                 (common.KERAS_MODEL,
                                                  common.KERAS_SAVED_MODEL)))
      },
      {
          'type': 'confirm',
          'name': common.SKIP_OP_CHECK,
          'message': 'Do you want to skip op validation? \n'
                     'This will allow conversion of unsupported ops, \n'
                     'you can implement them as custom ops in tfjs-converter.',
          'default': False,
          'when': lambda answers: value_in_list(answers, common.INPUT_FORMAT,
                                                (common.TF_SAVED_MODEL,
                                                 common.TF_HUB_MODEL))
      },
      {
          'type': 'confirm',
          'name': common.STRIP_DEBUG_OPS,
          'message': 'Do you want to strip debug ops? \n'
                     'This will improve model execution performance.',
          'default': True,
          'when': lambda answers: value_in_list(answers, common.INPUT_FORMAT,
                                                (common.TF_SAVED_MODEL,
                                                 common.TF_HUB_MODEL))
      },
      {
          'type': 'confirm',
          'name': common.CONTROL_FLOW_V2,
          'message': 'Do you want to enable Control Flow V2 ops? \n'
                     'This will improve branch and loop execution performance.',
          'default': True,
          'when': lambda answers: value_in_list(answers, common.INPUT_FORMAT,
                                                (common.TF_SAVED_MODEL,
                                                 common.TF_HUB_MODEL))
      }
  ]
  params = PyInquirer.prompt(questions, format_params, style=prompt_style)

  output_options = [
      {
          'type': 'input',
          'name': common.OUTPUT_PATH,
          'message': 'Which directory do you want to save '
                     'the converted model in?',
          'filter': lambda path: update_output_path(path, params),
          'validate': lambda path: len(path) > 0
      },
      {
          'type': 'confirm',
          'message': 'The output already directory exists, '
                     'do you want to overwrite it?',
          'name': 'overwrite_output_path',
          'default': False,
          'when': lambda ans: output_path_exists(ans[common.OUTPUT_PATH])
      }
  ]

  while (common.OUTPUT_PATH not in params or
         output_path_exists(params[common.OUTPUT_PATH]) and
         not params['overwrite_output_path']):
    params = PyInquirer.prompt(output_options, params, style=prompt_style)

  arguments = generate_arguments(params)
  print('converter command generated:')
  print('tensorflowjs_converter %s' % ' '.join(arguments))
  print('\n\n')

  log_file = os.path.join(tempfile.gettempdir(), 'converter_error.log')
  if not dryrun:
    try:
      converter.convert(arguments)
      print('\n\nFile(s) generated by conversion:')

      print("Filename {0:25} Size(bytes)".format(''))
      total_size = 0
      output_path = params[common.OUTPUT_PATH]
      if os.path.isfile(output_path):
        output_path = os.path.dirname(output_path)
      for basename in sorted(os.listdir(output_path)):
        filename = os.path.join(output_path, basename)
        size = os.path.getsize(filename)
        print("{0:35} {1}".format(basename, size))
        total_size += size
      print("Total size:{0:24} {1}".format('', total_size))
    except BaseException:
      exc_type, exc_value, exc_traceback = sys.exc_info()
      lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
      with open(log_file, 'a') as writer:
        writer.write(''.join(line for line in lines))
      print('Conversion failed, please check error log file %s.' % log_file)

def pip_main():
  """Entry point for pip-packaged binary.

  Note that pip-packaged binary calls the entry method without
  any arguments, which is why this method is needed in addition to the
  `main` method below.
  """
  main([' '.join(sys.argv[1:])])


def main(argv):
  if argv[0] and not argv[0] == '--dryrun':
    print("Usage: tensorflowjs_wizard [--dryrun]")
    sys.exit(1)
  dry_run = argv[0] == '--dryrun'
  run(dry_run)

if __name__ == '__main__':
  tf.app.run(main=main, argv=[' '.join(sys.argv[1:])])
