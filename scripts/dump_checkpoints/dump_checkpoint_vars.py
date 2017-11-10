# Copyright 2017 Google Inc. All Rights Reserved.
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

"""
This script is an entry point for dumping checkpoints for various deeplearning
frameworks.
"""
from __future__ import print_function

import argparse


def get_checkpoint_dumper(model_type, checkpoint_file, output_dir, remove_variables_regex):
  """Returns Checkpoint dumper instance for a given model type.

  Parameters
  ----------
  model_type : str
      Type of deeplearning framework
  checkpoint_file : str
      Path to checkpoint file
  output_dir : str
      Path to output directory
  remove_variables_regex : str
      Regex for variables to be ignored

  Returns
  -------
  (TensorflowCheckpointDumper, PytorchCheckpointDumper)
      Checkpoint Dumper Instance for corresponding model type

  Raises
  ------
  Error
      If particular model type is not supported
  """
  if model_type == 'tensorflow':
    from tensorflow_checkpoint_dumper import TensorflowCheckpointDumper

    return TensorflowCheckpointDumper(
      checkpoint_file, output_dir, remove_variables_regex)
  elif model_type == 'pytorch':
    from pytorch_checkpoint_dumper import PytorchCheckpointDumper

    return PytorchCheckpointDumper(
      checkpoint_file, output_dir, remove_variables_regex)
  else:
    raise Error('Currently, "%s" models are not supported'.format(model_type))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model_type',
      type=str,
      required=True,
      help='Model checkpoint type')
  parser.add_argument(
      '--checkpoint_file',
      type=str,
      required=True,
      help='Path to the model checkpoint')
  parser.add_argument(
      '--output_dir',
      type=str,
      required=True,
      help='The output directory where to store the converted weights')
  parser.add_argument(
      '--remove_variables_regex',
      type=str,
      default='',
      help='A regular expression to match against variable names that should '
      'not be included')
  FLAGS, unparsed = parser.parse_known_args()

  if unparsed:
    parser.print_help()
    print('Unrecognized flags: ', unparsed)
    exit(-1)

  checkpoint_dumper = get_checkpoint_dumper(
    FLAGS.model_type, FLAGS.checkpoint_file, FLAGS.output_dir, FLAGS.remove_variables_regex)
  checkpoint_dumper.build_and_dump_vars()
