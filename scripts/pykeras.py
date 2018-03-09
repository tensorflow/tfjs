# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================
"""Artifact conversion to and from Python Keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os

import h5py
from scripts import h5_conversion
import write_weights


def dispatch_pykeras_conversion(pykeras_file):
  """Returns a json object representing the contents of pykeras_file.

  Auto-detects saved_model versus weights-only and generates the correct
  json in either case.

  Args:
    pykeras_file: an opened HDF5 file containing keras data

  Returns:
    (json, groups)
      json: a json dictionary (empty if unused)
      groups: an array of weight_groups as defined in tfjs weights_writer
  """
  converter = h5_conversion.HDF5Converter()

  if 'layer_names' in pykeras_file.attrs:
    return ({}, converter.h5_weights_to_tfjs_format(pykeras_file))
  else:
    return converter.h5_merged_saved_model_to_tfjs_format(pykeras_file)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      'Converters for Python Keras-generated checkpoints')
  parser.add_argument(
      'h5_path', type=str, help='Path to the input HDF5 (.h5) file')
  parser.add_argument(
      'output_dir', type=str, help='Path for all output artifacts')
  parser.add_argument(
      'json_filename', type=str, help='Filename within output_dir for topology')

  args = parser.parse_args()
  if args.json_filename == '':
    filename = 'topology.json'
  else:
    filename = args.json_filename

  h5_file = h5py.File(args.h5_path)

  json, groups = dispatch_pykeras_conversion(h5_file)
  h5_conversion.HDF5Converter().write_artifacts(json, groups, args.output_dir, filename)
