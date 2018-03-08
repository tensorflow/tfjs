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

import h5py
from scripts import h5_conversion

def dispatch_pykeras_conversion(pykeras_file, decimal_places=6):
  """Returns a json object representing the contents of pykeras_file.

  Auto-detects saved_model versus weights-only and generates the correct
  json in either case.

  Args:
    pykeras_file: an opened HDF5 file containing keras data
    decimal_places: Number of decimal places to round to.

  Returns:
    A JSON dictionairy of the contents of the file.
  """
  converter = h5_conversion.HDF5Converter(decimal_places)

  if 'layer_names' in pykeras_file.attrs:
    return converter.h5_weights_to_json(pykeras_file)
  else:
    return converter.h5_merged_saved_model_to_json(pykeras_file)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      'Converters for Python Keras-generated checkpoints')
  parser.add_argument(
      'h5_path', type=str, help='Path to the input HDF5 (.h5) file')
  parser.add_argument(
      'json_path', type=str, help='Path to the output json file')
  parser.add_argument(
      '--decimal_places', type=int, default=6,
      help='Decimal places to keep for the float weight values')
  parser.add_argument(
      '--pretty', action='store_true',
      help='Format the JSON string in the output file prettily')

  args = parser.parse_args()
  h5_file = h5py.File(args.h5_path)

  out = dispatch_pykeras_conversion(h5_file, args.decimal_places)
  json_string = (json.dumps(out, indent=2,
                            separators=(',', ': ')) if args.pretty
                 else json.dumps(out))

  with open(args.json_path, 'wt') as json_file:
    json_file.write(json_string)
