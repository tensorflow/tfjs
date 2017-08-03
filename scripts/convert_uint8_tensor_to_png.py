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
"""Converts an array of 3D tensors to pngs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
from PIL import Image

FLAGS = None


def main():
  fpath = os.path.expanduser(FLAGS.uint8_tensor_file)
  with open(fpath, 'rb') as f:
    # a has shape N x Width x Height x Channels
    a = np.frombuffer(f.read(), np.uint8).reshape(
        [-1, FLAGS.size, FLAGS.size, FLAGS.num_channels])

  print('Read', a.shape[0], 'images')
  print('min/max pixel values: ', a.min(), '/', a.max())

  # Make each image take a single row in the big batch image by flattening the
  # width (2nd) and height (3rd) dimension.
  # a has shape N x (Width*Height) x Channels.
  a = a.reshape([a.shape[0], -1, FLAGS.num_channels]).squeeze()
  im = Image.fromarray(a)
  im.save(fpath + '.png')
  print('Saved image with width/height', im.size, 'at', fpath + '.png')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--uint8_tensor_file',
      type=str,
      required=True,
      help='File path to the binary uint8 tensor to convert to png')
  parser.add_argument(
      '--size', type=int, required=True, help='Width/Height of each image')
  parser.add_argument(
      '--num_channels', type=int, required=True, help='Number of channelse')
  FLAGS, unparsed = parser.parse_known_args()
  if unparsed:
    print('Error, unrecognized flags:', unparsed)
    exit(-1)
  main()
