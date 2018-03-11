# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================
"""Tools related to the Keras MobileNet application.

The model architecture and weights of this MobileNet implementation are defined
within Keras itself.  See details at https://keras.io/applications/#mobilenet


Mode "serialize":
  Exports the Keras MobileNet model into a format compatable
  with TensorFlow.js.

  example:

  python scripts/mobilenet.py \
      --mode=serialize \
      --artifacts ./dist/demo/mobilenet

Mode "apply":
  Applies the pre-trained Keras mobilenet model to a provided image.  This is
  useful as a check against the predictions made in the JS version.

  example:

  python scripts:mobilenet \
      --mode=apply \
      --img_file_name=/tmp/test.jpg
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import shutil

import numpy as np
from PIL import Image

from keras.applications import mobilenet
from scripts import h5_conversion


def _serialize_mobilenet_model_to_json():
  """Writes a complete keras.Model to disk into JS readable format.
  """
  model = mobilenet.MobileNet(alpha=.25)
  h5_conversion.save_model(model, FLAGS.artifacts_dir)


def _load_image(file_name, new_size):
  """Loads, resizes, and returns an image from the filesystem.

  Args:
    file_name: string full path of an image file on disk.
    new_size: A 2-tuple (width, height) to resize the image to before returning.

  Returns:
    A loaded PIL.Image, resized to the provided new_size.
  """
  original_image = Image.open(file_name)
  width, height = original_image.size
  print('The original image size is {wide} wide x {height} '
        'high'.format(wide=width, height=height))
  resized_image = original_image.resize(new_size)
  width, height = resized_image.size
  print('The resized image size is {wide} wide x {height} '
        'high'.format(wide=width, height=height))
  return resized_image


def _process_arr(raw_arr):
  """Normalizes the image between [-1, 1)."""
  return (raw_arr - 127.) / 127.


def _apply_mobilenet_model_to_image():
  """Applies the pre-trained keras MobileNet model to a provided image.

  Loads the image from the filesystem at the location provided via
  FLAGS.img_file_name.  Pre-processes the image by reshaping and normalizing
  between [-1.0, 1.0).  Prints the top predictions to stdout.
  """
  img_file_name = FLAGS.img_file_name
  resized_image = _load_image(img_file_name, (224, 224))
  raw_arr = np.array(resized_image)
  arr = _process_arr(raw_arr)
  batch = arr.reshape((1, 224, 224, 3))
  model = mobilenet.MobileNet()
  output = model.predict(batch)

  k = 10
  one_img_outputs = output[0, :]
  one_img_outputs.reshape((1000, 1))
  top_k = (-one_img_outputs).argsort()[0:k]
  print('The top predictions are indices %s' % str(top_k))
  print('They have scores %s' % str([one_img_outputs[i] for i in top_k]))
  with open('scripts/imagenet_class_names.json', 'r') as f:
    imagenet_class_names = json.load(f)
  print('They have names %s' % [imagenet_class_names[i] for i in top_k])

def main():
  if FLAGS.mode == 'serialize':
    _serialize_mobilenet_model_to_json()

    # Write ImageNet class names data to artifact directory.
    shutil.copyfile(
        'scripts/imagenet_class_names.json',
        os.path.join(FLAGS.artifacts_dir, 'imagenet_class_names.json'))
  elif FLAGS.mode == 'apply':
    _apply_mobilenet_model_to_image()
  else:
    print('Mode must be "serialize" or "apply". Got %s' % FLAGS.mode)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('MobileNet model tools')
  parser.add_argument(
      '--mode',
      type=str,
      default='',
      help='Mode for mobilenet tool.  Either "serialize" or "apply".')
  parser.add_argument(
      '--artifacts_dir',
      type=str,
      default='/tmp/mobilenet.keras',
      help='Local path for saving the TensorFlow.js artifacts.')
  parser.add_argument(
      '--img_file_name',
      type=str,
      default='/tmp/image.jpg',
      help='which file to load')
  FLAGS, _ = parser.parse_known_args()
  main()
