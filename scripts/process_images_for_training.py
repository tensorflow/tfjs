# Copyright 2017 Smesh LLC. All Rights Reserved.
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
# ======================================================================
# This program converts input images into a format suitable for use with the model builder demo.
# Contributed by Kaihu Chen, Smesh Labs (http://smesh.net/labs)
#
# Before running this program:            
#   - Put all images under <yourImageDir>   
#   - Each image must be prefixed with its class label, followed by '_'.
#       For example, cat_image00005.jpg
# To run: 
#   1. python process_images_for_training.py --inimgs <absolutePathToImages>
#       Example: python process_images_for_training.py --inimgs "/mnt/data/myBirdImages/*.jpg"
#
# Results: find in the current directoty an image file 'images.png' (extension '.png' added automatically),
# and a labels file 'labels' (or per command line options)
#
# Note:
#   - Make sure that the model-builder-datasets-configuration's data.labels.shape 
#     matches the number of classes found in data
#   - Make sure that the NN model's output layer matches the number of classes
# To-do:
#   - This code is not suitable for processing large number of images.
#   - Tested with python v2.7 and v3.5
#======================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import numpy as np
from PIL import Image
from PIL import ImageOps
import ntpath
import argparse
import os
import re
import sys

# for images
targetW = 32
targetH = 32
outImageFile = 'images.png'
## for labels
outLabels = 'labels'
delimiter = '_'

def preprocessImages(FLAGS):
  path = FLAGS.inimgs
  targetW = FLAGS.size
  targetH = FLAGS.size
  outImageFile = FLAGS.outimgs
  outLabels = FLAGS.outlabels
  delimiter = FLAGS.delimiter

  cwd = os.getcwd()
  print('...Current working directory =', cwd)
  print(len(sys.argv))
  if path:
    inputImages = path
  else:
    inputImages = cwd
        
  print('...Input imagefiles =', inputImages) 

  fileList = []
  newFiles = glob.glob(path)
  if newFiles is not None:
    print('...', len(newFiles), 'image files found')
    fileList.extend(newFiles)

  print('...Found a total of', len(fileList), 'images')
  if FLAGS.replicate > 1:
    fileList = np.tile(fileList, FLAGS.replicate)
    print('...Dataset has been replicated', FLAGS.replicate, 'times')

  thumbList = []
  for n in fileList:
    img = Image.open(n)
    thumb = ImageOps.fit(img, (targetW,targetH), Image.ANTIALIAS)
    thumbList.append(thumb)
    img.close()
  imageList = [np.array(im) for im in thumbList]
  imageList = np.array(imageList, dtype='uint8')
  a = imageList
  b = a.reshape([a.shape[0], -1, 3]).squeeze()
  im = Image.fromarray(b)
  im.save(outImageFile+'.png')
  print('...single output image size (width/height) =', FLAGS.size)
  print('...Saved composed image to:', outImageFile+'.png')

  # Process class labels
  pattern = '([a-zA-Z_]*)' + delimiter + '.*$'
  labels = [re.search(pattern, ntpath.basename(s)).group(1) for s in fileList]
  classesClearText = np.unique(np.asarray(labels))
  print('...', len(classesClearText), 'classes found:', classesClearText)
  FLAGS.nClassesIn = len(classesClearText)
  labels5 = pack(classesClearText.tolist(), labels, FLAGS)

  np.asarray(labels5).astype('uint8').tofile(outLabels)
  print('...Saved composed labels to:', outLabels)

def pack(classes, labels, FLAGS):
  i = 0
  nClassesIn = FLAGS.nClasses
  if nClassesIn:
    nClasses = nClassesIn
    print('...#classes forcibly set to', nClassesIn)
  else:
    nClasses = len(classes)
  length = len(labels)
  result = [ np.NaN ] * length * nClasses

  lumpedClass = None
  while i < length:
    oneLabel = labels[i]
    try:
      if FLAGS.vsOthers:
        matched = re.search(FLAGS.vsOthers, oneLabel)
        if matched: 
          index = classes.index(oneLabel)                   
        else:
          index = classes.index(oneLabel)
          if lumpedClass is None:
            lumpedClass = index
          else:
            index = lumpedClass
          print('...class altered from', oneLabel, 'to', index)
      else:
        index = classes.index(oneLabel)
      result[i*nClasses+index] = 1
    except:
      e = sys.exc_info()[0]
      print('!!! unknown class found, or invalid regex:', oneLabel)
      print( "<p>Error: %s</p>" % e )
    i += 1
  return result

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--inimgs',
    type=str,
    default='train/*.*',
    help='File path to the input images, default=./train/*.*')
  parser.add_argument(
    '--outimgs', type=str, default='images', help='Output file name (without extension) for the composed image')
  parser.add_argument(
    '--delimiter', type=str, default='_', help='Delimiter used for parsing class name out of file name. E.g., a file name of "ABC_butterflies.jpg" will yield a class a class name of ABC')
  parser.add_argument(
    '--outlabels', type=str, default='labels', help='Output file name for the composed labels')
  parser.add_argument(
    '--size', type=int, default=32, help='Width/Height of each image, default=32')
  parser.add_argument(
    '--num_channels', type=int, default=3, help='Number of channels')
  parser.add_argument(
    '--replicate', type=int, default=1, help='Replicate this dataset this many times, default=1')
  parser.add_argument(
    '--nClasses', type=int, help='Set the number of classes. If absent then compute based on data')
  parser.add_argument(
    '--vsOthers', type=str, help='When specified this regular expression causes all unmatched sample classes to be lumped into one single class')

  FLAGS, unparsed = parser.parse_known_args()
  if unparsed:
    print('Error, unrecognized flags:', unparsed)
    exit(-1)

  preprocessImages(FLAGS)
