from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pickle
import os
import numpy as np
import re
from PIL import Image
from os import listdir
from os.path import isfile, join
import json

path_to_images = 'runes/'
output_file_image_collage = 'output.png'
output_file_labels_packed = 'labels'
output_file_label_names = 'labelNames.json'
number_of_channels = 1

def pack(classes, labels):
    n_classes = len(classes)
    length = len(labels)
    result = [np.NaN] * length * n_classes

    i = 0
    while i < length:
        label_candidate = labels[i]
        index = classes.index(label_candidate)
        offset = (i * n_classes)
        result[offset + index] = 1
        i += 1
    return result

def select_alpha_channel(image_array):
    return np.array([image_array[3]])

def determine_image_alpha_channel(image):
    # Image represented as tensor (width x height x channels)
    image_as_array = np.array(image)
    # Select just the alpha channel values from tensor
    return np.apply_along_axis(select_alpha_channel, axis=2, arr=image_as_array)

paths_to_images = [f for f in listdir(path_to_images) if
                   isfile(join(path_to_images, f)) and re.match('.*\.png', f)]
example_file_pattern = r'.*_([^(\s]+)(?: ?\([0-9]+\))?\.png$'


# Determine label from filename
labels = [re.search(example_file_pattern, s).group(1) for s in paths_to_images]
if len(labels) != len(paths_to_images):
    raise ValueError('Expected number of labels to be equal to the number of example images!')

# Get array of distinct labels
labels_clear_text = np.unique(np.asarray(labels))
indexed_classes = labels_clear_text.tolist()
labels5 = pack(indexed_classes, labels)
print('...', len(labels_clear_text), 'classes found')

# Pack labels into the uint8 array that demo builder expects
packed_labels = np.asarray(labels5).astype('uint8')
packed_labels.tofile(output_file_labels_packed)
print('...Saved packed labels to:', output_file_labels_packed)

# Also emit a list of the names that we associate with the labels
with open(output_file_label_names, 'w') as outfile:
    json.dump(indexed_classes, outfile)
print('...Saved label names to:', output_file_label_names)

# Load all image data into array. Note we load just one channel: transparency (the alpha channel)
loaded_images = [determine_image_alpha_channel(
    Image.open(join(path_to_images, path_to_image))) for path_to_image in
                 paths_to_images]
images_array = np.array(loaded_images)

print('Read', images_array.shape[0], 'images')
print('min/max pixel values: ', images_array.min(), '/', images_array.max())

# Make each image take a single row in the big batch image by flattening the
# width (2nd) and height (3rd) dimension.
# a has shape N x (Width*Height) x Channels.
images_array = images_array.reshape([images_array.shape[0], -1, number_of_channels]).squeeze()
im = Image.fromarray(images_array)
im.save(output_file_image_collage)
print('Saved image with width/height', im.size, 'at', output_file_image_collage)
