---
layout: page
order: 6
---

# Example: Add a new data set to the model builder demo

In this tutorial we add a data set to the [model builder demo](https://deeplearnjs.org/demos/model-builder/) and train a neural network that can recognize individual glyphs of runes. It is similar to the canonical [MNIST example](https://www.tensorflow.org/get_started/mnist/pros) for recognizing handwritten digits, but with a different alphabet. This tutorial is meant as a minimal scaffold to start off your adventures in deep learning.

We will perform the following steps:

* Define a data set of rune images with their labels
* Convert the data to a format that the model builder demo can work with
* Add the data set to the model builder and actually train a neural network

The final step would be to export the trained model to use it in an actual application, but [the team is still working on that](https://github.com/PAIR-code/deeplearnjs/issues/33#issuecomment-323891195).

The complete code to follow along with this tutorial can be found in `demos/rune_recognition`. If you run into trouble during this tutorial, you might find inspiration in [the GitHub issue](https://github.com/PAIR-code/deeplearnjs/issues/20) that inspired this tutorial.

## Introduction
Suppose that you are making an app for recognizing runes, [the old Germanic letters](https://en.wikipedia.org/wiki/Runes) that were carved in stone and wood during the first millennium. The time and space where these runes were created varies a lot, and this leads to a lot of variation in the exact shape of these characters. Luckily the Unicode Consortium has
defined [a set of "idealized glyphs"](https://en.wikipedia.org/wiki/Runic_\(Unicode_block\)) that represent ideas of distinct runes. The variance in shapes makes the mapping of a given rune image to the Unicode pointer an interesting problem to solve through machine learning, so let's try to make a neural network that can recognize individual runes. To this end, we train a convolutional neural network (CNN) to interpret images of runes. The deeplearn.js model builder demo is already well-equipped with the common task to import a set of images along with their labels, so we add our own runic data set and try it out.

Before following along, make sure you have installed [Python](https://www.python.org/) and [node.js](https://nodejs.org/en/). I assume some rudimentary knowledge about neural networks / machine learning.

We get started by cloning the deeplearn.js source code: `git clone https://github.com/PAIR-code/deeplearnjs.git`.

## Prepare data set
The model builder expects two files that it can use for training a network: one file containing all example inputs and one file containing all example outputs. The inputs file is a png image in which every horizontal row represents one example instance, where individual pixels represent node activations. The labels file is likewise a large list of node activations, but it in a slightly different format.

I have prepared a reference set of rune images, hand-drawn and rendered by differents fonts, which you can find [here](https://github.com/digitalheir/deeplearnjs/releases/latest). If you want to generate your own samples, you can open `demos/rune_recognition/generate_train_examples_runes.html` in your browser.

Create a folder containing the example images and a file named `generate_rune_data_for_model_builder.py`. The script we are creating is inspired a helper script in the repository which you can find at `scripts/convert_uint8_tensor_to_png.py`.

First we get some bookkeeping out of the way:

````py
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

path_to_images = "runes/"
output_file_image_collage = 'rune_images.png'
output_file_labels_packed = 'rune_labels'
output_file_label_names = 'labelNames.json'
number_of_channels = 1

# Get all individual image paths from folder
paths_to_images = [f for f in listdir(path_to_images) if
                   isfile(join(path_to_images, f)) and re.match(".*\.png", f)]
# A convention we use in our file names: {any text}_{label}.png
example_file_pattern = r".*_([^(\s]+)(?: ?\([0-9]+\))?\.png$"

````

### Encode labels

In our neural network, the output layer is a list of nodes where each node represents one category. We encode our labels as a [one-hot](https://en.wikipedia.org/wiki/One-hot) array of zeros and ones so we can let the neural network know that these are the desired outputs.

First we determine the image labels from the filenames: and print them to a file that the model builder understands.

````py
# Determine label from filename
labels = [re.search(example_file_pattern, s).group(1) for s in paths_to_images]
if len(labels) != len(paths_to_images):
    raise ValueError("Expected number of labels to be equal to the number of example images!")

# Get array of distinct labels
labels_clear_text = np.unique(np.asarray(labels))
indexed_classes = labels_clear_text.tolist()
print('...', len(labels_clear_text), 'classes found')
````

Then we encode the labels to a one-hot encoded array of `uint8` numbers:

````py
def pack_labels(classes, labels):
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
one_hot_encoded_labels = pack(indexed_classes, labels)

# Pack labels into the uint8 array that demo builder expects
packed_labels = np.asarray(one_hot_encoded_labels).astype('uint8')
packed_labels.tofile(output_file_labels_packed)
print('...Saved packed labels to:', output_file_labels_packed)

# Also emit a list of the names that we associate with the labels, since they otherwise just an index in an array
with open(output_file_label_names, 'w') as outfile:
    json.dump(indexed_classes, outfile)
print('...Saved label names to:', output_file_label_names)

````

### Encode images

The input to our neural network is likewise an array of nodes. Our images are multi-dimensional, so we "squash" the image into a linear list of pixel activations. The neural network should then figure how these nodes are related to each other. Because our images are just black pixels on a transparent background, we throw away all color information and just select the transparency value. Note that we assume that all images are the same size!

```
def select_alpha_channel(image_array):
    return np.array([image_array[3]])

def determine_image_alpha_channel(image):
    # Image represented as tensor (width x height x channels)
    image_as_array = np.array(image)
    # Select just the alpha channel values from tensor
    return np.apply_along_axis(select_alpha_channel, axis=2, arr=image_as_array)

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
```

Open the output png file. You should see a large black image strewn with white dots. Each row in this image represents one of the images in the `runes` folder, as we shall when we load our data set in the model builder.

## Modify model builder demo

(See also the [development pointers](https://deeplearnjs.org/#development))

First we verify that we can run the model builder. From the project root, run:

* `yarn prep`
* `./scripts/watch-demo demos/model-builder`
* Will open a browser at http://127.0.0.1:8080/demos/model-builder/

If it works, open the file `demos/model-builder/model-builder-datasets-config.json`. The model builder defines its data sets in this file, so we are going to add our runes data set:

```
{
  "Runes": {
    "data": [{
      "name": "images",
      "path": "rune_images.png",
      "dataType": "png",
      "shape": [32, 32, 1]
    }, {
      "name": "labels",
      "path": "rune_labels",
      "dataType": "uint8",
      "shape": [90]
    }],
    "modelConfigs": {
      "Fully connected": {
        "path": "runes-fully-connected.json"
      },
      "Convolutional": {
        "path": "runes-conv.json"
      }
    }
  },
, ... all the other data sets ...
}
```

The data set definition refers to two different models: one for a fully connected neural net and one for a convolutional neural network. Because our task is so close to the task of recognizing hand-written digits, we make it easy for ourselves.
Just copy the `mnist-fully-connected.json` and `mnist-conv.json` files in this folder and changes their names to `runes-fully-connected.json` and `runes-conv.json`. There is one change though. We need to specify the number of labels in the output layer: in the `fully connected` layers, change the `hiddenUnits` field to the number of labels (90 for the example data set).

Now refresh the model builder. You should see the new runes data set. Select the convolutional model and click "Train". It should look something like this:

![Screenshot of model builder trained on runes](https://github.com/digitalheir/deeplearnjs/raw/rune_recognition_tutorial/demos/rune_recognition/runes_cnn.jpg)

Congratulations! You added just trained a neural network for recognizing runes.
