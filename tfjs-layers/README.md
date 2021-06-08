# TensorFlow.js Layers: High-Level Machine Learning Model API

A part of the TensorFlow.js ecosystem, TensorFlow.js Layers is a high-level
API built on [TensorFlow.js Core](/tfjs-core),
enabling users to build, train and execute deep learning models in the browser.
TensorFlow.js Layers is modeled after
[Keras](https://keras.io/) and
[tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras) and can
load models saved from those libraries.

## Importing

There are three ways to import TensorFlow.js Layers

1. You can access TensorFlow.js Layers through the union package
   between the TensorFlow.js Core and Layers:
   [@tensorflow/tfjs](https://www.npmjs.com/package/@tensorflow/tfjs)
2. You can get [TensorFlow.js](https://github.com/tensorflow/tfjs) Layers as a module:
   [@tensorflow/tfjs-layers](https://www.npmjs.com/package/@tensorflow/tfjs-layers).
   Note that `tfjs-layers` has peer dependency on tfjs-core, so if you import
   `@tensorflow/tfjs-layers`, you also need to import
   `@tensorflow/tfjs-core`.
3. As a standalone through [unpkg](https://unpkg.com/).

Option 1 is the most convenient, but leads to a larger bundle size (we will be
adding more packages to it in the future). Use option 2 if you care about bundle
size.

## Getting started

### Building, training and executing a model

The following example shows how to build a toy model with only one `dense` layer
to perform linear regression.

```js
import * as tf from '@tensorflow/tfjs';

// A sequential model is a container which you can add layers to.
const model = tf.sequential();

// Add a dense layer with 1 output unit.
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// Specify the loss type and optimizer for training.
model.compile({loss: 'meanSquaredError', optimizer: 'SGD'});

// Generate some synthetic data for training.
const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
const ys = tf.tensor2d([[1], [3], [5], [7]], [4, 1]);

// Train the model.
await model.fit(xs, ys, {epochs: 500});

// Ater the training, perform inference.
const output = model.predict(tf.tensor2d([[5]], [1, 1]));
output.print();
```

### Loading a pretrained Keras model

You can also load a model previously trained and saved from elsewhere (e.g.,
from Python Keras) and use it for inference or transfer learning in the browser.

For example, in Python, save your Keras model using
[tensorflowjs](https://pypi.org/project/tensorflowjs/),
which can be installed using `pip install tensorflowjs`.


```python
import tensorflowjs as tfjs

# ... Create and train your Keras model.

# Save your Keras model in TensorFlow.js format.
tfjs.converters.save_keras_model(model, '/path/to/tfjs_artifacts/')

# Then use your favorite web server to serve the directory at a URL, say
#   http://foo.bar/tfjs_artifacts/model.json
```

To load the model with TensorFlow.js Layers:

```js
import * as tf from '@tensorflow/tfjs';

const model = await tf.loadLayersModel('http://foo.bar/tfjs_artifacts/model.json');
// Now the model is ready for inference, evaluation or re-training.
```

## For more information

- [TensorFlow.js API documentation](https://js.tensorflow.org/api/latest/)
- [TensorFlow.js Tutorials](https://js.tensorflow.org/tutorials/)
