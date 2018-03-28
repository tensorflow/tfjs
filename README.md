# TensorFlow.js

TensorFlow.js is an open-source hardware-accelerated JavaScript library for
building, training and serving machine learning models. When running in the
browser, it utilizes WebGL acceleration. TensorFlow.js is also convenient and
intuitive, modeled after
[Keras](https://keras.io/) and
[tf.layers](https://www.tensorflow.org/api_docs/python/tf/layers) and can
load models saved from those libraries.

This repository conveniently contains the logic and scripts to form
a version-matched **union** package,
[@tensorflowjs/tfjs](https://www.npmjs.com/package/@tensorflow/tfjs), from

- [TensorFlow.js Core](https://github.com/tensorflow/tfjs-core),
  a flexible low-level API, formerly known as *deeplearn.js*.
- [TensorFlow.js Layers](https://github.com/tensorflow/tfjs-layers),
  a high-level API modeled after [Keras](https://keras.io/).


## Importing

You can import TensorFlow.js Union directly via yarn or npm.
`yarn add @tensorflow/tfjs` or `npm install @tensorflow/tfjs`.
See snippets below for examples.

Alternatively you can use a script tag. Here we load it from a CDN.
In this case it will be available as a global variable named `tf`.

You can replace also specify which version to load replacing `@latest`
with a specific
version string (e.g. `0.6.0`).

```html
<script src="https://cdn.jsdelivr.net/npm/tensorflow/tfjs@latest"></script>
<!-- or -->
<script src="https://unpkg.com/tensorflow/tfjs@latest"></script>
```


## Usage Examples

Many examples illustrating how to use TensorFlow.js in ES5, ES6 and
TypeScript are available from the
[Examples repository](https://github.com/tensorflow/tfjs-examples)
and the
[TensorFlow.js Tutorials](https://js.tensorflow.org/tutorials/)


### Direct tensor manipulation

Let's add a scalar value to a 1D Tensor. TensorFlow.js supports _broadcasting_
the value of scalar over all the elements in the tensor.

```js
import * as tf from '@tensorflow/tfjs'; // If not loading the script as a global

const a = tf.tensor1d([1, 2, 3]);
const b = tf.scalar(2);

const result = a.add(b); // a is not modified, result is a new tensor
result.data().then(data => console.log(data)); // Float32Array([3, 4, 5]

// Alternatively you can use a blocking call to get the data.
// However this might slow your program down if called repeatedly.
console.log(result.dataSync()); // Float32Array([3, 4, 5]
```

See the
[core-concepts tutorial](https://js.tensorflow.org/tutorials/core-concepts.html)
 for more.

### Building, training, and executing a model using Layers

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

For a deeper dive into building a layers classifier, see the
[MNIST tutorial](https://js.tensorflow.org/tutorials/mnist.html)


### Loading a pretrained Keras model using Layers

You can also load a model previously trained and saved from elsewhere (e.g.,
from Python Keras) and use it for inference or transfer learning in the browser.
 More details in the
 [import-keras tutorial](https://js.tensorflow.org/tutorials/import-keras.html)

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

const model = await tf.loadModel('http://foo.bar/tfjs_artifacts/model.json');
// Now the model is ready for inference, evaluation or re-training.
```




## How to find more!

Again, see the
[Examples repository](https://github.com/tensorflow/tfjs-examples) and the
[TensorFlow.js Tutorials](https://js.tensorflow.org/tutorials/)
 for many more examples of how to build models and manipulate tensors.


## Supported Environments

**TensorFlow.js** targets environments with WebGL 1.0 or WebGL 2.0. For devices
without the `OES_texture_float` extension, we fall back to fixed precision
floats backed by a `gl.UNSIGNED_BYTE` texture. For platforms without WebGL,
we provide CPU fallbacks.


## Additional Resources

TensorFlow.js is a part of the
[TensorFlow](https://www.tensorflow.org) ecosystem.
You can import pre-trained TensorFlow
[SavedModels](https://www.tensorflow.org/programmers_guide/saved_model) and
[Keras models](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model),
for execution and retraining.

For more information on the API, follow the links to their Core and Layers
repositories below, or visit [js.tensorflow.org](https://js.tensorflow.org).


