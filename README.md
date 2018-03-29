# TensorFlow.js

TensorFlow.js is an open-source hardware-accelerated JavaScript library for
training and deploying machine learning models.

**Develop ML in the Browser**

Use flexible and intuitive APIs to build models from scratch using the low-level
JavaScript linear algebra library or the high-level layers API.

**Run Existing models**

Use TensorFlow.js model converters to run pre-existing TensorFlow models right
in the browser.

**Retrain Existing models**

Retrain pre-existing ML models using sensor data connected to the browser, or
other client-side data.

## Importing

You can import TensorFlow.js directly via yarn or npm:
`yarn add @tensorflow/tfjs` or `npm install @tensorflow/tfjs`.

Alternatively you can use a script tag. The library will be available as
a global variable named `tf`:

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
<!-- or -->
<script src="https://unpkg.com/@tensorflow/tfjs@latest"></script>
```

You can also specify which version to load replacing `@latest`
with a specific version string (e.g. `0.6.0`).

## About this repo

This repository contains the logic and scripts that combine
two packages:
- [TensorFlow.js Core](https://github.com/tensorflow/tfjs-core),
  a flexible low-level API, formerly known as *deeplearn.js*.
- [TensorFlow.js Layers](https://github.com/tensorflow/tfjs-layers),
  a high-level API which implements functionality similar to
  [Keras](https://keras.io/).

If you care about bundle size, you can import those individual packages.

## Examples

Check out our
[examples repository](https://github.com/tensorflow/tfjs-examples)
and our [tutorials](https://js.tensorflow.org/tutorials/).


## Getting started

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

Now, let's build a toy model to perform linear regression.

```js
import * as tf from '@tensorflow/tfjs';

// A sequential model is a container which you can add layers to.
const model = tf.sequential();

// Add a dense layer with 1 output unit.
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// Specify the loss type and optimizer for training.
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Generate some synthetic data for training.
const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
const ys = tf.tensor2d([[1], [3], [5], [7]], [4, 1]);

// Train the model.
await model.fit(xs, ys, {epochs: 500});

// Ater the training, perform inference.
const output = model.predict(tf.tensor2d([[5]], [1, 1]));
output.print();
```

For a deeper dive into building models, see the
[MNIST tutorial](https://js.tensorflow.org/tutorials/mnist.html)

We also support porting pre-trained models from:
- [TensorFlow SavedModel](https://github.com/tensorflow/tfjs-converter).
- [Keras](https://js.tensorflow.org/tutorials/import-keras.html). 

## Find out more

TensorFlow.js is a part of the
[TensorFlow](https://www.tensorflow.org) ecosystem.
You can import pre-trained TensorFlow
[SavedModels](https://www.tensorflow.org/programmers_guide/saved_model) and
[Keras models](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model),
for execution and retraining.

For more information on the API, follow the links to their Core and Layers
repositories below, or visit [js.tensorflow.org](https://js.tensorflow.org).


