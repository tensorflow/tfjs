# TensorFlow.js

TensorFlow.js is an open-source hardware-accelerated JavaScript library for
training and deploying machine learning models.

**Develop ML in the Browser** <br/>
Use flexible and intuitive APIs to build models from scratch using the low-level
JavaScript linear algebra library or the high-level layers API.

**Run Existing models** <br/>
Use TensorFlow.js model converters to run pre-existing TensorFlow models right
in the browser.

**Retrain Existing models** <br/>
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

If you care about bundle size, you can import those packages individually.

## Examples

Check out our
[examples repository](https://github.com/tensorflow/tfjs-examples)
and our [tutorials](https://js.tensorflow.org/tutorials/).

## Migrating from deeplearn.js
See [these release notes](https://github.com/tensorflow/tfjs-core/releases/tag/v0.6.0)
for how to migrate from deeplearn.js to TensorFlow.js.

## Getting started

Let's add a scalar value to a vector. TensorFlow.js supports _broadcasting_
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

// After the training, perform inference.
const output = model.predict(tf.tensor2d([[5]], [1, 1]));
output.print();
```

For a deeper dive into building models, see the
[MNIST tutorial](https://js.tensorflow.org/tutorials/mnist.html)

## Importing pre-trained models

We support porting pre-trained models from:
- [TensorFlow SavedModel](https://github.com/tensorflow/tfjs-converter)
- [Keras](https://js.tensorflow.org/tutorials/import-keras.html)

## Find out more

[TensorFlow.js](https://js.tensorflow.org) is a part of the
[TensorFlow](https://www.tensorflow.org) ecosystem. For more info:
- [js.tensorflow.org](https://js.tensorflow.org)
- [Tutorials](https://js.tensorflow.org/tutorials)
- [API reference](https://js.tensorflow.org/api/latest/)
- [Help mailing list](https://groups.google.com/a/tensorflow.org/forum/#!forum/tfjs)

