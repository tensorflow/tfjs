<a id="travis-badge" href="https://travis-ci.org/tensorflow/tfjs-core" alt="Build Status">
  <img src="https://travis-ci.org/tensorflow/tfjs-core.svg?branch=master" />
</a>

# TensorFlow.js Core API

> NOTE: Building on the momentum of deeplearn.js, we have joined the TensorFlow
family and we are starting a new ecosystem of libraries and tools for Machine
Learning in Javascript, called [TensorFlow.js](https://js.tensorflow.org).
This repo moved from `PAIR-code/deeplearnjs` to `tensorflow/tfjs-core`.

A part of the TensorFlow.js ecosystem, this repo hosts `@tensorflow/tfjs-core`,
the TensorFlow.js Core API, which provides low-level, hardware-accelerated
linear algebra operations and an eager API for automatic differentiation.

Check out [js.tensorflow.org](https://js.tensorflow.org) for more
information about the library, tutorials and API docs.

To keep track of issues, we use the [tensorflow/tfjs](https://github.com/tensorflow/tfjs) Github repo.

## Importing

You can install TensorFlow.js via yarn or npm. We recommend using the
[@tensorflow/tfjs](https://www.npmjs.com/package/@tensorflow/tfjs) npm package,
which gives you both this Core API and the higher-level
[Layers API](https://github.com/tensorflow/tfjs-layers):

```js
import * as tf from '@tensorflow/tfjs';
// You have the Core API: tf.matMul(), tf.softmax(), ...
// You also have Layers API: tf.model(), tf.layers.dense(), ...
```

On the other hand, if you care about the bundle size and you do not use the
Layers API, you can import only the Core API:

```js
import * as tfc from '@tensorflow/tfjs-core';
// You have the Core API: tfc.matMul(), tfc.softmax(), ...
// No Layers API.
```

For info about development, check out [DEVELOPMENT.md](./DEVELOPMENT.md).

## For more information

- [TensorFlow.js API documentation](https://js.tensorflow.org/api/latest/)
- [TensorFlow.js Tutorials](https://js.tensorflow.org/tutorials/)

Thanks <a href="https://www.browserstack.com/">BrowserStack</a> for providing testing support.
