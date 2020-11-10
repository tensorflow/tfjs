# TensorFlow.js Core API

A part of the TensorFlow.js ecosystem, this repo hosts `@tensorflow/tfjs-core`,
the TensorFlow.js Core API, which provides low-level, hardware-accelerated
linear algebra operations and an eager API for automatic differentiation.

Check out [js.tensorflow.org](https://js.tensorflow.org) for more
information about the library, tutorials and API docs.

To keep track of issues we use the [tensorflow/tfjs](https://github.com/tensorflow/tfjs) Github repo.

## Importing

You can install TensorFlow.js via yarn or npm. We recommend using the
[@tensorflow/tfjs](https://www.npmjs.com/package/@tensorflow/tfjs) npm package,
which gives you both this Core API and the higher-level
[Layers API](/tfjs-layers):

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

**Note**: If you are only importing the Core API, you also need to import a
backend (e.g., [tfjs-backend-cpu](/tfjs-backend-cpu),
[tfjs-backend-webgl](/tfjs-backend-webgl), [tfjs-backend-wasm](/tfjs-backend-wasm)).

For info about development, check out [DEVELOPMENT.md](/DEVELOPMENT.md).

## For more information

- [TensorFlow.js API documentation](https://js.tensorflow.org/api/latest/)
- [TensorFlow.js Tutorials](https://js.tensorflow.org/tutorials/)

Thanks <a href="https://www.browserstack.com/">BrowserStack</a> for providing testing support.
