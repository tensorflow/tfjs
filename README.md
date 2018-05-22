# TensorFlow backend for TensorFlow.js via Node.js

**This repo is under active development and is not production-ready. We are
actively developing as an open source project.**

## Installing

TensorFlow.js for Node currently supports the following platforms:
- Mac OS X 10.12.6 (Siera) or higher
- Linux CPU (Ubuntu 16.04 or higher)
- Linux GPU (Ubuntu 16.04 or higher and Cuda 9.0 w/ CUDNN v7) ([see installation instructions](https://www.tensorflow.org/install/install_linux))

*Other Linux variants might also work but this project matches [core TensorFlow installation requirements](https://www.tensorflow.org/install/install_linux).*

#### Installing CPU TensorFlow.js for Node:

```sh
npm install @tensorflow/tfjs-node
(or)
yarn add @tensorflow/tfjs-node
```

#### Installing Linux GPU TensorFlow.js for Node:

```sh
npm install @tensorflow/tfjs-node-gpu
(or)
yarn add @tensorflow/tfjs-node-gpu
```

Before executing any TensorFlow.js code, load and set the backend to 'tensorflow'.

```js
import * as tf from '@tensorflow/tfjs';

// Load the binding
import '@tensorflow/tfjs-node';

// Or if running with GPU:
import '@tensorflow/tfjs-node-gpu';

tf.setBackend('tensorflow');
```

## Development

```sh
# Download and install JS dependencies, including libtensorflow 1.8.
yarn

# Run TFJS tests against Node.js backend:
yarn test
```

```sh
# Switch to GPU for local development:
yarn enable-gpu
```

See the `demo` directory that trains MNIST using TensorFlow.js with the
TensorFlow C backend.

```sh
cd demo/
yarn

# Run the training script. See demo/package.json for this script.
yarn mnist
```

The important line to note is at the top of `mnist.ts`, which sets the backend to
TensorFlow.

### Optional: Build libtensorflow From TensorFlow source

This requires installing bazel first.

```sh
bazel build --config=monolithic //tensorflow/tools/lib_package:libtensorflow
```
