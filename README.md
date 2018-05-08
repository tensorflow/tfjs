# TensorFlow backend for TensorFlow.js via Node.js

**This repo is under active development and is not production-ready. We are
actively developing as an open source project.**

## Installing 

```sh
npm install @tensorflow/tfjs-node
(or)
yarn add @tensorflow/tfjs-node
```

Before executing any TensorFlow.js code, load and set the backend to 'tensorflow'.

```js
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';

tf.setBackend('tensorflow');
```

## Development

```sh
# Download and install JS dependencies, including libtensorflow 1.8.
yarn

# Publish the NPM locally for usage with other packages.
yarn publish-local
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
bazel build //tensorflow/tools/lib_package:libtensorflow
```

## Supported Platforms

- Mac OS
- Linux
- ***Windows coming soon***
