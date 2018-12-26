<a id="travis-badge" href="https://travis-ci.org/tensorflow/tfjs-node" alt="Build Status">
  <img src="https://travis-ci.org/tensorflow/tfjs-node.svg?branch=master" />
</a>

# TensorFlow backend for TensorFlow.js via Node.js

**This repo is under active development and is not production-ready. We are
actively developing as an open source project.**

## Installing

TensorFlow.js for Node currently supports the following platforms:
- Mac OS X CPU (10.12.6 Siera or higher)
- Linux CPU (Ubuntu 14.04 or higher)
- Linux GPU (Ubuntu 14.04 or higher and Cuda 9.0 w/ CUDNN v7) ([see installation instructions](https://www.tensorflow.org/install/install_linux))
- Windows CPU (Win 7 or higher)
- Windows GPU (Win 7 or higher and Cuda 9.0 w/ CUDNN v7) ([see installation instructions](https://www.tensorflow.org/install/install_windows))

*Other Linux variants might also work but this project matches [core TensorFlow installation requirements](https://www.tensorflow.org/install/install_linux).*

#### Installing CPU TensorFlow.js for Node:

```sh
npm install @tensorflow/tfjs-node
(or)
yarn add @tensorflow/tfjs-node
```

#### Installing Linux/Windows GPU TensorFlow.js for Node:

```sh
npm install @tensorflow/tfjs-node-gpu
(or)
yarn add @tensorflow/tfjs-node-gpu
```

#### Windows Requires Python 2.7

Windows build support for `node-gyp` requires Python 2.7. Be sure to have this version before installing `@tensorflow/tfjs-node` or `@tensorflow/tfjs-node-gpu`. Machines with Python 3.x will not install the bindings properly.

*For more troubleshooting on Windows, check out [WINDOWS_TROUBLESHOOTING.md](./WINDOWS_TROUBLESHOOTING.md).*

#### Mac OS X Requires Xcode

If you do not have Xcode setup on your machine, please run the following commands:

```sh
$ xcode-select --install
```

After that operation completes, re-run `yarn add` or `npm install` for the `@tensorflow/tfjs-node` package.

You only need to include `@tensorflow/tfjs-node` or `@tensorflow/tfjs-node-gpu` in the package.json file, since those packages ship with `@tensorflow/tfjs` already.

## Using the binding

Before executing any TensorFlow.js code, import the node package:

```js
// Load the binding
import * as tf from '@tensorflow/tfjs-node';

// Or if running with GPU:
import * as tf from '@tensorflow/tfjs-node-gpu';
```

Note: you do not need to add the `@tensorflow/tfjs` package to your dependencies or import it directly.

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


## MNIST demo for Node.js

See the [tfjs-examples repository](https://github.com/tensorflow/tfjs-examples/tree/master/mnist-node) for training the MNIST dataset using the Node.js bindings.

### Optional: Build libtensorflow From TensorFlow source

This requires installing bazel first.

```sh
bazel build --config=monolithic //tensorflow/tools/lib_package:libtensorflow
```
