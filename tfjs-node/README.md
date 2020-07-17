# TensorFlow backend for TensorFlow.js via Node.js
This repository provides native TensorFlow execution in backend JavaScript applications under the Node.js runtime,
accelerated by the TensorFlow C binary under the hood. It provides the same API as [TensorFlow.js](https://js.tensorflow.org/api/latest/).

This package will work on Linux, Windows, and Mac platforms where TensorFlow is supported.

## Installing

TensorFlow.js for Node currently supports the following platforms:
- Mac OS X CPU (10.12.6 Siera or higher)
- Linux CPU (Ubuntu 14.04 or higher)
- Linux GPU (Ubuntu 14.04 or higher and Cuda 10.0 w/ CUDNN v7) ([see installation instructions](https://www.tensorflow.org/install/gpu#software_requirements))
- Windows CPU (Win 7 or higher)
- Windows GPU (Win 7 or higher and Cuda 10.0 w/ CUDNN v7) ([see installation instructions](https://www.tensorflow.org/install/gpu#windows_setup))

For GPU support, tfjs-node-gpu@1.2.4 or later requires the following NVIDIA® software installed on your system:

| Name | Version |
| ------------- | ------------- |
| [NVIDIA® GPU drivers](https://www.nvidia.com/Download/index.aspx?lang=en-us) | >410.x  |
| [CUDA® Toolkit](https://developer.nvidia.com/cuda-10.0-download-archive)  | 10.0  |
| [cuDNN SDK](https://developer.nvidia.com/rdp/cudnn-download)  | >=7.4.1  |

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

#### Windows / Mac OS X Requires Python 2.7

Windows & OSX build support for `node-gyp` requires Python 2.7. Be sure to have this version before installing `@tensorflow/tfjs-node` or `@tensorflow/tfjs-node-gpu`. Machines with Python 3.x will not install the bindings properly.

*For more troubleshooting on Windows, check out [WINDOWS_TROUBLESHOOTING.md](./WINDOWS_TROUBLESHOOTING.md).*

#### Mac OS X Requires Xcode

If you do not have Xcode setup on your machine, please run the following commands:

```sh
$ xcode-select --install
```
For Mac OS Catalina please follow [this guide](https://github.com/nodejs/node-gyp/blob/master/macOS_Catalina.md#installing-node-gyp-using-the-xcode-command-line-tools-via-manual-download) to install node-gyp.

After that operation completes, re-run `yarn add` or `npm install` for the `@tensorflow/tfjs-node` package.

You only need to include `@tensorflow/tfjs-node` or `@tensorflow/tfjs-node-gpu` in the package.json file, since those packages ship with `@tensorflow/tfjs` already.

#### Rebuild the package on Raspberry Pi

To use this package on Raspberry Pi, you need to rebuild the node native addon with the following command after you installed the package:

```sh
$ npm rebuild @tensorflow/tfjs-node --build-from-source
```

## Using the binding

Before executing any TensorFlow.js code, import the node package:

```js
// Load the binding
const tf = require('@tensorflow/tfjs-node');

// Or if running with GPU:
const tf = require('@tensorflow/tfjs-node-gpu');
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

### Optional: Build optimal TensorFlow from source

To get the most optimal TensorFlow build that can take advantage of your specific hardware (AVX512, MKL-DNN), you can build the `libtensorflow` library from source:
- [Install bazel](https://docs.bazel.build/versions/master/install.html)
- Checkout the [main tensorflow repo](https://github.com/tensorflow/tensorflow) and follow the instructions in [here](https://www.tensorflow.org/install/source) with **one difference**: instead of building the pip package, build `libtensorflow`:

```sh
./configure
bazel build --config=opt --config=monolithic //tensorflow/tools/lib_package:libtensorflow
```

The build might take a while and will produce a `bazel-bin/tensorflow/tools/lib_package/libtensorflow.tar.gz` file, which should be unpacked and replace the files in `deps` folder of `tfjs-node` repo:
```sh
cp bazel-bin/tensorflow/tools/lib_package/libtensorflow.tar.gz ~/myproject/node_modules/@tensorflow/tfjs-node/deps
cd path-to-my-project/node_modules/@tensorflow/tfjs-node/deps
tar -xf libtensorflow.tar.gz
```

If you want to publish an addon library with your own libtensorflow binary, you can host the custom libtensorflow binary and optional pre-compiled node addon module on the cloud service you choose, and add a `custom-binary.json` file in `scripts` folder with the following information:

```js
{
  "tf-lib": "url-to-download-customized-binary",
  "addon": {
    "host": "host-of-pre-compiled-addon",
    "remote_path": "remote-path-of-pre-compiled-addon",
    "package_name": "file-name-of-pre-compile-addon"
  }
}
```

The installation scripts will automatically catch this file and use the custom libtensorflow binary and addon. If `addon` is not provided, the installation script will compile addon from source.
