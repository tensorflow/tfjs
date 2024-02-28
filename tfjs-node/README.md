# TensorFlow backend for TensorFlow.js via Node.js
This repository provides native TensorFlow execution in backend JavaScript applications under the Node.js runtime,
accelerated by the TensorFlow C binary under the hood. It provides the same API as [TensorFlow.js](https://js.tensorflow.org/api/latest/).

This package will work on Linux, Windows, and Mac platforms where TensorFlow is supported.

## Installing

TensorFlow.js for Node currently supports the following platforms:
- Mac OS X CPU (10.12.6 Siera or higher)
- Linux CPU (Ubuntu 14.04 or higher)
- Linux GPU (Ubuntu 14.04 or higher and Cuda 11.2 w/ CUDNN v8) ([see installation instructions](https://www.tensorflow.org/install/gpu#software_requirements))
- Windows CPU (Win 7 or higher)
- Windows GPU (Win 7 or higher and Cuda 11.2 w/ CUDNN v8) ([see installation instructions](https://www.tensorflow.org/install/gpu#windows_setup))

For GPU support, tfjs-node-gpu@1.2.4 or later requires the following NVIDIA® software installed on your system:

| Name | Version |
| ------------- | ------------- |
| [NVIDIA® GPU drivers](https://www.nvidia.com/Download/index.aspx?lang=en-us) | >450.x  |
| [CUDA® Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)  | 11.2  |
| [cuDNN SDK](https://developer.nvidia.com/rdp/cudnn-download)  | 8.1.0  |

*Other Linux variants might also work but this project matches [core TensorFlow installation requirements](https://www.tensorflow.org/install/source).*

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

#### Windows / Mac OS X Requires Supported Version of Python

Windows & OSX build support for `node-gyp` requires that you have installed a [supported version of Python](https://devguide.python.org/versions/#supported-versions). Be sure to have supported version of Python before installing `@tensorflow/tfjs-node` or `@tensorflow/tfjs-node-gpu`.

*For more troubleshooting on Windows, check out [WINDOWS_TROUBLESHOOTING.md](./WINDOWS_TROUBLESHOOTING.md).*

#### Mac OS X Requires Xcode

If you do not have Xcode setup on your machine, please run the following commands:

```sh
$ xcode-select --install
```
For Mac OS Catalina please follow [this guide](https://github.com/nodejs/node-gyp/tree/main#on-macos) to install node-gyp.

After that operation completes, re-run `yarn add` or `npm install` for the `@tensorflow/tfjs-node` package.

You only need to include `@tensorflow/tfjs-node` or `@tensorflow/tfjs-node-gpu` in the package.json file, since those packages ship with `@tensorflow/tfjs` already.

#### Mac OS X with M1 chip
For Mac with M1 chip, tfjs-node only support arm64 build.
To install tfjs-node, you need to ensure rosetta has been turned off on your terminal app.
Start your terminal and verify following command shows `arm64` as response:
```
uname -m
```
Install your node version with arm64 binary. You can verify that with following command also shows `arm64`:
```
node -e 'console.log(os.arch())'
```
Now you can install tfjs-node as described before.

#### Rebuild the package on Raspberry Pi

To use this package on Raspberry Pi, you need to rebuild the node native addon with the following command after you installed the package:

```sh
$ npm rebuild @tensorflow/tfjs-node --build-from-source
```

#### Custom binaries URI

If you happen to be using a mirror for the libtensorflow binaries (default is [https://storage.googleapis.com/]), you have 3 options (in order of priority):

1. Set the environment variable `TFJS_NODE_CDN_STORAGE`. This has the same behavior as `CDN_STORAGE`, but introduced to prevent collisions with other npm packages that might use `CDN_STORAGE`.

```sh
TFJS_NODE_CDN_STORAGE="https://yourmirrorofchoice.com/" npm install <package>
(or)
TFJS_NODE_CDN_STORAGE="https://yourmirrorofchoice.com/" yarn install <package>
```

2. Add the variable `TFJS_NODE_CDN_STORAGE` to your `.npmrc` file.

```
TFJS_NODE_CDN_STORAGE=https://yourmirrorofchoice.com/
```

3. Set the environment variable `CDN_STORAGE`. This option is deprecated in favor of the `TFJS_NODE_` prefix version above and will be removed in a future release.

```sh
CDN_STORAGE="https://yourmirrorofchoice.com/" npm install <package>
(or)
CDN_STORAGE="https://yourmirrorofchoice.com/" yarn install <package>
```

If your "mirror" uses a custom URI path that doesn't match the default, you have 2 options (in order of priority):

1. Set the environment variable `TFJS_NODE_BASE_URI`

```sh
TFJS_NODE_BASE_URI="https://yourhost.com/your/path/libtensorflow-" npm install <package>
(or)
TFJS_NODE_BASE_URI="https://yourhost.com/your/path/libtensorflow-" yarn install <package>
```

2. Add the variable `TFJS_NODE_BASE_URI` to your `.npmrc` file

```
TFJS_NODE_BASE_URI=https://yourhost.com/your/path/libtensorflow-
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
