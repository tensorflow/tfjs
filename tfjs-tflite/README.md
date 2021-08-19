# TFLite support for Tensorflow.js

_WORK IN PROGRESS_

This package enables users to run arbitary TFLite models on the web. Users can
load a TFLite model from a URL, use TFJS tensors to set the model's input
data, run inference, and get the output back in TFJS tensors. Under the hood,
the TFLite C++ runtime is packaged in a set of WASM modules, and the one with
the best performance will be automatically loaded based on user's current
environment (e.g. whether WebAssembly SIMD and/or multi-threading is supported
or not).

Check out this [demo][demo] where we use this package to run a
[CartoonGAN][model] TFLite model on the web.

# Usage

## Import the packages

To use this package, you will need a TFJS backend installed. We recommend the
CPU backend. You will also need to import `@tensorflow/tfjs-core` for
manipulating tensors.

### Via NPM

```js
// Adds the CPU backend.
import '@tensorflow/tfjs-backend-cpu';
// Import @tensorflow/tfjs-core
import * as tf from '@tensorflow/tfjs-core';
// Import @tensorflow/tfjs-tflite.
import {loadTFLiteModel, TFLiteModel} from '@tensorflow/tfjs-tflite';
```

### Via a script tag

```html
<!-- Import @tensorflow/tfjs-core -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core"></script>
<!-- Adds the CPU backend -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-cpu"></script>
<!--
  Import @tensorflow/tfjs-tflite

  Note that we need to explicitly load dist/tf-tflite.min.js so that it can
  locate WASM module files from their default location (dist/).
-->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite/dist/tf-tflite.min.js"></script>
```

## Set WASM modules location (optional)

By default, it will try to load the WASM modules from the same location where
the package or your own script is served. Use `setWasmPath` to set your own
location. See `src/tflite_web_api_client.d.ts` for more details.


```js
import {setWasmPath} from '@tensorflow/tfjs-tflite';

setWasmPath('https://your-server/path');
```

## Load a TFLite model
```js
const tfliteModel = await tflite.loadTFLiteModel('url/to/your/model.tflite');
```

## Run inference
```js
// Prepare input tensors.
const img = tf.browser.fromPixels(document.querySelector('img'));
const input = tf.sub(tf.div(tf.expandDims(img), 127.5), 1);

// Run inference and get output tensors.
let outputTensor = tfliteModel.predict(input) as tf.Tensor;
console.log(outputTensor.dataSync());
```

# Performance

Similar to TFJS WASM backend, this package uses [XNNPACK][xnnpack] to accelerate
model inference. To achieve the best performance, use a browser that supports
"WebAssembly SIMD" and "WebAssembly threads". In Chrome, these can be enabled
in `chrome://flags/`. As of March 2021, XNNPACK can only be enabled for
non-quantized TFLite models. Quantized models can still be used, but not
accelerated. Support for quantized model acceleration is in the works.

Setting the number of threads when calling `loadTFLiteModel` can also help with
the performance. In most cases, the threads count should be the same as the
number of physical cores, which is half of `navigator.hardwareConcurrency` on
many x86-64 processors.

```js
const tfliteModel = await loadTFLiteModel(
    'path/to/your/my_model.tflite',
    {numThreads: navigator.hardwareConcurrency / 2});
```

# Development

## Building

```sh
$ yarn
# This script will download the TFLite Web API WASM module files and JS client
# to deps/.
#
# The version number is optional. By default, the script will use the current
# version from the package.json file.
$ ./script/download-tflite-web-api.sh [version number]
$ yarn build
```

## Testing

```sh
$ yarn test
```

## Deployment
```sh
$ yarn build-npm
# (TODO): publish
```

[demo]: https://storage.googleapis.com/tfweb/demos/cartoonizer/index.html
[model]: https://blog.tensorflow.org/2020/09/how-to-create-cartoonizer-with-tf-lite.html
[xnnpack]: https://github.com/google/XNNPACK
