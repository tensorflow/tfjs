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
import * as tflite from '@tensorflow/tfjs-tflite';
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
tflite.setWasmPath('https://your-server/path');
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

This package uses [XNNPACK][xnnpack] to accelerate inference for floating-point
and quantized models. See [XNNPACK documentation][xnnpack doc] for the full list
of supported floating-point and quantized operators.

To achieve the best performance, use a browser that supports
"WebAssembly SIMD" and "WebAssembly threads". In Chrome 92+, these features are
enabled by default. In older versions of Chrome, they can be enabled in
`chrome://flags/`.

Starting from Chrome 92, **cross-origin isolation** needs to be set up in your
site in order to take advantage of the multi-threading support. Without this, it
will fallback to the WASM binary with SIMD-only support (or the vanila version
if SIMD is not enabled). Without multi-threading support, certain models might
not achieve the best performance. See [here][cross origin setup steps] for the
high-level steps to set up the cross-origin isolation.

By default, the runtime uses the number of physical cores as the thread count.
You can tune this number by setting the `numThreads` option when loading the
TFLite model:

```js
const tfliteModel = await tflite.loadTFLiteModel(
    'path/to/your/my_model.tflite',
    {numThreads: navigator.hardwareConcurrency / 2});
```

# Profiling

Profiling can be enabled by setting the `enableProfiling` option to true when
loading the TFLite model:

```js
const tfliteModel = await tflite.loadTFLiteModel(
    'path/to/your/my_model.tflite',
    {enableProfiling: true});
```

Once it is enabled, the runtime will record per-op latency data when the
`predict` method is called. The profiling results can be retrieved in two ways:

- `tfliteModel.getProfilingResults()`: this method will return an array of
  `{nodeType, nodeName, execTimeInMs}`.
- `tfliteModel.getProfilingSummary()`: this method will return a human-readable
  profiling result summary that looks like [this][profiling summary].

# Development

## Building

```sh
$ yarn
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
[xnnpack doc]: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/xnnpack/README.md#limitations-and-supported-operators
[cross origin setup steps]: https://github.com/tensorflow/tfjs/tree/master/tfjs-backend-wasm#setting-up-cross-origin-isolation
[profiling summary]: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/benchmark/README.md#profiling-model-operators
