# Tensorflow Decision Forests support for Tensorflow.js

This package enables users to run arbitrary Tensorflow Decision Forests models
on the web that are converted using tfjs-converter.
Users can load a TFDF model from a URL, use TFJS tensors to set
the model's input data, run inference, and get the output back in TFJS tensors.
Under the hood, the TFDF C++ runtime is packaged in a set of WASM modules.

# Usage

## Import the packages

To use this package, you will need a TFJS backend installed.
You will also need to import `@tensorflow/tfjs-core` for
manipulating tensors, and `@tensorflow/tfjs-converter` for loading models.

### Via NPM

```js
// Import @tensorflow/tfjs-core
import * as tf from '@tensorflow/tfjs-core';
// Adds the CPU backend.
import '@tensorflow/tfjs-backend-cpu';
// Import @tensorflow/tfjs-converter
import * as tf from '@tensorflow/tfjs-converter';
// Import @tensorflow/tfjs-tfdf.
import * as tfdf from '@tensorflow/tfjs-tfdf';
```

### Via a script tag

```html
<!-- Import @tensorflow/tfjs-core -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core"></script>
<!-- Adds the CPU backend -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-cpu"></script>
<!-- Import @tensorflow/tfjs-converter -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-converter"></script>
<!--
  Import @tensorflow/tfjs-tfdf

  Note that we need to explicitly load dist/tf-tfdf.min.js so that it can
  locate WASM module files from their default location (dist/).
-->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tfdf/dist/tf-tfdf.min.js"></script>
```

## Set WASM modules location (optional)

By default, it will try to load the WASM modules from the same location where
the package or your own script is served. Use `setLocateFile` to set your own
location. See `src/tfdf_web_api_client.d.ts` for more details.

```js
// `base` is the URL to the main javascript file's directory.
// To return the default URL of the file use `${base}${path}`.
tfdf.setLocateFile((path, base) => {
  return `https://your-server/.../${path}`;
});
```

## Load a TFDF model
```js
const tfdfModel = await tfdf.loadTFDFModel('url/to/your/model.json');
```

## Run inference
```js
// Prepare input tensors.
const input = tf.tensor1d(['test', 'strings']);

// Run inference and get output tensors.
const outputTensor = await tfdfModel.executeAsync(input);
console.log(outputTensor.dataSync());
```

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
```
