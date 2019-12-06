# Usage

This package adds a WebAssembly backend to TensorFlow.js. This is currently in
**alpha** and subject to change. Not every op in TensorFlow.js is supported on this
backend.

Importing this package augments the TensorFlow.js package
(@tensorflow/tfjs-core) by registering a new backend meaning existing
TensorFlow.js code, models, and dependent packages will work with only a few
lines of code changed.

## Importing the backend

### Via NPM

```js
// Import @tensorflow/tfjs or @tensorflow/tfjs-core
import * as tf from '@tensorflow/tfjs';

// Import the WASM backend.
import '@tensorflow/tfjs-backend-wasm';
```

### Via a script tag

```html
<!-- Import @tensorflow/tfjs or @tensorflow/tfjs-core -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>

<!-- Import the WASM backend. -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm"></script>
```

## Using the backend with MobileNet

```js
async function main() {
  // Set the backend to WASM and wait for the module to be ready.
  await tf.setBackend('wasm');

  let img = tf.browser.fromPixels(document.getElementById('img'))
      .resizeBilinear([224, 224])
      .expandDims(0)
      .toFloat();

  let model = await tf.loadGraphModel(
    'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2',
    {fromTFHub: true});
  const y = model.predict(img);

  y.print();
}
main();
```

Our WASM backend builds on top of the
[XNNPACK library](https://github.com/google/XNNPACK) which provides
high-efficiency floating-point neural network inference operators.

# FAQ

### How many ops have you implemented?
See [`all_kernels.ts`](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/src/kernels/all_kernels.ts)
for an up-to-date list of supported ops. We love contributions. See the
[contributing](https://github.com/tensorflow/tfjs/blob/master/CONTRIBUTING.md#adding-functionality)
document for more info.

### Do you support training?
Maybe. There are still a decent number of ops that we are missing in WASM that
are needed for gradient computation. At this point we are focused on making
inference as fast as possible.

### Do you work in node?
Yes. If you run into issues, please let us know.

### Do you support SIMD?
We are actively working on adding SIMD before we do the official release.
The switch to SIMD should happen transparently for browsers that support it.

### Do you support multi-threading?
Multi-threading support is not a priority for us at this point since it is still
a proposal. We will keep a close eye on it as the proposal progresses.

# Development

## Emscripten installation

Install the Emscripten SDK (version 1.39.1):

```sh
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install 1.39.1
./emsdk activate 1.39.1
```

## Prepare the environment

Before developing, make sure the environment variable `EMSDK` points to the
emscripten directory (e.g. `~/emsdk`). Emscripten provides a script that does
the setup for you:

Cd into the emsdk directory and run:

```sh
source ./emsdk_env.sh
```

For details, see instructions
[here](https://emscripten.org/docs/getting_started/downloads.html#installation-instructions).

## Building

```sh
yarn build
```

## Testing

```sh
yarn test
```

## Deployment
```sh
./scripts/build-npm.sh
npm publish
```
