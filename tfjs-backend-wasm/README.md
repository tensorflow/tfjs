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

## Using bundlers

The shipped library on NPM consists of 2 files:
- the main js file (bundled js for browsers)
- the WebAssembly binary in `dist/tfjs-backend-wasm.wasm`

There is a [proposal](https://github.com/WebAssembly/esm-integration) to add
WASM support for ES6 modules. In the meantime, we have to manually read the wasm
file. When the WASM backend is initialized, we make a `fetch`/`readFile`
for `tfjs-backend-wasm.wasm` relative from the main js file. This means that
bundlers such as Parcel and WebPack need to be able to serve the `.wasm` file in
production. See [starter/parcel](./starter/parcel/) and
[starter/webpack](./starter/webpack/) for how to setup your favorite bundler.

If your server is serving the `.wasm` file on a different path or a different
name, use `setWasmPath` before you initialize the backend:

```ts
import {setWasmPath} from '@tensorflow/tfjs-backend-wasm';
setWasmPath(yourCustomPath); // or tf.wasm.setWasmPath when using <script> tags.
tf.setBackend('wasm').then(() => {...});
```

## Benchmarks

The benchmarks below show inference times (ms) for two different edge-friendly
models: MobileNet V2 (a medium-sized model) and Face Detector (a lite model).
All the benchmarks were run in Chrome 79.0 using
[this benchmark page](../tfjs-core/benchmarks/index.html) across our three
backends: CPU (vanilla JS), WebGL and WASM. Inference times are averaged
across 200 runs.

### MobileNet V2

MobileNet is a medium-sized model with 3.48M params and ~300M multiply-adds.
For this model, the WASM backend is between ~3X-11.5X faster than the vanilla
CPU backend, and ~5.3-7.7X slower than the WebGL backend.

<img src="./mobilenet-v2-bench.svg">

| MobileNet inference (ms) | WASM  | WebGL | CPU   |
|-------------------|-------|-------|-------|
| iPhone X          | 147.1 | 20.3  | 941.3 |
| iPhone XS         | 140   | 18.1  | 426.4 |
| Desktop Linux     | 91.5  | 17.1  | 1049  |
| Macbook Pro       |       |       |       |



### Face Detector

Face detector is a lite model with 0.1M params and ~20M multiply-adds. For this model,
the WASM backend is between ~8.2-19.8X faster than the vanilla CPU backend, and
only 1X-1.7X slower than the WebGL backend.

<img src="./face-detector-bench.svg">

| Face Detector inference (ms) | WASM | WebGL | CPU   |
|---------------|------|-------|-------|
| iPhone X      | 23   | 13.5  | 318   |
| iPhone XS     | 21.4 | 10.5  | 176.9 |
| Desktop Linux | 12.6 | 12.7  | 249.5 |
| Macbook Pro   |      |       |       |

# FAQ

### When should I use the WASM backend?
You should always try to use the WASM backend over the CPU backend
(which is implemented in vanilla js) since it is strictly faster on all devices,
across all model sizes.
Compared to the WebGL backend, the WASM backend has better numerical stability,
and wider device support. Performance-wise, our benchmarks show that:
- For medium-sized models (~100-500M multiply-adds), the WASM backend is several
times slower than the WebGL backend.
- For lite models (~20-60M multiply-adds), the WASM backend has comparable
performance to the WebGL backend
(see the [Face Detector model](#face-detector) above).

We are committed to supporting the WASM backend and will continue to improve
performance. We plan to follow the WebAssembly standard closely and benefit from
its upcoming features such as SIMD and multi-threading.

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
