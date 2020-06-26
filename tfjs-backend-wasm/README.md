# Usage

This package adds a WebAssembly backend to TensorFlow.js. This is currently in
**alpha** and has enough op support to run the following models
from our [models](https://github.com/tensorflow/tfjs-models) repo:
- MobileNet
- BodyPix
- PoseNet
- CocoSSD
- AutoML Image classification
- AutoML Object detection

## Importing the backend

### Via NPM

```js
// Import @tensorflow/tfjs or @tensorflow/tfjs-core
import * as tf from '@tensorflow/tfjs';
// Adds the WASM backend to the global backend registry.
import '@tensorflow/tfjs-backend-wasm';
// Set the backend to WASM and wait for the module to be ready.
tf.setBackend('wasm').then(() => main());
```

### Via a script tag

```html
<!-- Import @tensorflow/tfjs or @tensorflow/tfjs-core -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>

<!-- Adds the WASM backend to the global backend registry -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm/dist/tf-backend-wasm.js"></script>
<script>
tf.setBackend('wasm').then(() => main());
</script>
```

## Running MobileNet

```js
async function main() {
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

If you are using platform that does not support fetch directly, please set the
optional `usePlatformFetch` to true:

```ts
import {setWasmPath} from '@tensorflow/tfjs-backend-wasm';
const usePlatformFetch = true;
setWasmPath(yourCustomPath, usePlatformFetch); // or tf.wasm.setWasmPath when using <script> tags.
tf.setBackend('wasm').then(() => {...});
```
## Benchmarks

The benchmarks below show inference times (ms) for two different edge-friendly
models: MobileNet V2 (a medium-sized model) and Face Detector (a lite model).
All the benchmarks were run in Chrome 79.0 using
[this benchmark page](../tfjs-core/benchmarks/index.html) across our three
backends: Plain JS (CPU), WebGL and WASM. Inference times are averaged
across 200 runs.

### MobileNet V2

MobileNet is a medium-sized model with 3.48M params and ~300M multiply-adds.
For this model, the WASM backend is between ~3X-11.5X faster than the plain
JS backend, and ~5.3-7.7X slower than the WebGL backend.

<img src="./mobilenet-v2-bench.svg">

| MobileNet inference (ms) | WASM  | WebGL | Plain JS | WASM + SIMD |
|--------------------------|-------|-------|----------|-------------|
| iPhone X                 | 147.1 | 20.3  | 941.3    | N/A         |
| iPhone XS                | 140   | 18.1  | 426.4    | N/A         |
| Pixel 3                  | 266.2 | 77.3  | 2345.2   | N/A         |
| Desktop Linux            | 91.5  | 17.1  | 1049     | N/A         |
| Desktop Windows          | 123.1 | 41.6  | 1117     | 37.2        |
| Macbook Pro              | 98.4  | 19.6  | 893.5    | 30.2        |



### Face Detector

Face detector is a lite model with 0.1M params and ~20M multiply-adds. For this model,
the WASM backend is between ~8.2-19.8X faster than the plain JS backend and
comparable to the WebGL backend (up to ~1.7X faster, or 2X slower, depending on
the device).

<img src="./face-detector-bench.svg">

| Face Detector inference (ms) | WASM | WebGL | Plain JS | WASM + SIMD |
|------------------------------|------|-------|----------|-------------|
| iPhone X                     | 22.4 | 13.5  | 318      | N/A         |
| iPhone XS                    | 21.4 | 10.5  | 176.9    | N/A         |
| Pixel 3                      | 40.7 | 31.8  | 535.2    | N/A         |
| Desktop Linux                | 12.6 | 12.7  | 249.5    | N/A         |
| Desktop Windows              | 16.2 | 7.1   | 270.9    | 7.5         |
| Macbook Pro 15 2019          | 13.6 | 22.7  | 209.1    | 7.9         |

# FAQ

### When should I use the WASM backend?
You should always try to use the WASM backend over the plain JS backend since
it is strictly faster on all devices, across all model sizes.
Compared to the WebGL backend, the WASM backend has better numerical stability,
and wider device support. Performance-wise, our benchmarks show that:
- For medium-sized models (~100-500M multiply-adds), the WASM backend is several
times slower than the WebGL backend.
- For lite models (~20-60M multiply-adds), the WASM backend has comparable
performance to the WebGL backend
(see the [Face Detector model](#face-detector) above).

We are committed to supporting the WASM backend and will continue to improve
performance. We plan to follow the WebAssembly standard closely and benefit from
its upcoming features such as multi-threading.

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
Yes. We take advantage of SIMD wherever it is supported. If you intend to serve the WASM assets yourself, note that the SIMD-enabled WASM binary is separate from the default binary.

### Do you support multi-threading?
Multi-threading support is not a priority for us at this point since it is still
a proposal. We will keep a close eye on it as the proposal progresses.

### How do I give feedback?
We'd love your feedback as we develop this backend! Please file an issue
[here](https://github.com/tensorflow/tfjs/issues/new).

# Development

## Emscripten installation

Install the Emscripten SDK (version 1.39.15):

```sh
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install 1.39.15
./emsdk activate 1.39.15
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
