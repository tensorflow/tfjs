# Usage

This package adds a GPU accelerated [WebGPU](https://www.w3.org/TR/webgpu/) backend to TensorFlow.js. It currently supports
the following models:
- BlazeFace
- BodyPix
- Face landmarks detection
- HandPose
- MobileNet
- PoseDetection
- Universal sentence encoder
- AutoML Image classification
- AutoML Object detection
- Speech commands

Note that WebGPU 1.0 hasn't been officially released by W3C group. Currently WebGPU in chrome is in Origin Trial. To try webgpu features, you need to launch chrome canary browser with flag --enable-unsafe-webgpu.


## Importing the backend

### Via NPM

```js
// Import @tensorflow/tfjs or @tensorflow/tfjs-core
import * as tf from '@tensorflow/tfjs';
// Adds the WebGPU backend to the global backend registry.
import '@tensorflow/tfjs-backend-webgpu';
// Set the backend to WebGPU and wait for the module to be ready.
tf.setBackend('webgpu').then(() => main());
```

### Via a script tag

```html
<!-- Import @tensorflow/tfjs or @tensorflow/tfjs-core -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js"> </script>

<!-- Adds the WebGPU backend to the global backend registry -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgpu/dist/tf-backend-webgpu.js"></script>
<script>
tf.setBackend('webgpu').then(() => main());
</script>
```

# FAQ

### When should I use the WebGPU backend?
WebGPU as the successor of WebGL will provide a hardware accelerated solution.
Theoretically, webgpu is recommended to be used in any scenario used by WebGL
and provide comparable or better performance.

We are committed to supporting the WebGPU backend and will continue to improve
performance. We plan to follow the WebGPU standard closely and benefit from
its upcoming features such as fp16, int8.

### How many ops have you implemented?
See [`register_all_kernels.ts`](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-webgpu/src/register_all_kernels.ts)
for an up-to-date list of supported ops. We love contributions. See the
[contributing](https://github.com/tensorflow/tfjs/blob/master/CONTRIBUTING.md#adding-functionality)
document for more info.

### Do you support training?
Maybe. There are still a decent number of ops that we are missing in WebGPU that
are needed for gradient computation. At this point we are focused on making
inference as fast as possible.

### Do you work in node?
Yes. If you run into issues, please let us know.

### How do I give feedback?
We'd love your feedback as we develop this backend! Please file an issue
[here](https://github.com/tensorflow/tfjs/issues/new).

# Development

## Building

```sh
yarn build
```

## Testing
The `$CHROME_BIN` environment variable must be set to the location of the Chrome Canary application.

e.g.
`export CHROME_BIN="/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary"`

```sh
yarn test
```
