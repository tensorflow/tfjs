# Usage

This package adds a GPU accelerated [WebGPU](https://www.w3.org/TR/webgpu/)
backend to TensorFlow.js. It currently supports
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

Google Chrome started to support WebGPU by default in M113 on May 2, 2023.


## Importing the backend

### Via NPM

```js
// Import @tensorflow/tfjs or @tensorflow/tfjs-core
import * as tf from '@tensorflow/tfjs';
// Add the WebGPU backend to the global backend registry.
import '@tensorflow/tfjs-backend-webgpu';
// Set the backend to WebGPU and wait for the module to be ready.
tf.setBackend('webgpu').then(() => main());
```

### Via a script tag

```html
<!-- Import @tensorflow/tfjs or @tensorflow/tfjs-core -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js"> </script>

<!-- Add the WebGPU backend to the global backend registry -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgpu/dist/tf-backend-webgpu.js"></script>
<script>
// Set the backend to WebGPU and wait for the module to be ready
tf.setBackend('webgpu').then(() => main());
</script>
```

# FAQ

### When should I use the WebGPU backend?
The mission of WebGPU backend is to achieve the best performance among all
approaches. However, this target can not be met overnight, but we are committed
to supporting it with rapid and continuous performance improvement. Many
exciting features, like FP16, DP4A, will be brought in soon.

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
Currently the Canary channel of Chrome is used for testing of the WebGPU
backend:

```sh
yarn test  # --test_env=CHROME_CANARY_BIN=/path/to/chrome
```
