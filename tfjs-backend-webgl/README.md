# Usage

This package implements a GPU accelerated WebGL backend for TensorFlow.js.

## Importing the backend

Note: this backend is included by default in `@tensorflow/tfjs`.

### Via NPM

```js
// Import @tensorflow/tfjs-core
import * as tf from '@tensorflow/tfjs-core';
// Adds the WebGL backend to the global backend registry.
import '@tensorflow/tfjs-backend-webgl';
```

### Via a script tag

```html
<!-- Import @tensorflow/tfjs-core -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core"></script>

<!-- Adds the WebGL backend to the global backend registry -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl"></script>
```

You can also get ES2017 code using the following links

```html
<!-- Import @tensorflow/tfjs-core -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core@2.0.0-rc.4/dist/tf-core.es2017.js"></script>

<!-- Adds the WebGL backend to the global backend registry -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl@2.0.0-rc.4/dist/tf-backend-webgl.es2017.js"></script>
```

