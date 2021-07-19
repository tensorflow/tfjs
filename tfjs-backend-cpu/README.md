# Usage

This package implements a JavaScript based CPU backend to TensorFlow.js.

## Importing the backend

Note: this backend is included by default in `@tensorflow/tfjs`.

### Via NPM

```js
// Import @tensorflow/tfjs-core
import * as tf from '@tensorflow/tfjs-core';
// Adds the CPU backend to the global backend registry.
import '@tensorflow/tfjs-backend-cpu';
```

### Via a script tag

```html
<!-- Import @tensorflow/tfjs-core -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core"></script>

<!-- Adds the CPU backend to the global backend registry -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-cpu"></script>
```

You can also get ES2017 code using the following links

```html
<!-- Import @tensorflow/tfjs-core -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core@2.0.0-rc.4/dist/tf-core.es2017.js"></script>

<!-- Adds the CPU backend to the global backend registry -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-cpu@2.0.0-rc.4/dist/tf-backend-cpu.es2017.js"></script>
```

