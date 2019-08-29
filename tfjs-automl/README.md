# AutoML Edge API

This packages provides a set of APIs to load and run models produced by AutoML Edge.

## Status

__Early development__. This is still an unpublished experimental package.


## Installation

If you are using npm/yarn
```sh
npm i @tensorflow/tfjs-automl
```

If you are using CDN:
```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-automl"></script>
```

Currently we support loading 2 types of models:
1) Image classification
2) Object detection

## Image classification

### Demo

The image classification demo lives in [demo/img_classification](./demo/img_classification). To run it:

```sh
cd demo/img_classification
yarn
yarn watch
```

This will start a local HTTP server on port 1234 that serves the demo.

### Loading the model
```js
import * as automl from '@tensorflow/tfjs-automl';
// URL to the model.json file produced by AutoML.
const modelUrl = 'model.json';
const model = automl.loadImageClassification(modelUrl);
```

### Making a prediction
The input `img` can be `HTMLImageElement`, `HTMLVideoElement`, `ImageData` or a 3D `Tensor`:

```js
const img = document.getElementById('image-tag');
const predictions = model.classify(img, options);
```

`options` is optional and currently has the following properties:
- `options.centerCrop` - Defaults to true. Since the ML model expects a square image, we need to either resize of crop the image. If true, the image will be cropped to a central square.

The result `predictions` is a sorted list of predicted labels and their probabilities:
```js
[
  {label: "daisy", prob: 0.931},
  {label: "dandelion", prob: 0.027},
  {label: "roses", prob: 0.013},
  ...
]
```

## Object detection

TODO(smilkov): Write this when you get a model.
