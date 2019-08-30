# AutoML Edge API

This packages provides a set of APIs to load and run models produced by AutoML
Edge.

## Status

__Early development__. This is an unpublished experimental package.


## Installation

If you are using npm/yarn
```sh
npm i @tensorflow/tfjs-automl
```

If you are using CDN:
```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-automl"></script>
```

We support the following types of AutoML Edge models:
1) [Image classification](#image-classification)
2) **[In progress]** [Object detection](#object-detection)

## Image classification

AutoML Image classification model will output the following set of files:
- `model.json`, the model topology
- `dict.txt`, a newline-separated list of labels
- One or more of `*.bin` files which hold the weights

Make sure you can access those files as static assets from your web app by serving them locally or on Google Cloud Storage.

### Demo

The image classification demo lives in
[demo/img_classification](./demo/img_classification). To run it:

```sh
cd demo/img_classification
yarn
yarn watch
```

This will start a local HTTP server on port 1234 that serves the demo.

### Loading the model
```js
import * as automl from '@tensorflow/tfjs-automl';
const modelUrl = 'model.json'; // URL to the model.json file.
const model = await automl.loadImageClassification(modelUrl);
```

### Making a prediction
The input `img` can be
[`HTMLImageElement`](https://developer.mozilla.org/en-US/docs/Web/API/HTMLImageElement),
[`HTMLCanvasElement`](https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement),
[`HTMLVideoElement`](https://developer.mozilla.org/en-US/docs/Web/API/HTMLVideoElement),
[`ImageData`](https://developer.mozilla.org/en-US/docs/Web/API/ImageData) or
a 3D [`Tensor`](https://js.tensorflow.org/api/latest/#class:Tensor):

```html
<img id="img" src="PATH_TO_IMAGE" />
```

```js
const img = document.getElementById('img');
const options = {};
const predictions = await model.classify(img, options);
```

`options` is optional and has the following properties:
- `centerCrop` - Defaults to true. Since the ML model expects a square image,
we need to resize. If true, the image will be cropped first to the center before
resizing.

The result `predictions` is a sorted list of predicted labels and their
probabilities:

```js
[
  {label: "daisy", prob: 0.931},
  {label: "dandelion", prob: 0.027},
  {label: "roses", prob: 0.013},
  ...
]
```

## Object detection

TODO(smilkov): Write this when object detection is ready.
