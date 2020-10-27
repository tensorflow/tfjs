# AutoML Edge API

This packages provides a set of APIs to load and run models produced by AutoML
Edge.

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
2) [Object detection](#object-detection)

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

If you do not want (or cannot) load the model over HTTP you can also load the model separately and directly use the constuctor. 
This is particularly relevant for __non-browser__ platforms.

The following psuedocode demonstrates this approach:

```js
import * as automl from '@tensorflow/tfjs-automl';
import * as tf from '@tensorflow/tfjs';
// You can load the graph model using any IO handler
const graphModel = await tf.loadGraphModel(string|io.IOHandler); // a url or ioHandler instance
// You can load the dictionary using any api available to the platform
const dict = loadDictionary("path/to/dict.txt");
const model = new automl.ImageClassificationModel(graphModel, dict);
```

### Making a prediction

The AutoML library takes care of any image preprocessing
(normalize, resize, crop). The input `img` you provide can be
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
const options = {centerCrop: true};
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

### Advanced usage

Advanced users can access the underlying
[`GraphModel`](https://js.tensorflow.org/api/latest/#class:GraphModel) via
`model.graphModel`. The `GraphModel` allows users to call lower level methods
such as `predict()`, `execute()` and `executeAsync()` which return tensors.

`model.dictionary` gives you access to the ordered list of labels.

## Object detection

AutoML Object detection model will output the following set of files:
- `model.json`, the model topology
- `dict.txt`, a newline-separated list of labels
- One or more of `*.bin` files which hold the weights

Make sure you can access those files as static assets from your web app by serving them locally or on Google Cloud Storage.

### Demo

The object detection demo lives in
[demo/object_detection](./demo/object_detection). To run it:

```sh
cd demo/object_detection
yarn
yarn watch
```

This will start a local HTTP server on port 1234 that serves the demo.

### Loading the model
```js
import * as automl from '@tensorflow/tfjs-automl';
const modelUrl = 'model.json'; // URL to the model.json file.
const model = await automl.loadObjectDetection(modelUrl);
```

If you do not want (or cannot) load the model over HTTP you can also load the model separately and directly use the constuctor. 
This is particularly relevant for __non-browser__ platforms.

The following psuedocode demonstrates this approach:

```js
import * as automl from '@tensorflow/tfjs-automl';
import * as tf from '@tensorflow/tfjs';
// You can load the graph model using any IO handler
const graphModel = await tf.loadGraphModel(string|io.IOHandler); // a url or ioHandler instance
// You can load the dictionary using any api available to the platform
const dict = readDictionary("path/to/dict.txt");
const model = new automl.ObjectDetectionModel(graphModel, dict);
```

### Making a prediction

The AutoML library takes care of any image preprocessing
(normalize, resize, crop). The input `img` you provide can be
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
const options = {score: 0.5, iou: 0.5, topk: 20};
const predictions = await model.detect(img, options);
```

`options` is optional and has the following properties:
- `score` - Probability score between 0 and 1. Defaults to 0.5. Boxes with score lower than this threshold will be ignored.
- `topk` - Only the `topk` most likely objects are returned. The actual number of objects might be less than this number.
- `iou` - Intersection over union threshold. IoU is a metric between 0 and 1 used to measure the overlap of two boxes. The predicted boxes will not overlap more than the specified threshold.

The result `predictions` is a sorted list of predicted objects:

```js
[
  {
    box: {
      left: 105.1,
      top: 22.2,
      width: 70.6,
      height: 55.7
    },
    label: "Tomato",
    score: 0.972
  },
  ...
]
```

### Advanced usage

Advanced users can access the underlying
[`GraphModel`](https://js.tensorflow.org/api/latest/#class:GraphModel) via
`model.graphModel`. The `GraphModel` allows users to call lower level methods
such as `predict()`, `execute()` and `executeAsync()` which return tensors.

`model.dictionary` gives you access to the ordered list of labels.
