# tfjs-vis

__tfjs-vis__ is a small library for _in browser_ visualization intended for use
with TensorFlow.js.

Its main features are:

* A set of visualizations useful for visualizing model behaviour
* A set of high level functions for visualizing objects specific to TensorFlow.js
* A way to organize visualizations (the visor) of model behaviour that won't interfere with your web application

The library also aims to be flexible and make it easy for you to incorporate
custom visualizations using tools of your choosing, such as d3, Chart.js or plotly.js.

## Example Screenshots

### Training Metrics

![Training metrics (loss and accuracy) for a model](https://storage.googleapis.com/tfjs-assets/tfjs-vis/tfjs-vis-training.png)

### Model Evauation

![Dataset accuracy metrics in a table and confusion matrix visualization](https://storage.googleapis.com/tfjs-assets/tfjs-vis/tfjs-vis-evaluation.png)

### Model Internals

![Model summary table and histogram of conv2d weights](https://storage.googleapis.com/tfjs-assets/tfjs-vis/tfjs-vis-model-details.png)

### Activations and custom visualizations

![visualization of dataset activations in a conv2d layer and a dense layer](https://storage.googleapis.com/tfjs-assets/tfjs-vis/tfjs-vis-model-internals.png)


## Demos

- [Visualizing Training with tfjs-vis](https://storage.googleapis.com/tfjs-vis/mnist/dist/index.html)
- [Looking inside a digit recognizer](https://storage.googleapis.com/tfjs-vis/mnist_internals/dist/index.html)

## Installation

You can install this using npm with

```
npm install @tensorflow/tfjs-vis
```

or using yarn with

```
yarn add @tensorflow/tfjs-vis
```

You can also load it via script tag using the following tag, however you need
to have TensorFlow.js also loaded on the page to work. Including both is shown
below.

```
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"> </script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis"></script>
```

## API

See https://js.tensorflow.org/api_vis/latest/ for interactive API documentation.

## Sample Usage

```js
const data = [
  { index: 0, value: 50 },
  { index: 1, value: 100 },
  { index: 2, value: 150 },
];

// Get a surface
const surface = tfvis.visor().surface({ name: 'Barchart', tab: 'Charts' });

// Render a barchart on that surface
tfvis.render.barchart(surface, data, {});
```

## Issues

Found a bug or have a feature request? Please file an [issue](https://github.com/tensorflow/tfjs/issues/new) on the main [TensorFlow.js repository](https://github.com/tensorflow/tfjs/issues)

## Building from source

To build the library, you need to have node.js installed. We use `yarn`
instead of `npm` but you can use either.

First install dependencies with

```
yarn
```

or

```
npm install
```

Then do a build with

```
yarn build
```

or

```
npm run build
```

This should produce a `tfjs-vis.umd.min.js` file in the `dist` folder that you can
use.


