# TensorFlow.js

TensorFlow.js is an open-source hardware-accelerated JavaScript library for
training and deploying machine learning models.

> :warning: We recently released **TensorFlow.js 2.0**. If you have been using TensorFlow.js
> via a script tag without specifying a version and see an error saying no backends
> are found, then you should read our [release notes](https://github.com/tensorflow/tfjs/releases/tag/tfjs-v2.0.0)
> for instructions on how to upgrade.


**Develop ML in the Browser** <br/>
Use flexible and intuitive APIs to build models from scratch using the low-level
JavaScript linear algebra library or the high-level layers API.

**Develop ML in Node.js** <br/>
Execute native TensorFlow with the same TensorFlow.js API under the Node.js
runtime.

**Run Existing models** <br/>
Use TensorFlow.js model converters to run pre-existing TensorFlow models right
in the browser.

**Retrain Existing models** <br/>
Retrain pre-existing ML models using sensor data connected to the browser or
other client-side data.

## About this repo

This repository contains the logic and scripts that combine
several packages.

APIs:
- [TensorFlow.js Core](/tfjs-core),
  a flexible low-level API for neural networks and numerical computation.
- [TensorFlow.js Layers](/tfjs-layers),
  a high-level API which implements functionality similar to
  [Keras](https://keras.io/).
- [TensorFlow.js Data](/tfjs-data),
  a simple API to load and prepare data analogous to
  [tf.data](https://www.tensorflow.org/guide/datasets).
- [TensorFlow.js Converter](/tfjs-converter),
  tools to import a TensorFlow SavedModel to TensorFlow.js
- [TensorFlow.js Vis](/tfjs-vis),
  in-browser visualization for TensorFlow.js models
- [TensorFlow.js AutoML](/tfjs-automl),
  Set of APIs to load and run models produced by
  [AutoML Edge](https://cloud.google.com/vision/automl/docs/edge-quickstart).


Backends/Platforms:
- [TensorFlow.js CPU Backend](/tfjs-backend-cpu), Node backend via TensorFlow C++.
- [TensorFlow.js WebGL Bakend](/tfjs-backend-webgl), Node backend via TensorFlow C++.
- [TensorFlow.js Node](/tfjs-node), Node backend via TensorFlow C++.
- [TensorFlow.js WASM](/tfjs-backend-wasm), WebAssembly backend.
- [TensorFlow.js React Native](/tfjs-react-native), React Native backend/platform adapter.
- [TensorFlow.js WebGPU](/tfjs-backend-webgpu), WebGPU backend.

If you care about bundle size, you can import those packages individually.

If you are looking for Node.js support, check out the [TensorFlow.js Node directory](/tfjs-node).

## Examples

Check out our
[examples repository](https://github.com/tensorflow/tfjs-examples)
and our [tutorials](https://js.tensorflow.org/tutorials/).

## Gallery

Be sure to check out [the gallery](GALLERY.md) of all projects related to TensorFlow.js.

## Pre-trained models

Be sure to also check out our [models repository](https://github.com/tensorflow/tfjs-models) where we host pre-trained models
on NPM.

## Benchmarks

[Benchmark tool](https://tensorflow.github.io/tfjs/e2e/benchmarks/). Use this webpage tool to test the performance related metrics (speed, memory, power, etc) of TensorFlow.js models on your local device with CPU, WebGL or WASM backend.

## Getting started

There are two main ways to get TensorFlow.js in your JavaScript project:
via <a href="https://developer.mozilla.org/en-US/docs/Learn/HTML/Howto/Use_JavaScript_within_a_webpage" target="_blank">script tags</a> <strong>or</strong> by installing it from <a href="https://www.npmjs.com/" target="_blank">NPM</a>
and using a build tool like <a href="https://parceljs.org/" target="_blank">Parcel</a>,
<a href="https://webpack.js.org/" target="_blank">WebPack</a>, or <a href="https://rollupjs.org/guide/en" target="_blank">Rollup</a>.

### via Script Tag

Add the following code to an HTML file:

```html
<html>
  <head>
    <!-- Load TensorFlow.js -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js"> </script>


    <!-- Place your code in the script tag below. You can also use an external .js file -->
    <script>
      // Notice there is no 'import' statement. 'tf' is available on the index-page
      // because of the script tag above.

      // Define a model for linear regression.
      const model = tf.sequential();
      model.add(tf.layers.dense({units: 1, inputShape: [1]}));

      // Prepare the model for training: Specify the loss and the optimizer.
      model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

      // Generate some synthetic data for training.
      const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
      const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

      // Train the model using the data.
      model.fit(xs, ys).then(() => {
        // Use the model to do inference on a data point the model hasn't seen before:
        // Open the browser devtools to see the output
        model.predict(tf.tensor2d([5], [1, 1])).print();
      });
    </script>
  </head>

  <body>
  </body>
</html>
```

Open up that HTML file in your browser, and the code should run!

### via NPM

Add TensorFlow.js to your project using <a href="https://yarnpkg.com/en/" target="_blank">yarn</a> <em>or</em> <a href="https://docs.npmjs.com/cli/npm" target="_blank">npm</a>. <b>Note:</b> Because
we use ES2017 syntax (such as `import`), this workflow assumes you are using a modern browser or a bundler/transpiler
to convert your code to something older browsers understand. See our
<a href='https://github.com/tensorflow/tfjs-examples' target="_blank">examples</a>
to see how we use <a href="https://parceljs.org/" target="_blank">Parcel</a> to build
our code. However, you are free to use any build tool that you prefer.



```js
import * as tf from '@tensorflow/tfjs';

// Define a model for linear regression.
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// Prepare the model for training: Specify the loss and the optimizer.
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Generate some synthetic data for training.
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// Train the model using the data.
model.fit(xs, ys).then(() => {
  // Use the model to do inference on a data point the model hasn't seen before:
  model.predict(tf.tensor2d([5], [1, 1])).print();
});
```

See our <a href="https://js.tensorflow.org/tutorials/" target="_blank">tutorials</a>, <a href="https://github.com/tensorflow/tfjs-examples" target="_blank">examples</a>
and <a href="https://js.tensorflow.org/api/latest/">documentation</a> for more details.

## Importing pre-trained models

We support porting pre-trained models from:
- [TensorFlow SavedModel](https://github.com/tensorflow/tfjs-converter)
- [Keras](https://js.tensorflow.org/tutorials/import-keras.html)

## Find out more

[TensorFlow.js](https://js.tensorflow.org) is a part of the
[TensorFlow](https://www.tensorflow.org) ecosystem. For more info:
- For help from the community, use [`tensorflow.js`](https://stackoverflow.com/questions/tagged/tensorflow.js) tag on Stack Overflow.
- [js.tensorflow.org](https://js.tensorflow.org)
- [Tutorials](https://js.tensorflow.org/tutorials)
- [API reference](https://js.tensorflow.org/api/latest/)
- [Discussion mailing list](https://groups.google.com/a/tensorflow.org/forum/#!forum/tfjs)
- [TensorFlow.js Blog](https://blog.tensorflow.org/search?label=TensorFlow.js)

Thanks, <a href="https://www.browserstack.com/">BrowserStack</a>, for providing testing support.
