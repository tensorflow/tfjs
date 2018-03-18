# TensorFlow.js: Union Package

TensorFlow.js is a JavaScript library for building, training and serving
machine learning models. When running in the browser, it utilizes WebGL
acceleration. TensorFlow.js is a part of the
[TensorFlow](https://www.tensorflow.org) ecosystem.
You can import pre-trained TensorFlow
[SavedModels](https://www.tensorflow.org/programmers_guide/saved_model) and
[Keras models](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model),
for execution and retraining.

For more information on the API, follow the links to their Core and Layers
repositories below, or visit [js.tensorflow.org](https://js.tensorflow.org).

This repository contains the logic and scripts to form a **union** package,
[@tensorflowjs/tfjs](https://www.npmjs.com/package/@tensorflow/tfjs), from

- [TensorFlow.js Core](https://github.com/tensorflow/tfjs-core),
  a flexible low-level API, formerly known as *deeplearn.js*.
- [TensorFlow.js Layers](https://github.com/tensorflow/tfjs-layers),
  a high-level API modeled after [Keras](https://keras.io/).
