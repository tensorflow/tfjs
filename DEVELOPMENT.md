## Development

This repository contains only the logic and scripts that combine
two packages:
- [TensorFlow.js Core](https://github.com/tensorflow/tfjs-core),
  a flexible low-level API, formerly known as *deeplearn.js*.
- [TensorFlow.js Layers](https://github.com/tensorflow/tfjs-layers),
  a high-level API which implements functionality similar to
  [Keras](https://keras.io/).

To develop:
- `tfjs-core`, see [this doc](https://github.com/tensorflow/tfjs-core/blob/master/DEVELOPMENT.md).
- `tfjs-layers` see [this doc](https://github.com/tensorflow/tfjs-layers/blob/master/DEVELOPMENT.md).
- `tfjs` with locally modified `tfjs-layers` (or `tfjs-core`) see [this section](https://github.com/tensorflow/tfjs-layers/blob/master/DEVELOPMENT.md#changing-tensorflowtfjs-layers-and-testing-tensorflowtfjs) in [tfjs-layers/DEVELOPMENT.md](https://github.com/tensorflow/tfjs-layers) repo.
