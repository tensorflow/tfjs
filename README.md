# TensorFlow.js Data

**This repo is under active development and is not production-ready. We are
actively developing as an open source project.**

TensorFlow.js Data provides simple APIs to load and parse data from disk or over
the web in a variety of formats, and to prepare that data for use in machine
learning models (e.g. via operations like filter, map, shuffle, and batch).

This project is the JavaScript analogue of
[tf.data](https://www.tensorflow.org/get_started/datasets_quickstart) on the
Python/C++ side.  TF.js Data will match the tf.data API to the extent possible.

## Importing

There are two ways to import TensorFlow.js Data

1. You can access TensorFlow.js Data through the union package: [@tensorflow/tfjs](https://www.npmjs.com/package/@tensorflow/tfjs)
2. You can get TensorFlow.js Data as a module:
   [@tensorflow/tfjs-data](https://www.npmjs.com/package/@tensorflow/tfjs-data).
   Note that `tfjs-data` has peer dependency on tfjs-core, so if you import
   `@tensorflow/tfjs-data`, you also need to import
   `@tensorflow/tfjs-core`.

## Sample Usage

Reading a CSV file

```js
import * as tf from '@tensorflow/tfjs-data';

...

const csvFeatureDataset = await tf.data.csv(csv_feature_url, /* header */ true);
const csvTargetDataset = await tf.data.csv(csv_target_url, /* header */ true);

const numFeatures = csvFeatureDataset.csvColumnNames.length;

const trainDataset = await tf.zip([csvFeatureDataset, csvTargetDataset]).shuffle(100);

await model.fitDataset(trainDataset, modelFitDatasetConfig);
...
```

## For more information

- [TensorFlow.js API documentation](https://js.tensorflow.org/api/index.html)
- [TensorFlow.js Tutorials](https://js.tensorflow.org/tutorials/)
