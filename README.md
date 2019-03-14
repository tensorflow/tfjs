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
import * as tf from '@tensorflow/tfjs';

const csvUrl = 'https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/boston-housing-train.csv';

async function run() {
  // We want to predict the column "medv", which represents a median value of a
  // home (in $1000s), so we mark it as a label.
  const csvDataset = tf.data.csv(
    csvUrl, {
      columnConfigs: {
        medv: {
          isLabel: true
        }
      }
    });
  // Number of features is the number of column names minus one for the label
  // column.
  const numOfFeatures = (await csvDataset.columnNames()).length - 1;

  // Prepare the Dataset for training.
  const flattenedDataset =
    csvDataset
    .map(({xs, ys}) => {
      // Convert xs(features) and ys(labels) from object form (keyed by column
      // name) to array form.
      return {xs: Object.values(xs), ys: Object.values(ys)};
    })
    .batch(10);

  // Define the model.
  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [numOfFeatures],
    units: 1
  }));
  model.compile({
    optimizer: tf.train.sgd(0.000001),
    loss: 'meanSquaredError'
  });

  // Fit the model using the prepared Dataset
  return model.fitDataset(flattenedDataset, {
    epochs: 10,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(epoch, logs.loss);
      }
    }
  });
}

run().then(() => console.log('Done'));
```

## For more information

- [TensorFlow.js API documentation](https://js.tensorflow.org/api/latest/)
- [TensorFlow.js Tutorials](https://js.tensorflow.org/tutorials/)
