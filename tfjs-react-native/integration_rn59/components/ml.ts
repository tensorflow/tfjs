/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as mobilenet from '@tensorflow-models/mobilenet';
import * as tf from '@tensorflow/tfjs';
import {asyncStorageIO, bundleResourceIO} from '@tensorflow/tfjs-react-native';

// All functions (i.e. 'runners") in this file are async
// functions that return a function that can be invoked to
// do some ML operation.

/**
 * A runner for a simple math op
 */
export async function simpleOpRunner() {
  return async () => {
    const res = tf.square(3);
    const data = (await res.data())[0];
    return JSON.stringify(data);
  };
}

/**
 * A runner that does a basic precision test.
 */
export async function precisionTestRunner() {
  return async () => {
    const res = tf.tidy(() => tf.scalar(2.4).square());
    const data = (await res.data())[0];
    return JSON.stringify(data);
  };
}

/**
 * A runner that does a mobilenet prediction
 */
export async function mobilenetRunner() {
  const model = await mobilenet.load();
  // warmup
  const input = tf.zeros([1, 224, 224, 3]);
  await model.classify(input);

  return async () => {
    const pred = await model.classify(input);
    return JSON.stringify(pred);
  };
}

/**
 * A runner that loads a model bundled with the app and runs a prediction
 * through it.
 */
const modelJson = require('../assets/model/bundle_model_test.json');
const modelWeights = require('../assets/model/bundle_model_test_weights.bin');
export async function localModelRunner() {
  const model =
      await tf.loadLayersModel(bundleResourceIO(modelJson, modelWeights));

  return async () => {
    const res = model.predict(tf.randomNormal([1, 10])) as tf.Tensor;
    const data = await res.data();
    return JSON.stringify(data);
  };
}

/**
 * A runner that loads a model bundled with the app and runs a prediction
 * through it.
 */
const modelJson2 = require('../assets/graph_model/model.json');
const modelWeights2 = require('../assets/graph_model/group1-shard1of1.bin');
export async function localGraphModelRunner() {
  const model =
      await tf.loadGraphModel(bundleResourceIO(modelJson2, modelWeights2));
  return async () => {
    const res = model.predict(tf.randomNormal([1, 10])) as tf.Tensor;
    const data = await res.data();
    return JSON.stringify(data);
  };
}

/**
 * A runner that traines a model.
 */
export async function trainModelRunner() {
  // Define a model for linear regression.
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 5, inputShape: [1]}));
  model.add(tf.layers.dense({units: 1}));
  model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

  // Generate some synthetic data for training.
  const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
  const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

  return async () => {
    // Train the model using the data.
    await model.fit(xs, ys, {epochs: 20});

    return 'done';
  };
}

/**
 * A runner that saves and loads a model to/from asyncStorage
 */
export async function saveModelRunner() {
  // Define a model for linear regression.
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 5, inputShape: [1]}));
  model.add(tf.layers.dense({units: 1}));
  model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

  return async () => {
    await model.save(asyncStorageIO('custom-model-test'));
    await tf.loadLayersModel(asyncStorageIO('custom-model-test'));

    return 'done';
  };
}
