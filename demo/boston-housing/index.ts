/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import * as tf from '@tensorflow/tfjs';
// TODO(kangyi, soergel): Remove this once we have a public statistics API.
import {computeDatasetStatistics, DatasetStatistics} from '@tensorflow/tfjs-data/dist/statistics';
import {BostonHousingDataset} from './data';
import * as ui from './ui';

// Some hyperparameters for model training.
const NUM_EPOCHS = 250;
const BATCH_SIZE = 40;
const LEARNING_RATE = 0.01;

interface PreparedData {
  normalizedTrainFeatures: tf.Tensor2D;
  trainTarget: tf.Tensor2D;
  normalizedTestFeatures: tf.Tensor2D;
  testTarget: tf.Tensor2D;
}

const preparedData: PreparedData = {
  normalizedTrainFeatures: null,
  trainTarget: null,
  normalizedTestFeatures: null,
  testTarget: null
};

let bostonData: BostonHousingDataset;
let stats: DatasetStatistics;

// TODO(kangyizhang): Remove this function when model.fitDataset(dataset) is
//  available. This work should be done by dataset class itself.

// Converts loaded data into tensors and creates normalized versions of the
// features.
export async function loadDataAndNormalize() {
  // TODO(kangyizhang): Statistics should be generated from trainDataset
  // directly. Update following codes after
  // https://github.com/tensorflow/tfjs-data/issues/32 is resolved.

  // Gets mean and standard deviation of data.
  stats = await computeDatasetStatistics(await bostonData.trainDataset.map(
      (row: {features: {key: number}, target: {key: number}}) => row.features));

  // Normalizes features data.
  const normalizedTrainData = bostonData.trainDataset.map(normalizeFeatures);
  const normalizedTestData = bostonData.testDataset.map(normalizeFeatures);

  // Materializes data into arrays. Following codes should be removed once
  // model.fitDataset is available.
  const trainIter = await normalizedTrainData.iterator();
  const trainData = await trainIter.collect();
  const testIter = await normalizedTestData.iterator();
  const testData = await testIter.collect();

  preparedData.normalizedTrainFeatures = tf.tensor2d(trainData.map(
      (row: {normalizedFeatures: number[], target: number[]}) =>
          row.normalizedFeatures));
  preparedData.trainTarget = tf.tensor2d(trainData.map(
      (row: {normalizedFeatures: number[], target: number[]}) => row.target));
  preparedData.normalizedTestFeatures = tf.tensor2d(testData.map(
      (row: {normalizedFeatures: number[], target: number[]}) =>
          row.normalizedFeatures));
  preparedData.testTarget = tf.tensor2d(testData.map(
      (row: {normalizedFeatures: number[], target: number[]}) => row.target));
}

/**
 * Normalizes features with statistics and returns a new object.
 */
function normalizeFeatures(row: {features: number[], target: number[]}) {
  const features = row.features;
  const normalizedFeatures: number[] = [];
  features.forEach(
      (value, index) => normalizedFeatures.push(
          (value - stats[index].mean) / stats[index].stddev));
  return {normalizedFeatures, target: row.target};
}

/**
 * Builds and returns Linear Regression Model.
 *
 * @returns {tf.Sequential} The linear regression model.
 */
export const linearRegressionModel = (): tf.Sequential => {
  const model = tf.sequential();
  model.add(tf.layers.dense({inputShape: [bostonData.numFeatures], units: 1}));

  return model;
};

/**
 * Builds and returns Multi Layer Perceptron Regression Model
 * with 2 hidden layers, each with 10 units activated by sigmoid.
 *
 * @returns {tf.Sequential} The multi layer perceptron regression model.
 */
export const multiLayerPerceptronRegressionModel = (): tf.Sequential => {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [bostonData.numFeatures],
    units: 50,
    activation: 'sigmoid'
  }));
  model.add(tf.layers.dense({units: 50, activation: 'sigmoid'}));
  model.add(tf.layers.dense({units: 1}));

  return model;
};

/**
 * Compiles `model` and trains it using the train data and runs model against
 * test data. Issues a callback to update the UI after each epcoh.
 *
 * @param {tf.Sequential} model Model to be trained.
 */
export const run = async (model: tf.Sequential) => {
  await ui.updateStatus('Compiling model...');
  model.compile(
      {optimizer: tf.train.sgd(LEARNING_RATE), loss: 'meanSquaredError'});

  let trainLoss: number;
  let valLoss: number;
  await ui.updateStatus('Starting training process...');
  await model.fit(
      preparedData.normalizedTrainFeatures, preparedData.trainTarget, {
        batchSize: BATCH_SIZE,
        epochs: NUM_EPOCHS,
        validationSplit: 0.2,
        callbacks: {
          onEpochEnd: async (epoch: number, logs) => {
            await ui.updateStatus(
                `Epoch ${epoch + 1} of ${NUM_EPOCHS} completed.`);
            trainLoss = logs.loss;
            valLoss = logs.val_loss;
            await ui.plotData(epoch, trainLoss, valLoss);
          }
        }
      });

  await ui.updateStatus('Running on test data...');
  const result =
      model.evaluate(
          preparedData.normalizedTestFeatures, preparedData.testTarget,
          {batchSize: BATCH_SIZE}) as tf.Tensor;
  const testLoss = result.dataSync()[0];
  await ui.updateStatus(
      `Final train-set loss: ${trainLoss.toFixed(4)}\n` +
      `Final validation-set loss: ${valLoss.toFixed(4)}\n` +
      `Test-set loss: ${testLoss.toFixed(4)}`);
};

export const computeBaseline = () => {
  const avgPrice = tf.mean(preparedData.trainTarget);
  console.log(`Average price: ${avgPrice.dataSync()}`);
  const baseline =
      tf.mean(tf.pow(tf.sub(preparedData.testTarget, avgPrice), 2));
  console.log(`Baseline loss: ${baseline.dataSync()}`);
  const baselineMsg = `Baseline loss (meanSquaredError) is ${
      baseline.dataSync()[0].toFixed(2)}`;
  ui.updateBaselineStatus(baselineMsg);
};

document.addEventListener('DOMContentLoaded', async () => {
  bostonData = await BostonHousingDataset.create();
  ui.updateStatus('Data loaded, converting to tensors');
  await loadDataAndNormalize();
  ui.updateStatus(
      'Data is now available as tensors.\n' +
      'Click a train button to begin.');
  ui.updateBaselineStatus('Estimating baseline loss');
  computeBaseline();
  await ui.setup();
}, false);
