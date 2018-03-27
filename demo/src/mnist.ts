/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import * as dl from 'deeplearn';
import * as tf from 'tfjs-node';

import {MnistDataset} from './mnist_data';
import {Timer} from './timer';

tf.bindTensorFlowBackend();

const HIDDEN_1 = 128;
const HIDDEN_2 = 32;

const NUM_CLASSES = 10;
const IMAGE_SIZE = 28;
const IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE;

// first layer weights
const weights1 =
    dl.variable(dl.truncatedNormal(
        [IMAGE_PIXELS, HIDDEN_1], null, 1.0 / Math.sqrt(IMAGE_PIXELS))) as
    dl.Tensor2D;
const biases1 = dl.zeros([HIDDEN_1]);

// second layer weights
const weights2 =
    dl.variable(dl.truncatedNormal(
        [HIDDEN_1, HIDDEN_2], null, 1.0 / Math.sqrt(HIDDEN_1))) as dl.Tensor2D;
const biases2 = dl.zeros([HIDDEN_2]);

// third layer weights
const weights3 =
    dl.variable(dl.truncatedNormal(
        [HIDDEN_2, NUM_CLASSES], null, 1.0 / Math.sqrt(HIDDEN_2))) as
    dl.Tensor2D;
const biases3 = dl.zeros([NUM_CLASSES]);

// Hyperparameters.
const LEARNING_RATE = .1;
const BATCH_SIZE = 100;
const TRAIN_STEPS = 2000;

const optimizer = dl.train.sgd(LEARNING_RATE);

function model(inputImages: dl.Tensor2D): dl.Tensor2D {
  const hidden1 = dl.matMul(inputImages, weights1).add(biases1).relu() as dl.Tensor2D;
  const hidden2 = dl.matMul(hidden1, weights2).add(biases2).relu() as dl.Tensor2D;
  return dl.matMul(hidden2, weights3).add(biases3) as dl.Tensor2D;
}

function loss(labels: dl.Tensor2D, ys: dl.Tensor2D): dl.Scalar {
  return dl.losses.softmaxCrossEntropy(labels, ys).mean() as dl.Scalar;
}

async function runTraining() {
  const data = new MnistDataset();
  const timer = new Timer();

  console.log('  * Loading training data...');
  timer.start();
  await data.loadData();
  timer.end();
  console.log(
      `  * Loaded training data in : ${timer.seconds().toFixed(3)} secs`);

  let trainSecs = 0;
  console.log('  * Starting Training...');
  const totalTimer = new Timer();
  totalTimer.start();
  for (let i = 0; i < TRAIN_STEPS; i++) {
    const fetchCost = i % 100 === 0;

    if (!data.hasMoreData()) {
      // TODO(kreeger): Change data class to return random data.
      data.reset();
    }

    timer.start();
    const cost = optimizer.minimize(() => {
      const batch = data.nextTrainBatch(BATCH_SIZE);
      return loss(batch.label, model(batch.image));
    }, fetchCost);
    timer.end();

    trainSecs += timer.seconds();

    if (fetchCost) {
      console.log(`Step ${i}: loss = ${cost.dataSync()} in ${
          timer.seconds().toFixed(3)} secs`);
    }
  }
  totalTimer.end();
  console.log(`  * Trained in ${totalTimer.seconds().toFixed(3)} secs`);
  console.log(`  * Average train step time: ${
      (trainSecs / TRAIN_STEPS).toFixed(3)} secs`);
}

runTraining();
