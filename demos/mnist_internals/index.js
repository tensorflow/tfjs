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
import * as tfvis from '../../src'
import {getModel, loadData} from './model';

window.tf = tf;
window.tfvis = tfvis;

window.data;
window.model;

async function initData() {
  window.data = await loadData();
  window.examples = data.nextTestBatch(10)

  showExamples(document.querySelector('#mnist-examples'), 200);
}

function initModel() {
  window.model = getModel();
}

async function showExamples(drawArea, numExamples) {
  // Get the examples
  const examples = data.nextTestBatch(numExamples);
  const tensorsToDispose = [];
  const drawPromises = [];
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      return examples.xs.slice([i, 0], [1, examples.xs.shape[1]]).reshape([
        28, 28, 1
      ]);
    });

    // Create a canvas element to render each example
    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = 'margin: 4px;';
    const drawPromise = tf.toPixels(imageTensor, canvas);
    drawArea.appendChild(canvas);

    tensorsToDispose.push(imageTensor);
    drawPromises.push(drawPromise);
  }

  await Promise.all(drawPromises);
  tf.dispose(tensorsToDispose);
}

function setupListeners() {
  document.querySelector('#load-data').addEventListener('click', async (e) => {
    await initData();
    document.querySelector('#start-training-1').disabled = false;
    e.target.disabled = true;
  });
}

document.addEventListener('DOMContentLoaded', function() {
  initModel();
  setupListeners();
});
