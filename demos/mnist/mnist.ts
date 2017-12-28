/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

// tslint:disable-next-line:max-line-length
import {Array1D, Array2D, CheckpointLoader, ENV, NDArray, Scalar} from 'deeplearn';

// manifest.json lives in the same directory as the mnist demo.
const reader = new CheckpointLoader('.');
reader.getAllVariables().then(vars => {
  // Get sample data.
  const xhr = new XMLHttpRequest();
  xhr.open('GET', 'sample_data.json');
  xhr.onload = async () => {
    const data = JSON.parse(xhr.responseText) as SampleData;
    const math = ENV.math;

    // Wrap everything in a math.scope so we clean up intermediate NDArrays.
    math.scope(async () => {
      console.log(`Evaluation set: n=${data.images.length}.`);

      let numCorrect = 0;
      for (let i = 0; i < data.images.length; i++) {
        const x = Array1D.new(data.images[i]);

        // Infer through the model to get a prediction.
        const predictedLabel = Math.round(await infer(x, vars).val());
        console.log(`Item ${i}, predicted label ${predictedLabel}.`);

        // Aggregate correctness to show accuracy.
        const label = data.labels[i];
        if (label === predictedLabel) {
          numCorrect++;
        }

        // Show the image.
        const result =
            renderResults(Array1D.new(data.images[i]), label, predictedLabel);
        document.body.appendChild(result);
      }

      // Compute final accuracy.
      const accuracy = numCorrect * 100 / data.images.length;
      document.getElementById('accuracy').innerHTML = `${accuracy}%`;
    });
  };
  xhr.onerror = (err) => console.error(err);
  xhr.send();
});

export interface SampleData {
  images: number[][];
  labels: number[];
}

/**
 * Infers through a 3-layer fully connected MNIST model using the Math API. This
 * is the lowest level user-facing API in deeplearn.js giving the most control
 * to the user. Math commands execute immediately, like numpy.
 */
export function infer(
    x: Array1D, vars: {[varName: string]: NDArray}): Scalar<'int32'> {
  const hidden1W = vars['hidden1/weights'] as Array2D;
  const hidden1B = vars['hidden1/biases'] as Array1D;
  const hidden2W = vars['hidden2/weights'] as Array2D;
  const hidden2B = vars['hidden2/biases'] as Array1D;
  const softmaxW = vars['softmax_linear/weights'] as Array2D;
  const softmaxB = vars['softmax_linear/biases'] as Array1D;
  const math = ENV.math;
  const hidden1 =
      math.relu(math.add(math.vectorTimesMatrix(x, hidden1W), hidden1B)) as
      Array1D;
  const hidden2 =
      math.relu(math.add(
          math.vectorTimesMatrix(hidden1, hidden2W), hidden2B)) as Array1D;

  const logits = math.add(math.vectorTimesMatrix(hidden2, softmaxW), softmaxB);

  return math.argMax(logits);
}

function renderMnistImage(array: Array1D) {
  const width = 28;
  const height = 28;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const float32Array = array.dataSync();
  const imageData = ctx.createImageData(width, height);
  for (let i = 0; i < float32Array.length; ++i) {
    const j = i * 4;
    const value = Math.round(float32Array[i] * 255);
    imageData.data[j + 0] = value;
    imageData.data[j + 1] = value;
    imageData.data[j + 2] = value;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

function renderResults(array: Array1D, label: number, predictedLabel: number) {
  const root = document.createElement('div');
  root.appendChild(renderMnistImage(array));
  const actual = document.createElement('div');
  actual.innerHTML = `Actual: ${label}`;
  root.appendChild(actual);
  const predicted = document.createElement('div');
  predicted.innerHTML = `Predicted: ${predictedLabel}`;
  root.appendChild(predicted);

  if (label !== predictedLabel) {
    root.classList.add('error');
  }

  root.classList.add('result');
  return root;
}
