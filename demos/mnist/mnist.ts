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
import {Array1D, Array2D, CheckpointLoader, Graph, NDArray, NDArrayInitializer, NDArrayMath, NDArrayMathGPU, Scalar, Session, Tensor} from '../deeplearnjs';

// manifest.json lives in the same directory as the mnist demo.
const reader = new CheckpointLoader('.');
reader.getAllVariables().then(vars => {
  // Get sample data.
  const xhr = new XMLHttpRequest();
  xhr.open('GET', 'sample_data.json');
  xhr.onload = () => {
    const data = JSON.parse(xhr.responseText) as SampleData;
    const math = new NDArrayMathGPU();
    const [input, probs] = buildModelLayersAPI(data, vars);
    const sess = new Session(input.node.graph, math);

    math.scope(() => {
      console.log(`Evaluation set: n=${data.images.length}.`);

      let numCorrect = 0;
      for (let i = 0; i < data.images.length; i++) {
        const inputData = Array1D.new(data.images[i]);
        const probsVal = sess.eval(probs, [{tensor: input, data: inputData}]);
        console.log(`Item ${i}, probsVal ${probsVal.get()}.`);
        const label = data.labels[i];
        const predictedLabel = probsVal.get();
        if (label === predictedLabel) {
          numCorrect++;
        }
        const result =
            renderResults(Array1D.new(data.images[i]), label, predictedLabel);
        document.body.appendChild(result);
      }
      const accuracy = numCorrect * 100 / data.images.length;
      document.getElementById('accuracy').innerHTML = accuracy + '%';
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
 * Builds a 3-layer fully connected MNIST model using the Math API. This is the
 * lowest level user-facing API in Learn.js giving the most control to the user.
 * Math commands execute immediately, like numpy. Math commands are wrapped in
 * math.scope() so that NDArrays created by intermediate math commands are
 * automatically cleaned up.
 */
export function buildModelMathAPI(
    math: NDArrayMath, data: SampleData,
    vars: {[varName: string]: NDArray}): (x: Array1D) => Scalar {
  const hidden1W = vars['hidden1/weights'] as Array2D;
  const hidden1B = vars['hidden1/biases'] as Array1D;
  const hidden2W = vars['hidden2/weights'] as Array2D;
  const hidden2B = vars['hidden2/biases'] as Array1D;
  const softmaxW = vars['softmax_linear/weights'] as Array2D;
  const softmaxB = vars['softmax_linear/biases'] as Array1D;

  return (x: Array1D): Scalar => {
    return math.scope(() => {
      const hidden1 = math.relu(
          math.add(math.vectorTimesMatrix(x, hidden1W), hidden1B)) as Array1D;
      const hidden2 = math.relu(math.add(
          math.vectorTimesMatrix(hidden1, hidden2W), hidden2B)) as Array1D;
      const logits =
          math.add(math.vectorTimesMatrix(hidden2, softmaxW), softmaxB);
      return math.argMax(logits);
    });
  };
}

/**
 * Builds a 3-layers fully connected MNIST model using the Graph API. This API
 * mimics the TensorFlow API, providing a lazy execution with feeds and fetches.
 * Users do not need to worry about GPU-related memory leaks, other than their
 * input data.
 */
export function buildModelGraphAPI(
    data: SampleData, vars: {[varName: string]: NDArray}): Tensor[] {
  const g = new Graph();
  // TODO: Support batching.
  const input = g.placeholder('input', [784]);
  const hidden1W = g.constant(vars['hidden1/weights']);
  const hidden1B = g.constant(vars['hidden1/biases']);
  const hidden1 = g.relu(g.add(g.matmul(input, hidden1W), hidden1B));

  const hidden2W = g.constant(vars['hidden2/weights']);
  const hidden2B = g.constant(vars['hidden2/biases']);
  const hidden2 = g.relu(g.add(g.matmul(hidden1, hidden2W), hidden2B));

  const softmaxW = g.constant(vars['softmax_linear/weights']);
  const softmaxB = g.constant(vars['softmax_linear/biases']);
  const logits = g.add(g.matmul(hidden2, softmaxW), softmaxB);
  return [input, g.argmax(logits)];
}

/**
 * Builds a 3-layers fully connected MNIST model using the Graph API in
 * conjuction with `Graph.layers`, which mimics the Keras layers API.
 */
function buildModelLayersAPI(
    data: SampleData, vars: {[varName: string]: NDArray}): Tensor[] {
  const g = new Graph();
  // TODO: Support batching.
  const input = g.placeholder('input', [784]);
  const hidden1W = vars['hidden1/weights'];
  const hidden1B = vars['hidden1/biases'];
  const hidden1 = g.layers.dense(
      'hidden1', input, hidden1W.shape[1], (x) => g.relu(x), true,
      new NDArrayInitializer(hidden1W), new NDArrayInitializer(hidden1B));

  const hidden2W = vars['hidden2/weights'];
  const hidden2B = vars['hidden2/biases'];
  const hidden2 = g.layers.dense(
      'hidden2', hidden1, hidden2W.shape[1], (x) => g.relu(x), true,
      new NDArrayInitializer(hidden2W), new NDArrayInitializer(hidden2B));

  const softmaxW = vars['softmax_linear/weights'];
  const softmaxB = vars['softmax_linear/biases'];
  const logits = g.layers.dense(
      'softmax', hidden2, softmaxW.shape[1], null, true,
      new NDArrayInitializer(softmaxW), new NDArrayInitializer(softmaxB));
  return [input, g.argmax(logits)];
}

function renderMnistImage(array: Array1D) {
  console.log('renderMnistImage', array);
  const width = 28;
  const height = 28;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const float32Array = array.getData().values;
  const imageData = ctx.createImageData(width, height);
  for (let i = 0; i < float32Array.length; i++) {
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
