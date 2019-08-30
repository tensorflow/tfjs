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

import * as tf from '@tensorflow/tfjs-core';
import {Tensor3D, test_util} from '@tensorflow/tfjs-core';
import {BROWSER_ENVS, describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

import * as automl from './index';
import {ClassificationPrediction} from './types';

const MODEL_URL =
    'https://storage.googleapis.com/tfjs-testing/tfjs-automl/img_classification/model.json';

const DAISY_URL =
    'https://storage.googleapis.com/tfjs-testing/tfjs-automl/img_classification/daisy.jpg';

describeWithFlags('nodejs+browser integration', {}, () => {
  let model: automl.ImageClassificationModel = null;

  beforeAll(async () => {
    model = await automl.loadImageClassification(MODEL_URL);
  });

  it('make prediction from a tensor', async () => {
    const img: Tensor3D = tf.zeros([100, 80, 3]);
    const predictions = await model.classify(img);
    expect(predictions[0].label).toBe('daisy');
    expect(predictions[1].label).toBe('dandelion');
    expect(predictions[2].label).toBe('roses');

    test_util.expectNumbersClose(predictions[0].prob, 0.5806022);
    test_util.expectNumbersClose(predictions[1].prob, 0.32249659);
    test_util.expectNumbersClose(predictions[2].prob, 0.0283515);
  });

  it('make prediction from a tensor without cropping', async () => {
    const img: Tensor3D = tf.zeros([100, 80, 3]);
    const predictions = await model.classify(img, {centerCrop: false});
    expect(predictions[0].label).toBe('daisy');
    expect(predictions[1].label).toBe('dandelion');
    expect(predictions[2].label).toBe('roses');

    test_util.expectNumbersClose(predictions[0].prob, 0.5806022);
    test_util.expectNumbersClose(predictions[1].prob, 0.32249659);
    test_util.expectNumbersClose(predictions[2].prob, 0.0283515);
  });

  it('no memory leak when making a prediction', async () => {
    const img: Tensor3D = tf.zeros([100, 80, 3]);
    const numTensorsBefore = tf.memory().numTensors;
    await model.classify(img);
    const numTensorsAfter = tf.memory().numTensors;
    expect(numTensorsAfter).toEqual(numTensorsBefore);
  });
});

describeWithFlags('browser integration', BROWSER_ENVS, () => {
  let model: automl.ImageClassificationModel = null;
  let daisyImg: HTMLImageElement;

  beforeAll(async () => {
    model = await automl.loadImageClassification(MODEL_URL);
    daisyImg = await fetchImage(DAISY_URL);
  });

  function assertTop3PredsForDaisy(
      predictions: ClassificationPrediction, centerCrop: boolean) {
    const probs = centerCrop ? [0.9310929, 0.0273733, 0.0130559] :
                               [0.8411523, 0.0729438, 0.03020708];
    expect(predictions[0].label).toBe('daisy');
    tf.test_util.expectNumbersClose(predictions[0].prob, probs[0]);

    expect(predictions[1].label).toBe('dandelion');
    tf.test_util.expectNumbersClose(predictions[1].prob, probs[1]);

    expect(predictions[2].label).toBe('roses');
    tf.test_util.expectNumbersClose(predictions[2].prob, probs[2]);
  }

  it('make prediction from an image element', async () => {
    const predictions = await model.classify(daisyImg);
    assertTop3PredsForDaisy(predictions, true /* centerCrop */);
  });

  it('make prediction from a canvas element', async () => {
    // Copy the pixels from the image to a canvas.
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = daisyImg.width;
    canvas.height = daisyImg.height;
    ctx.drawImage(daisyImg, 0, 0, daisyImg.width, daisyImg.height);

    const predictions = await model.classify(canvas);
    assertTop3PredsForDaisy(predictions, true /* centerCrop */);
  });

  it('make prediction without center cropping', async () => {
    const predictions = await model.classify(daisyImg, {centerCrop: false});
    assertTop3PredsForDaisy(predictions, false /* centerCrop */);
  });
});

async function fetchImage(url: string): Promise<HTMLImageElement> {
  const response = await fetch(url);
  const blob = await response.blob();
  const img = new Image();
  const blobUrl = URL.createObjectURL(blob);
  return new Promise((resolve, reject) => {
    img.onload = () => {
      URL.revokeObjectURL(blobUrl);
      resolve(img);
    };
    img.onerror = (evt /* Arg is an event, not error. Can't rethrow it */) => {
      reject(new Error('Failed to load blob as image.'));
    };
    img.src = blobUrl;
  });
}
