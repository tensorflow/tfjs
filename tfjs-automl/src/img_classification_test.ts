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

import {GraphModel} from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {BROWSER_ENVS, describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

import * as automl from './index';
import {fetchImage} from './test_util';

const MODEL_URL =
    'https://storage.googleapis.com/tfjs-testing/tfjs-automl/img_classification/model.json';

const DAISY_URL =
    'https://storage.googleapis.com/tfjs-testing/tfjs-automl/img_classification/daisy.jpg';

describeWithFlags('image classification', {}, () => {
  let model: automl.ImageClassificationModel = null;

  beforeAll(async () => {
    model = await automl.loadImageClassification(MODEL_URL);
  });

  it('make prediction from a tensor', async () => {
    const img: tf.Tensor3D = tf.zeros([100, 80, 3]);
    const predictions = await model.classify(img);
    expect(predictions[0].label).toBe('daisy');
    expect(predictions[1].label).toBe('dandelion');
    expect(predictions[2].label).toBe('roses');

    tf.test_util.expectNumbersClose(predictions[0].prob, 0.5806022);
    tf.test_util.expectNumbersClose(predictions[1].prob, 0.32249659);
    tf.test_util.expectNumbersClose(predictions[2].prob, 0.0283515);
  });

  it('make prediction from a tensor without cropping', async () => {
    const img: tf.Tensor3D = tf.zeros([100, 80, 3]);
    const predictions = await model.classify(img, {centerCrop: false});
    expect(predictions[0].label).toBe('daisy');
    expect(predictions[1].label).toBe('dandelion');
    expect(predictions[2].label).toBe('roses');

    tf.test_util.expectNumbersClose(predictions[0].prob, 0.5806022);
    tf.test_util.expectNumbersClose(predictions[1].prob, 0.32249659);
    tf.test_util.expectNumbersClose(predictions[2].prob, 0.0283515);
  });

  it('no memory leak when making a prediction', async () => {
    const img: tf.Tensor3D = tf.zeros([100, 80, 3]);
    const numTensorsBefore = tf.memory().numTensors;
    await model.classify(img);
    const numTensorsAfter = tf.memory().numTensors;
    expect(numTensorsAfter).toEqual(numTensorsBefore);
  });

  it('has access to dictionary', () => {
    expect(model.dictionary).toEqual([
      'daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'
    ]);
  });

  it('can access the underlying graph model', () => {
    expect(model.graphModel instanceof GraphModel).toBe(true);
    expect(model.graphModel.inputNodes).toEqual(['image']);
    expect(model.graphModel.outputNodes).toEqual(['scores']);
    const img: tf.Tensor = tf.zeros([1, 224, 224, 3]);
    const scores = model.graphModel.predict(img) as tf.Tensor;
    expect(scores.shape).toEqual([1, 5]);
  });
});

describeWithFlags('image classification browser', BROWSER_ENVS, () => {
  let model: automl.ImageClassificationModel = null;
  let daisyImg: HTMLImageElement;

  beforeAll(async () => {
    model = await automl.loadImageClassification(MODEL_URL);
    daisyImg = await fetchImage(DAISY_URL);
  });

  function assertTop3PredsForDaisy(
      predictions: automl.ImagePrediction[], centerCrop: boolean) {
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
