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
    'https://storage.googleapis.com/tfjs-testing/tfjs-automl/object_detection/model.json';

const SALAD_IMAGE =
    'https://storage.googleapis.com/tfjs-testing/tfjs-automl/object_detection/test_image.jpg';

describeWithFlags('object detection', {}, () => {
  let model: automl.ObjectDetectionModel = null;
  const originalTimeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
  jasmine.DEFAULT_TIMEOUT_INTERVAL = 40000;

  beforeAll(async () => {
    model = await automl.loadObjectDetection(MODEL_URL);
  });

  afterAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = originalTimeout;
  });

  it('prediction from a zero tensor', async () => {
    const img: tf.Tensor3D = tf.zeros([100, 80, 3]);
    const predictions = await model.detect(img);
    expect(predictions.length).toBe(0);
  });

  it('prediction from a zero tensor with score threshold', async () => {
    const img: tf.Tensor3D = tf.zeros([100, 80, 3]);
    const options = {score: 0.11};
    const predictions = await model.detect(img, options);
    expect(predictions.length).toBe(1);
    const {box: {left, top, width, height}, label, score} = predictions[0];

    expect(label).toBe('Salad');
    tf.test_util.expectNumbersClose(score, 0.116391);
    tf.test_util.expectNumbersClose(left, 26.6741156);
    tf.test_util.expectNumbersClose(top, 94.003981);
    tf.test_util.expectNumbersClose(width, 22.6277661);
    tf.test_util.expectNumbersClose(height, 6.30063414);
  });

  it('prediction with iou threshold', async () => {
    const img: tf.Tensor3D = tf.zeros([100, 80, 3]);
    const options: automl.ObjectDetectionOptions = {score: 0.10};
    options.iou = 0.9;
    let predictions = await model.detect(img, options);
    expect(predictions.length).toBe(9);

    options.iou = 0.1;
    predictions = await model.detect(img, options);
    expect(predictions.length).toBe(1);
  });

  it('no memory leak when making a prediction', async () => {
    const img: tf.Tensor3D = tf.zeros([100, 80, 3]);
    const numTensorsBefore = tf.memory().numTensors;
    await model.detect(img);
    const numTensorsAfter = tf.memory().numTensors;
    expect(numTensorsAfter).toEqual(numTensorsBefore);
  });

  it('has access to dictionary', () => {
    expect(model.dictionary).toEqual([
      'background', 'Tomato', 'Seafood', 'Salad', 'Baked Goods', 'Cheese'
    ]);
  });

  it('can access the underlying graph model', () => {
    expect(model.graphModel instanceof GraphModel).toBe(true);
  });
});

describeWithFlags('object detection browser', BROWSER_ENVS, () => {
  let model: automl.ObjectDetectionModel = null;
  let saladImg: HTMLImageElement;

  beforeAll(async () => {
    model = await automl.loadObjectDetection(MODEL_URL);
    saladImg = await fetchImage(SALAD_IMAGE);
  });

  function assertTop3PredsForSalad(predictions: automl.PredictedObject[]) {
    expect(predictions.length).toBe(3);
    const [top1, top2, top3] = predictions;
    expect(top1.label).toBe('Tomato');
    tf.test_util.expectNumbersClose(top1.score, 0.97170084);

    expect(top2.label).toBe('Tomato');
    tf.test_util.expectNumbersClose(top2.score, 0.93456619);

    expect(top3.label).toBe('Salad');
    tf.test_util.expectNumbersClose(top3.score, 0.9074271);
  }

  it('make prediction from an image element', async () => {
    const predictions = await model.detect(saladImg);
    assertTop3PredsForSalad(predictions);
  });

  it('make prediction from a canvas element', async () => {
    // Copy the pixels from the image to a canvas.
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = saladImg.width;
    canvas.height = saladImg.height;
    ctx.drawImage(saladImg, 0, 0, saladImg.width, saladImg.height);

    const predictions = await model.detect(canvas);
    assertTop3PredsForSalad(predictions);
  });

  it('make prediction from image data', async () => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = saladImg.width;
    canvas.height = saladImg.height;
    ctx.drawImage(saladImg, 0, 0, saladImg.width, saladImg.height);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    const predictions = await model.detect(imageData);
    assertTop3PredsForSalad(predictions);
  });
});
