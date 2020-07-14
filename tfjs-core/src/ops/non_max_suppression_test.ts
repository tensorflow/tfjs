/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
import * as tf from '../index';
import {ALL_ENVS, describeWithFlags} from '../jasmine_util';
import {expectArraysEqual} from '../test_util';

describeWithFlags('nonMaxSuppression', ALL_ENVS, () => {
  it('select from three clusters', async () => {
    const boxes = tf.tensor2d(
        [
          0, 0,  1, 1,  0, 0.1,  1, 1.1,  0, -0.1, 1, 0.9,
          0, 10, 1, 11, 0, 10.1, 1, 11.1, 0, 100,  1, 101
        ],
        [6, 4]);
    const scores = tf.tensor1d([0.9, 0.75, 0.6, 0.95, 0.5, 0.3]);
    const maxOutputSize = 3;
    const iouThreshold = 0.5;
    const scoreThreshold = 0;

    const before = tf.memory().numTensors;
    const indices = tf.image.nonMaxSuppression(
        boxes, scores, maxOutputSize, iouThreshold, scoreThreshold);
    const after = tf.memory().numTensors;

    expect(indices.shape).toEqual([3]);
    expectArraysEqual(await indices.data(), [3, 0, 5]);
    expect(after).toEqual(before + 1);
  });

  it('select from three clusters flipped coordinates', async () => {
    const boxes = tf.tensor2d(
        [
          1, 1,  0, 0,  0, 0.1,  1, 1.1,  0, .9,  1, -0.1,
          0, 10, 1, 11, 1, 10.1, 0, 11.1, 1, 101, 0, 100
        ],
        [6, 4]);
    const scores = tf.tensor1d([0.9, 0.75, 0.6, 0.95, 0.5, 0.3]);
    const maxOutputSize = 3;
    const iouThreshold = 0.5;
    const scoreThreshold = 0;
    const indices = tf.image.nonMaxSuppression(
        boxes, scores, maxOutputSize, iouThreshold, scoreThreshold);

    expect(indices.shape).toEqual([3]);
    expectArraysEqual(await indices.data(), [3, 0, 5]);
  });

  it('select at most two boxes from three clusters', async () => {
    const boxes = tf.tensor2d(
        [
          0, 0,  1, 1,  0, 0.1,  1, 1.1,  0, -0.1, 1, 0.9,
          0, 10, 1, 11, 0, 10.1, 1, 11.1, 0, 100,  1, 101
        ],
        [6, 4]);
    const scores = tf.tensor1d([0.9, 0.75, 0.6, 0.95, 0.5, 0.3]);
    const maxOutputSize = 2;
    const iouThreshold = 0.5;
    const scoreThreshold = 0;
    const indices = tf.image.nonMaxSuppression(
        boxes, scores, maxOutputSize, iouThreshold, scoreThreshold);

    expect(indices.shape).toEqual([2]);
    expectArraysEqual(await indices.data(), [3, 0]);
  });

  it('select at most thirty boxes from three clusters', async () => {
    const boxes = tf.tensor2d(
        [
          0, 0,  1, 1,  0, 0.1,  1, 1.1,  0, -0.1, 1, 0.9,
          0, 10, 1, 11, 0, 10.1, 1, 11.1, 0, 100,  1, 101
        ],
        [6, 4]);
    const scores = tf.tensor1d([0.9, 0.75, 0.6, 0.95, 0.5, 0.3]);
    const maxOutputSize = 30;
    const iouThreshold = 0.5;
    const scoreThreshold = 0;
    const indices = tf.image.nonMaxSuppression(
        boxes, scores, maxOutputSize, iouThreshold, scoreThreshold);

    expect(indices.shape).toEqual([3]);
    expectArraysEqual(await indices.data(), [3, 0, 5]);
  });

  it('select single box', async () => {
    const boxes = tf.tensor2d([0, 0, 1, 1], [1, 4]);
    const scores = tf.tensor1d([0.9]);
    const maxOutputSize = 3;
    const iouThreshold = 0.5;
    const scoreThreshold = 0;
    const indices = tf.image.nonMaxSuppression(
        boxes, scores, maxOutputSize, iouThreshold, scoreThreshold);

    expect(indices.shape).toEqual([1]);
    expectArraysEqual(await indices.data(), [0]);
  });

  it('select from ten identical boxes', async () => {
    const numBoxes = 10;
    const corners = new Array(numBoxes)
                        .fill(0)
                        .map(_ => [0, 0, 1, 1])
                        .reduce((arr, curr) => arr.concat(curr));
    const boxes = tf.tensor2d(corners, [numBoxes, 4]);
    const scores = tf.tensor1d(Array(numBoxes).fill(0.9));
    const maxOutputSize = 3;
    const iouThreshold = 0.5;
    const scoreThreshold = 0;
    const indices = tf.image.nonMaxSuppression(
        boxes, scores, maxOutputSize, iouThreshold, scoreThreshold);

    expect(indices.shape).toEqual([1]);
    expectArraysEqual(await indices.data(), [0]);
  });

  it('inconsistent box and score shapes', () => {
    const boxes = tf.tensor2d(
        [
          0, 0,  1, 1,  0, 0.1,  1, 1.1,  0, -0.1, 1, 0.9,
          0, 10, 1, 11, 0, 10.1, 1, 11.1, 0, 100,  1, 101
        ],
        [6, 4]);
    const scores = tf.tensor1d([0.9, 0.75, 0.6, 0.95, 0.5]);
    const maxOutputSize = 30;
    const iouThreshold = 0.5;
    const scoreThreshold = 0;
    expect(
        () => tf.image.nonMaxSuppression(
            boxes, scores, maxOutputSize, iouThreshold, scoreThreshold))
        .toThrowError(/scores has incompatible shape with boxes/);
  });

  it('invalid iou threshold', () => {
    const boxes = tf.tensor2d([0, 0, 1, 1], [1, 4]);
    const scores = tf.tensor1d([0.9]);
    const maxOutputSize = 3;
    const iouThreshold = 1.2;
    const scoreThreshold = 0;
    expect(
        () => tf.image.nonMaxSuppression(
            boxes, scores, maxOutputSize, iouThreshold, scoreThreshold))
        .toThrowError(/iouThreshold must be in \[0, 1\]/);
  });

  it('empty input', async () => {
    const boxes = tf.tensor2d([], [0, 4]);
    const scores = tf.tensor1d([]);
    const maxOutputSize = 3;
    const iouThreshold = 0.5;
    const scoreThreshold = 0;
    const indices = tf.image.nonMaxSuppression(
        boxes, scores, maxOutputSize, iouThreshold, scoreThreshold);

    expect(indices.shape).toEqual([0]);
    expectArraysEqual(await indices.data(), []);
  });

  it('accepts a tensor-like object', async () => {
    const boxes = [[0, 0, 1, 1], [0, 1, 1, 2]];
    const scores = [1, 2];
    const indices = tf.image.nonMaxSuppression(boxes, scores, 10);
    expect(indices.shape).toEqual([2]);
    expect(indices.dtype).toEqual('int32');
    expectArraysEqual(await indices.data(), [1, 0]);
  });

  it('works when inputs are not explicitly initialized on the CPU',
     async () => {
       // This test ensures that asynchronous backends work with NMS, which
       // requires inputs to reside on the CPU.
       const boxes = tf.tensor2d(
           [
             0, 0,  1, 1,  0, 0.1,  1, 1.1,  0, -0.1, 1, 0.9,
             0, 10, 1, 11, 0, 10.1, 1, 11.1, 0, 100,  1, 101
           ],
           [6, 4]);
       const a = tf.tensor1d([0, 1, -2, -4, 4, -4]);
       const b = tf.tensor1d([0.15, 0.2, 0.25, 0.5, 0.7, 1.2]);
       const scores = a.div(b);
       const maxOutputSize = 2;
       const iouThreshold = 0.5;
       const scoreThreshold = 0;
       await scores.data();
       const indices = tf.image.nonMaxSuppression(
           boxes, scores as tf.Tensor1D, maxOutputSize, iouThreshold,
           scoreThreshold);

       expect(indices.shape).toEqual([2]);
       expectArraysEqual(await indices.data(), [4, 1]);
     });
});
