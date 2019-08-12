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
import * as tf from '../index';
import {ALL_ENVS, describeWithFlags} from '../jasmine_util';
import {expectArraysClose, expectArraysEqual} from '../test_util';

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
    const indices = tf.image.nonMaxSuppression(
        boxes, scores, maxOutputSize, iouThreshold, scoreThreshold);

    expect(indices.shape).toEqual([3]);
    expectArraysEqual(await indices.data(), [3, 0, 5]);
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
});

describeWithFlags('nonMaxSuppressionAsync', ALL_ENVS, () => {
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
    const indices = await tf.image.nonMaxSuppressionAsync(
        boxes, scores, maxOutputSize, iouThreshold, scoreThreshold);

    expect(indices.shape).toEqual([3]);
    expectArraysEqual(await indices.data(), [3, 0, 5]);
  });

  it('accepts a tensor-like object', async () => {
    const boxes = [[0, 0, 1, 1], [0, 1, 1, 2]];
    const scores = [1, 2];
    const indices = await tf.image.nonMaxSuppressionAsync(boxes, scores, 10);
    expect(indices.shape).toEqual([2]);
    expect(indices.dtype).toEqual('int32');
    expectArraysEqual(await indices.data(), [1, 0]);
  });
});

describeWithFlags('cropAndResize', ALL_ENVS, () => {
  it('1x1-bilinear', async () => {
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([0, 0, 1, 1], [1, 4]);

    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [1, 1], 'bilinear', 0);
    expect(output.shape).toEqual([1, 1, 1, 1]);
    expectArraysClose(await output.data(), [2.5]);
  });
  it('1x1-nearest', async () => {
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([0, 0, 1, 1], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [1, 1], 'nearest', 0);
    expect(output.shape).toEqual([1, 1, 1, 1]);
    expectArraysClose(await output.data(), [4.0]);
  });
  it('1x1Flipped-bilinear', async () => {
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([1, 1, 0, 0], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [1, 1], 'bilinear', 0);
    expect(output.shape).toEqual([1, 1, 1, 1]);
    expectArraysClose(await output.data(), [2.5]);
  });
  it('1x1Flipped-nearest', async () => {
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([1, 1, 0, 0], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [1, 1], 'nearest', 0);
    expect(output.shape).toEqual([1, 1, 1, 1]);
    expectArraysClose(await output.data(), [4.0]);
  });
  it('3x3-bilinear', async () => {
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([0, 0, 1, 1], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [3, 3], 'bilinear', 0);
    expect(output.shape).toEqual([1, 3, 3, 1]);
    expectArraysClose(await output.data(), [1, 1.5, 2, 2, 2.5, 3, 3, 3.5, 4]);
  });
  it('3x3-nearest', async () => {
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([0, 0, 1, 1], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [3, 3], 'nearest', 0);
    expect(output.shape).toEqual([1, 3, 3, 1]);
    expectArraysClose(await output.data(), [1, 2, 2, 3, 4, 4, 3, 4, 4]);
  });
  it('3x3Flipped-bilinear', async () => {
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([1, 1, 0, 0], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [3, 3], 'bilinear', 0);
    expect(output.shape).toEqual([1, 3, 3, 1]);
    expectArraysClose(await output.data(), [4, 3.5, 3, 3, 2.5, 2, 2, 1.5, 1]);
  });
  it('3x3Flipped-nearest', async () => {
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([1, 1, 0, 0], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [3, 3], 'nearest', 0);
    expect(output.shape).toEqual([1, 3, 3, 1]);
    expectArraysClose(await output.data(), [4, 4, 3, 4, 4, 3, 2, 2, 1]);
  });
  it('3x3to2x2-bilinear', async () => {
    const image: tf.Tensor4D =
        tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3, 1]);
    const boxes: tf.Tensor2D =
        tf.tensor2d([0, 0, 1, 1, 0, 0, 0.5, 0.5], [2, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0, 0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [2, 2], 'bilinear', 0);
    expect(output.shape).toEqual([2, 2, 2, 1]);
    expectArraysClose(await output.data(), [1, 3, 7, 9, 1, 2, 4, 5]);
  });
  it('3x3to2x2-nearest', async () => {
    const image: tf.Tensor4D =
        tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3, 1]);
    const boxes: tf.Tensor2D =
        tf.tensor2d([0, 0, 1, 1, 0, 0, 0.5, 0.5], [2, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0, 0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [2, 2], 'nearest', 0);
    expect(output.shape).toEqual([2, 2, 2, 1]);
    expectArraysClose(await output.data(), [1, 3, 7, 9, 1, 2, 4, 5]);
  });
  it('3x3to2x2Flipped-bilinear', async () => {
    const image: tf.Tensor4D =
        tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3, 1]);
    const boxes: tf.Tensor2D =
        tf.tensor2d([1, 1, 0, 0, 0.5, 0.5, 0, 0], [2, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0, 0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [2, 2], 'bilinear', 0);
    expect(output.shape).toEqual([2, 2, 2, 1]);
    expectArraysClose(await output.data(), [9, 7, 3, 1, 5, 4, 2, 1]);
  });
  it('3x3to2x2Flipped-nearest', async () => {
    const image: tf.Tensor4D =
        tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3, 1]);
    const boxes: tf.Tensor2D =
        tf.tensor2d([1, 1, 0, 0, 0.5, 0.5, 0, 0], [2, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0, 0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [2, 2], 'nearest', 0);
    expect(output.shape).toEqual([2, 2, 2, 1]);
    expectArraysClose(await output.data(), [9, 7, 3, 1, 5, 4, 2, 1]);
  });
  it('3x3-BoxisRectangular', async () => {
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([0, 0, 1, 1.5], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [3, 3], 'bilinear', 0);
    expect(output.shape).toEqual([1, 3, 3, 1]);
    expectArraysClose(
        await output.data(), [1, 1.75, 0, 2, 2.75, 0, 3, 3.75, 0]);
  });
  it('3x3-BoxisRectangular-nearest', async () => {
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([0, 0, 1, 1.5], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [3, 3], 'nearest', 0);
    expect(output.shape).toEqual([1, 3, 3, 1]);
    expectArraysClose(await output.data(), [1, 2, 0, 3, 4, 0, 3, 4, 0]);
  });
  it('2x2to3x3-Extrapolated', async () => {
    const val = -1;
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([-1, -1, 1, 1], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [3, 3], 'bilinear', val);
    expect(output.shape).toEqual([1, 3, 3, 1]);
    expectArraysClose(
        await output.data(), [val, val, val, val, 1, 2, val, 3, 4]);
  });
  it('2x2to3x3-Extrapolated-Float', async () => {
    const val = -1.5;
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([-1, -1, 1, 1], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [3, 3], 'bilinear', val);
    expect(output.shape).toEqual([1, 3, 3, 1]);
    expectArraysClose(
        await output.data(), [val, val, val, val, 1, 2, val, 3, 4]);
  });
  it('2x2to3x3-NoCrop', async () => {
    const val = -1.0;
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([], [0, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [3, 3], 'bilinear', val);
    expect(output.shape).toEqual([0, 3, 3, 1]);
    expectArraysClose(await output.data(), []);
  });
  it('MultipleBoxes-DifferentBoxes', async () => {
    const image: tf.Tensor4D =
        tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 1]);
    const boxes: tf.Tensor2D =
        tf.tensor2d([0, 0, 1, 1.5, 0, 0, 1.5, 1], [2, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0, 1], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [3, 3], 'bilinear', 0);
    expect(output.shape).toEqual([2, 3, 3, 1]);
    expectArraysClose(
        await output.data(),
        [1, 1.75, 0, 2, 2.75, 0, 3, 3.75, 0, 5, 5.5, 6, 6.5, 7, 7.5, 0, 0, 0]);
  });
  it('MultipleBoxes-DifferentBoxes-Nearest', async () => {
    const image: tf.Tensor4D =
        tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([0, 0, 1, 1.5, 0, 0, 2, 1], [2, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0, 1], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [3, 3], 'nearest', 0);
    expect(output.shape).toEqual([2, 3, 3, 1]);
    expectArraysClose(
        await output.data(),
        [1, 2, 0, 3, 4, 0, 3, 4, 0, 5, 6, 6, 7, 8, 8, 0, 0, 0]);
  });
});
