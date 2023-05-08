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
import * as tf from '../../index';
import {ALL_ENVS, describeWithFlags} from '../../jasmine_util';
import {expectArraysClose} from '../../test_util';

describeWithFlags('cropAndResize', ALL_ENVS, () => {
  it('1x1-bilinear', async () => {
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([0, 0, 1, 1], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');

    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [1, 1], 'bilinear', 0);

    expect(output.shape).toEqual([1, 1, 1, 1]);
    expect(output.dtype).toBe('float32');
    expectArraysClose(await output.data(), [2.5]);
  });

  it('5x5-bilinear, no change in shape', async () => {
    const image: tf.Tensor4D = tf.ones([1, 5, 5, 3]);
    const boxes: tf.Tensor2D = tf.tensor2d([0, 0, 1, 1], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');

    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [5, 5], 'bilinear', 0);

    expect(output.shape).toEqual([1, 5, 5, 3]);
    expect(output.dtype).toBe('float32');
    expectArraysClose(await output.data(), await image.data());
  });

  it('5x5-bilinear, no arguments passed in for method or extrapolation',
     async () => {
       const image: tf.Tensor4D = tf.ones([1, 5, 5, 3]);
       const boxes: tf.Tensor2D = tf.tensor2d([0, 0, 1, 1], [1, 4]);
       const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');

       const output = tf.image.cropAndResize(image, boxes, boxInd, [5, 5]);

       expect(output.shape).toEqual([1, 5, 5, 3]);
       expect(output.dtype).toBe('float32');
       expectArraysClose(await output.data(), await image.data());
     });

  it('5x5-bilinear, just a crop, no resize', async () => {
    const image: tf.Tensor4D = tf.ones([1, 6, 6, 3]);
    const boxes: tf.Tensor2D = tf.tensor2d([0.5, 0.5, 1, 1], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');

    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [3, 3], 'bilinear', 0);

    expect(output.shape).toEqual([1, 3, 3, 3]);
    expect(output.dtype).toBe('float32');
    expectArraysClose(await output.data(), await tf.ones([1, 3, 3, 3]).data());
  });

  it('1x1-nearest', async () => {
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([0, 0, 1, 1], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');

    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [1, 1], 'nearest', 0);

    expect(output.shape).toEqual([1, 1, 1, 1]);
    expect(output.dtype).toBe('float32');
    expectArraysClose(await output.data(), [4.0]);
  });
  it('1x1Flipped-bilinear', async () => {
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([1, 1, 0, 0], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');

    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [1, 1], 'bilinear', 0);

    expect(output.shape).toEqual([1, 1, 1, 1]);
    expect(output.dtype).toBe('float32');
    expectArraysClose(await output.data(), [2.5]);
  });
  it('1x1Flipped-nearest', async () => {
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([1, 1, 0, 0], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');

    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [1, 1], 'nearest', 0);

    expect(output.shape).toEqual([1, 1, 1, 1]);
    expect(output.dtype).toBe('float32');
    expectArraysClose(await output.data(), [4.0]);
  });
  it('3x3-bilinear', async () => {
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([0, 0, 1, 1], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');

    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [3, 3], 'bilinear', 0);

    expect(output.shape).toEqual([1, 3, 3, 1]);
    expect(output.dtype).toBe('float32');
    expectArraysClose(await output.data(), [1, 1.5, 2, 2, 2.5, 3, 3, 3.5, 4]);
  });
  it('3x3-nearest', async () => {
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([0, 0, 1, 1], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');

    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [3, 3], 'nearest', 0);

    expect(output.shape).toEqual([1, 3, 3, 1]);
    expect(output.dtype).toBe('float32');
    expectArraysClose(await output.data(), [1, 2, 2, 3, 4, 4, 3, 4, 4]);
  });
  it('3x3Flipped-bilinear', async () => {
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([1, 1, 0, 0], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');

    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [3, 3], 'bilinear', 0);

    expect(output.shape).toEqual([1, 3, 3, 1]);
    expect(output.dtype).toBe('float32');
    expectArraysClose(await output.data(), [4, 3.5, 3, 3, 2.5, 2, 2, 1.5, 1]);
  });
  it('3x3Flipped-nearest', async () => {
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([1, 1, 0, 0], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');

    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [3, 3], 'nearest', 0);

    expect(output.shape).toEqual([1, 3, 3, 1]);
    expect(output.dtype).toBe('float32');
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
    expect(output.dtype).toBe('float32');
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
    expect(output.dtype).toBe('float32');
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
    expect(output.dtype).toBe('float32');
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
    expect(output.dtype).toBe('float32');
    expectArraysClose(await output.data(), [9, 7, 3, 1, 5, 4, 2, 1]);
  });
  it('3x3-BoxisRectangular', async () => {
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([0, 0, 1, 1.5], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');

    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [3, 3], 'bilinear', 0);

    expect(output.shape).toEqual([1, 3, 3, 1]);
    expect(output.dtype).toBe('float32');
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
    expect(output.dtype).toBe('float32');
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
    expect(output.dtype).toBe('float32');
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
    expect(output.dtype).toBe('float32');
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
    expect(output.dtype).toBe('float32');
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
    expect(output.dtype).toBe('float32');
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
    expect(output.dtype).toBe('float32');
    expectArraysClose(
        await output.data(),
        [1, 2, 0, 3, 4, 0, 3, 4, 0, 5, 6, 6, 7, 8, 8, 0, 0, 0]);
  });
  it('int32 image returns float output', async () => {
    const image: tf.Tensor4D =
        tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 1], 'int32');
    const boxes: tf.Tensor2D =
        tf.tensor2d([0, 0, 1, 1.5, 0, 0, 1.5, 1], [2, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0, 1], 'int32');

    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [3, 3], 'bilinear', 0);

    expect(output.shape).toEqual([2, 3, 3, 1]);
    expect(output.dtype).toBe('float32');
    expectArraysClose(
        await output.data(),
        [1, 1.75, 0, 2, 2.75, 0, 3, 3.75, 0, 5, 5.5, 6, 6.5, 7, 7.5, 0, 0, 0]);
  });
});
