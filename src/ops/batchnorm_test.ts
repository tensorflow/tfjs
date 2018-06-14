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

import * as tf from '../index';
import {describeWithFlags} from '../jasmine_util';
// tslint:disable-next-line:max-line-length
import {ALL_ENVS, expectArraysClose} from '../test_util';

describeWithFlags('batchNormalization4D', ALL_ENVS, () => {
  it('simple batchnorm4D, no offset or scale, 2x1x1x2', () => {
    const x = tf.tensor4d([2, 4, 9, 23], [2, 1, 1, 2]);
    const mean = tf.tensor1d([1, 2]);
    const variance = tf.tensor1d([2, 3]);
    const varianceEpsilon = .001;

    const result = tf.batchNormalization4d(
        x, mean, variance, varianceEpsilon, undefined, undefined);

    expectArraysClose(result, [
      (x.get(0, 0, 0, 0) - mean.get(0)) * 1 /
          Math.sqrt(variance.get(0) + varianceEpsilon),
      (x.get(0, 0, 0, 1) - mean.get(1)) * 1 /
          Math.sqrt(variance.get(1) + varianceEpsilon),
      (x.get(1, 0, 0, 0) - mean.get(0)) * 1 /
          Math.sqrt(variance.get(0) + varianceEpsilon),
      (x.get(1, 0, 0, 1) - mean.get(1)) * 1 /
          Math.sqrt(variance.get(1) + varianceEpsilon)
    ]);
  });

  it('simple batchnorm4D, no offset, 2x1x1x2', () => {
    const x = tf.tensor4d([2, 4, 9, 23], [2, 1, 1, 2]);
    const mean = tf.tensor1d([1, 2]);
    const variance = tf.tensor1d([2, 3]);
    const scale = tf.tensor1d([4, 5]);
    const varianceEpsilon = .001;

    const result = tf.batchNormalization4d(
        x, mean, variance, varianceEpsilon, scale, undefined);

    expectArraysClose(result, [
      (x.get(0, 0, 0, 0) - mean.get(0)) * scale.get(0) /
          Math.sqrt(variance.get(0) + varianceEpsilon),
      (x.get(0, 0, 0, 1) - mean.get(1)) * scale.get(1) /
          Math.sqrt(variance.get(1) + varianceEpsilon),
      (x.get(1, 0, 0, 0) - mean.get(0)) * scale.get(0) /
          Math.sqrt(variance.get(0) + varianceEpsilon),
      (x.get(1, 0, 0, 1) - mean.get(1)) * scale.get(1) /
          Math.sqrt(variance.get(1) + varianceEpsilon)
    ]);
  });

  it('simple batchnorm4D, no scale, 2x1x1x2', () => {
    const x = tf.tensor4d([2, 4, 9, 23], [2, 1, 1, 2]);
    const mean = tf.tensor1d([1, 2]);
    const variance = tf.tensor1d([2, 3]);
    const offset = tf.tensor1d([4, 5]);

    const varianceEpsilon = .001;

    const result = tf.batchNormalization4d(
        x, mean, variance, varianceEpsilon, undefined, offset);

    expectArraysClose(result, [
      offset.get(0) +
          (x.get(0, 0, 0, 0) - mean.get(0)) * 1 /
              Math.sqrt(variance.get(0) + varianceEpsilon),
      offset.get(1) +
          (x.get(0, 0, 0, 1) - mean.get(1)) * 1 /
              Math.sqrt(variance.get(1) + varianceEpsilon),
      offset.get(0) +
          (x.get(1, 0, 0, 0) - mean.get(0)) * 1 /
              Math.sqrt(variance.get(0) + varianceEpsilon),
      offset.get(1) +
          (x.get(1, 0, 0, 1) - mean.get(1)) * 1 /
              Math.sqrt(variance.get(1) + varianceEpsilon)
    ]);
  });

  it('simple batchnorm4D, 2x1x1x2', () => {
    const x = tf.tensor4d([2, 4, 9, 23], [2, 1, 1, 2]);
    const mean = tf.tensor1d([1, 2]);
    const variance = tf.tensor1d([2, 3]);
    const offset = tf.tensor1d([3, 4]);
    const scale = tf.tensor1d([4, 5]);

    const varianceEpsilon = .001;

    const result = tf.batchNormalization4d(
        x, mean, variance, varianceEpsilon, scale, offset);

    expectArraysClose(result, [
      offset.get(0) +
          (x.get(0, 0, 0, 0) - mean.get(0)) * scale.get(0) /
              Math.sqrt(variance.get(0) + varianceEpsilon),
      offset.get(1) +
          (x.get(0, 0, 0, 1) - mean.get(1)) * scale.get(1) /
              Math.sqrt(variance.get(1) + varianceEpsilon),
      offset.get(0) +
          (x.get(1, 0, 0, 0) - mean.get(0)) * scale.get(0) /
              Math.sqrt(variance.get(0) + varianceEpsilon),
      offset.get(1) +
          (x.get(1, 0, 0, 1) - mean.get(1)) * scale.get(1) /
              Math.sqrt(variance.get(1) + varianceEpsilon)
    ]);
  });

  it('simple batchnorm4D gradients, 2x1x1x2', () => {
    const x = tf.tensor4d([2, 4, 9, 23], [2, 1, 1, 2]);
    const mean = tf.tensor1d([1, 2]);
    const variance = tf.tensor1d([2, 3]);
    const offset = tf.tensor1d([3, 4]);
    const scale = tf.tensor1d([2, 5]);

    const varianceEpsilon = .001;

    const dy = tf.tensor4d([-1, -1, -1, -1], [2, 1, 1, 2]);
    const gradX = tf.grad(
        (x: tf.Tensor4D) => tf.batchNormalization4d(
            x, mean, variance, varianceEpsilon, scale, offset))(x, dy);
    expectArraysClose(
        gradX, tf.tensor4d([-1.414, -2.887, -1.414, -2.887], [2, 1, 1, 2]));
    const gradMean = tf.grad(
        (mean: tf.Tensor1D) => tf.batchNormalization4d(
            x, mean, variance, varianceEpsilon, scale, offset))(mean, dy);
    expectArraysClose(gradMean, tf.tensor1d([2.828, 5.773]));
    const gradVariance = tf.grad(
        (variance: tf.Tensor1D) => tf.batchNormalization4d(
            x, mean, variance, varianceEpsilon, scale, offset))(variance, dy);
    expectArraysClose(gradVariance, tf.tensor1d([3.180, 11.060]));
    const gradOffset = tf.grad(
        (offset: tf.Tensor1D) => tf.batchNormalization4d(
            x, mean, variance, varianceEpsilon, scale, offset))(offset, dy);
    expectArraysClose(gradOffset, dy.sum([0, 1, 2]));
    const gradScale = tf.grad(
        (scale: tf.Tensor1D) => tf.batchNormalization4d(
            x, mean, variance, varianceEpsilon, scale, offset))(scale, dy);
    expectArraysClose(gradScale, tf.tensor1d([-6.362, -13.277]));
  });

  it('batchnorm4D gradients, same shapes in x, mean and variance', () => {
    const x = tf.tensor4d([10, 20, 30, 40], [2, 1, 1, 2]);
    const mean = tf.tensor4d([0, 5, 10, 15], [2, 1, 1, 2]);
    const variance = tf.tensor4d([2, 4, 6, 8], [2, 1, 1, 2]);
    const scale = tf.tensor4d([2, 5, 2, 5], [2, 1, 1, 2]);
    const offset = tf.tensor4d([0, 0, 0, 0], [2, 1, 1, 2]);

    const varianceEpsilon = .001;

    const dy = tf.tensor4d([-1, -1, -1, -1], [2, 1, 1, 2]);
    const gradX = tf.grad(
        (x: tf.Tensor4D) => tf.batchNormalization4d(
            x, mean, variance, varianceEpsilon, scale, offset))(x, dy);
    expectArraysClose(
        gradX, tf.tensor4d([-1.414, -2.500, -0.816, -1.768], [2, 1, 1, 2]));
    const gradMean = tf.grad(
        (mean: tf.Tensor4D) => tf.batchNormalization4d(
            x, mean, variance, varianceEpsilon, scale, offset))(mean, dy);
    expectArraysClose(
        gradMean, tf.tensor4d([1.414, 2.500, 0.816, 1.768], [2, 1, 1, 2]));
    const gradVariance = tf.grad(
        (variance: tf.Tensor4D) => tf.batchNormalization4d(
            x, mean, variance, varianceEpsilon, scale, offset))(variance, dy);
    expectArraysClose(
        gradVariance, tf.tensor4d([3.533, 4.686, 1.360, 2.762], [2, 1, 1, 2]));
    const gradOffset = tf.grad(
        (offset: tf.Tensor4D) => tf.batchNormalization4d(
            x, mean, variance, varianceEpsilon, scale, offset))(offset, dy);
    expectArraysClose(gradOffset, dy);
    const gradScale = tf.grad(
        (scale: tf.Tensor4D) => tf.batchNormalization4d(
            x, mean, variance, varianceEpsilon, scale, offset))(scale, dy);
    expectArraysClose(
        gradScale, tf.tensor4d([-7.069, -7.499, -8.164, -8.838], [2, 1, 1, 2]));
  });
});

describeWithFlags('batchNormalization3D', ALL_ENVS, () => {
  it('simple batchnorm3D, no offset or scale, 2x1x2', () => {
    const x = tf.tensor3d([2, 4, 9, 23], [2, 1, 2]);
    const mean = tf.tensor1d([1, 2]);
    const variance = tf.tensor1d([2, 3]);
    const varianceEpsilon = .001;

    const result = tf.batchNormalization3d(
        x, mean, variance, varianceEpsilon, undefined, undefined);

    expectArraysClose(result, [
      (x.get(0, 0, 0) - mean.get(0)) * 1 /
          Math.sqrt(variance.get(0) + varianceEpsilon),
      (x.get(0, 0, 1) - mean.get(1)) * 1 /
          Math.sqrt(variance.get(1) + varianceEpsilon),
      (x.get(1, 0, 0) - mean.get(0)) * 1 /
          Math.sqrt(variance.get(0) + varianceEpsilon),
      (x.get(1, 0, 1) - mean.get(1)) * 1 /
          Math.sqrt(variance.get(1) + varianceEpsilon)
    ]);
  });

  it('simple batchnorm3D, no offset, 2x1x2', () => {
    const x = tf.tensor3d([2, 4, 9, 23], [2, 1, 2]);
    const mean = tf.tensor1d([1, 2]);
    const variance = tf.tensor1d([2, 3]);
    const scale = tf.tensor1d([4, 5]);
    const varianceEpsilon = .001;

    const result = tf.batchNormalization3d(
        x, mean, variance, varianceEpsilon, scale, undefined);

    expectArraysClose(result, [
      (x.get(0, 0, 0) - mean.get(0)) * scale.get(0) /
          Math.sqrt(variance.get(0) + varianceEpsilon),
      (x.get(0, 0, 1) - mean.get(1)) * scale.get(1) /
          Math.sqrt(variance.get(1) + varianceEpsilon),
      (x.get(1, 0, 0) - mean.get(0)) * scale.get(0) /
          Math.sqrt(variance.get(0) + varianceEpsilon),
      (x.get(1, 0, 1) - mean.get(1)) * scale.get(1) /
          Math.sqrt(variance.get(1) + varianceEpsilon)
    ]);
  });

  it('simple batchnorm3D, no scale, 2x1x2', () => {
    const x = tf.tensor3d([2, 4, 9, 23], [2, 1, 2]);
    const mean = tf.tensor1d([1, 2]);
    const variance = tf.tensor1d([2, 3]);
    const offset = tf.tensor1d([4, 5]);

    const varianceEpsilon = .001;

    const result = tf.batchNormalization3d(
        x, mean, variance, varianceEpsilon, undefined, offset);

    expectArraysClose(result, [
      offset.get(0) +
          (x.get(0, 0, 0) - mean.get(0)) * 1 /
              Math.sqrt(variance.get(0) + varianceEpsilon),
      offset.get(1) +
          (x.get(0, 0, 1) - mean.get(1)) * 1 /
              Math.sqrt(variance.get(1) + varianceEpsilon),
      offset.get(0) +
          (x.get(1, 0, 0) - mean.get(0)) * 1 /
              Math.sqrt(variance.get(0) + varianceEpsilon),
      offset.get(1) +
          (x.get(1, 0, 1) - mean.get(1)) * 1 /
              Math.sqrt(variance.get(1) + varianceEpsilon)
    ]);
  });

  it('simple batchnorm3D, 2x1x2', () => {
    const x = tf.tensor3d([2, 4, 9, 23], [2, 1, 2]);
    const mean = tf.tensor1d([1, 2]);
    const variance = tf.tensor1d([2, 3]);
    const offset = tf.tensor1d([3, 4]);
    const scale = tf.tensor1d([4, 5]);

    const varianceEpsilon = .001;

    const result = tf.batchNormalization3d(
        x, mean, variance, varianceEpsilon, scale, offset);

    expectArraysClose(result, [
      offset.get(0) +
          (x.get(0, 0, 0) - mean.get(0)) * scale.get(0) /
              Math.sqrt(variance.get(0) + varianceEpsilon),
      offset.get(1) +
          (x.get(0, 0, 1) - mean.get(1)) * scale.get(1) /
              Math.sqrt(variance.get(1) + varianceEpsilon),
      offset.get(0) +
          (x.get(1, 0, 0) - mean.get(0)) * scale.get(0) /
              Math.sqrt(variance.get(0) + varianceEpsilon),
      offset.get(1) +
          (x.get(1, 0, 1) - mean.get(1)) * scale.get(1) /
              Math.sqrt(variance.get(1) + varianceEpsilon)
    ]);
  });

  it('batchnorm3D, x,mean,var,offset,scale are all 3D', () => {
    const shape: [number, number, number] = [2, 1, 2];
    const x = tf.tensor3d([2, 4, 9, 23], shape);
    const mean = tf.tensor3d([1, 2, 3, 4], shape);
    const variance = tf.tensor3d([2, 3, 4, 5], shape);
    const offset = tf.tensor3d([3, 4, 5, 6], shape);
    const scale = tf.tensor3d([4, 5, 6, 7], shape);

    const varianceEpsilon = .001;

    const result = tf.batchNormalization3d(
        x, mean, variance, varianceEpsilon, scale, offset);

    expectArraysClose(result, [
      offset.get(0, 0, 0) +
          (x.get(0, 0, 0) - mean.get(0, 0, 0)) * scale.get(0, 0, 0) /
              Math.sqrt(variance.get(0, 0, 0) + varianceEpsilon),
      offset.get(0, 0, 1) +
          (x.get(0, 0, 1) - mean.get(0, 0, 1)) * scale.get(0, 0, 1) /
              Math.sqrt(variance.get(0, 0, 1) + varianceEpsilon),
      offset.get(1, 0, 0) +
          (x.get(1, 0, 0) - mean.get(1, 0, 0)) * scale.get(1, 0, 0) /
              Math.sqrt(variance.get(1, 0, 0) + varianceEpsilon),
      offset.get(1, 0, 1) +
          (x.get(1, 0, 1) - mean.get(1, 0, 1)) * scale.get(1, 0, 1) /
              Math.sqrt(variance.get(1, 0, 1) + varianceEpsilon)
    ]);
  });

  it('simple batchnorm3D gradients, 2x1x2', () => {
    const x = tf.tensor3d([2, 4, 9, 23], [2, 1, 2]);
    const mean = tf.tensor1d([1, 2]);
    const variance = tf.tensor1d([2, 3]);
    const offset = tf.tensor1d([3, 4]);
    const scale = tf.tensor1d([2, 5]);

    const varianceEpsilon = .001;

    const dy = tf.tensor3d([1, 1, 1, 1], [2, 1, 2]);
    const gradX = tf.grad(
        (x: tf.Tensor3D) => tf.batchNormalization3d(
            x, mean, variance, varianceEpsilon, scale, offset))(x, dy);
    expectArraysClose(
        gradX, tf.tensor3d([1.414, 2.887, 1.414, 2.887], [2, 1, 2]));
    const gradMean = tf.grad(
        (mean: tf.Tensor1D) => tf.batchNormalization3d(
            x, mean, variance, varianceEpsilon, scale, offset))(mean, dy);
    expectArraysClose(gradMean, tf.tensor1d([-2.828, -5.773]));
    const gradVariance = tf.grad(
        (variance: tf.Tensor1D) => tf.batchNormalization3d(
            x, mean, variance, varianceEpsilon, scale, offset))(variance, dy);
    expectArraysClose(gradVariance, tf.tensor1d([-3.180, -11.060]));
    const gradOffset = tf.grad(
        (offset: tf.Tensor1D) => tf.batchNormalization3d(
            x, mean, variance, varianceEpsilon, scale, offset))(offset, dy);
    expectArraysClose(gradOffset, tf.onesLike(offset).mul(tf.scalar(2)));
    const gradScale = tf.grad(
        (scale: tf.Tensor1D) => tf.batchNormalization3d(
            x, mean, variance, varianceEpsilon, scale, offset))(scale, dy);
    expectArraysClose(gradScale, tf.tensor1d([6.362, 13.277]));
  });

  it('batchnorm3D gradients, same shapes in x, mean and variance', () => {
    const x = tf.tensor3d([10, 20, 30, 40], [2, 1, 2]);
    const mean = tf.tensor3d([0, 5, 10, 15], [2, 1, 2]);
    const variance = tf.tensor3d([2, 4, 6, 8], [2, 1, 2]);
    const scale = tf.tensor3d([2, 5, 2, 5], [2, 1, 2]);
    const offset = tf.tensor3d([0, 0, 0, 0], [2, 1, 2]);

    const varianceEpsilon = .001;

    const dy = tf.tensor3d([1, 1, 1, 1], [2, 1, 2]);
    const gradX = tf.grad(
        (x: tf.Tensor3D) => tf.batchNormalization3d(
            x, mean, variance, varianceEpsilon, scale, offset))(x, dy);
    expectArraysClose(
        gradX, tf.tensor3d([1.414, 2.500, 0.816, 1.768], [2, 1, 2]));
    const gradMean = tf.grad(
        (mean: tf.Tensor3D) => tf.batchNormalization3d(
            x, mean, variance, varianceEpsilon, scale, offset))(mean, dy);
    expectArraysClose(
        gradMean, tf.tensor3d([-1.414, -2.500, -0.816, -1.768], [2, 1, 2]));
    const gradVariance = tf.grad(
        (variance: tf.Tensor3D) => tf.batchNormalization3d(
            x, mean, variance, varianceEpsilon, scale, offset))(variance, dy);
    expectArraysClose(
        gradVariance, tf.tensor3d([-3.533, -4.686, -1.360, -2.762], [2, 1, 2]));
    const gradOffset = tf.grad(
        (offset: tf.Tensor3D) => tf.batchNormalization3d(
            x, mean, variance, varianceEpsilon, scale, offset))(offset, dy);
    expectArraysClose(gradOffset, tf.onesLike(offset));
    const gradScale = tf.grad(
        (scale: tf.Tensor3D) => tf.batchNormalization3d(
            x, mean, variance, varianceEpsilon, scale, offset))(scale, dy);
    expectArraysClose(
        gradScale, tf.tensor3d([7.069, 7.499, 8.164, 8.838], [2, 1, 2]));
  });

  it('batchnorm matches tensorflow, 2x3x3', () => {
    const x = tf.tensor3d(
        [
          0.49955603, 0.04158615, -1.09440524, 2.03854165, -0.61578344,
          2.87533573, 1.18105987, 0.807462, 1.87888837, 2.26563962, -0.37040935,
          1.35848753, -0.75347094, 0.15683117, 0.91925946, 0.34121279,
          0.92717143, 1.89683965
        ],
        [2, 3, 3]);
    const mean = tf.tensor1d([0.39745062, -0.48062894, 0.4847822]);
    const variance = tf.tensor1d([0.32375343, 0.67117643, 1.08334653]);
    const offset = tf.tensor1d([0.69398749, -1.29056387, 0.9429723]);
    const scale = tf.tensor1d([-0.5607271, 0.9878457, 0.25181573]);
    const varianceEpsilon = .001;

    const result = tf.batchNormalization3d(
        x, mean, variance, varianceEpsilon, scale, offset);

    expectArraysClose(result, [
      0.59352049, -0.66135202, 0.5610874, -0.92077015, -1.45341019, 1.52106473,
      -0.07704776, 0.26144429, 1.28010017, -1.14422404, -1.15776136, 1.15425493,
      1.82644104, -0.52249442, 1.04803919, 0.74932291, 0.40568101, 1.2844412
    ]);
  });
});

describeWithFlags('batchNormalization2D', ALL_ENVS, () => {
  it('simple batchnorm2D, no offset or scale, 2x2', () => {
    const x = tf.tensor2d([2, 4, 9, 23], [2, 2]);
    const mean = tf.tensor1d([1, 2]);
    const variance = tf.tensor1d([2, 3]);
    const varianceEpsilon = .001;

    const result = tf.batchNormalization2d(
        x, mean, variance, varianceEpsilon, undefined, undefined);

    expectArraysClose(result, [
      (x.get(0, 0) - mean.get(0)) * 1 /
          Math.sqrt(variance.get(0) + varianceEpsilon),
      (x.get(0, 1) - mean.get(1)) * 1 /
          Math.sqrt(variance.get(1) + varianceEpsilon),
      (x.get(1, 0) - mean.get(0)) * 1 /
          Math.sqrt(variance.get(0) + varianceEpsilon),
      (x.get(1, 1) - mean.get(1)) * 1 /
          Math.sqrt(variance.get(1) + varianceEpsilon)
    ]);
  });
  it('simple batchnorm2D, no offset, 2x2', () => {
    const x = tf.tensor2d([2, 4, 9, 23], [2, 2]);
    const mean = tf.tensor1d([1, 2]);
    const variance = tf.tensor1d([2, 3]);
    const scale = tf.tensor1d([4, 5]);
    const varianceEpsilon = .001;

    const result = tf.batchNormalization2d(
        x, mean, variance, varianceEpsilon, scale, undefined);

    expectArraysClose(result, [
      (x.get(0, 0) - mean.get(0)) * scale.get(0) /
          Math.sqrt(variance.get(0) + varianceEpsilon),
      (x.get(0, 1) - mean.get(1)) * scale.get(1) /
          Math.sqrt(variance.get(1) + varianceEpsilon),
      (x.get(1, 0) - mean.get(0)) * scale.get(0) /
          Math.sqrt(variance.get(0) + varianceEpsilon),
      (x.get(1, 1) - mean.get(1)) * scale.get(1) /
          Math.sqrt(variance.get(1) + varianceEpsilon)
    ]);
  });

  it('simple batchnorm2D, no scale, 2x2', () => {
    const x = tf.tensor2d([2, 4, 9, 23], [2, 2]);
    const mean = tf.tensor1d([1, 2]);
    const variance = tf.tensor1d([2, 3]);
    const offset = tf.tensor1d([4, 5]);

    const varianceEpsilon = .001;

    const result = tf.batchNormalization2d(
        x, mean, variance, varianceEpsilon, undefined, offset);

    expectArraysClose(result, [
      offset.get(0) +
          (x.get(0, 0) - mean.get(0)) * 1 /
              Math.sqrt(variance.get(0) + varianceEpsilon),
      offset.get(1) +
          (x.get(0, 1) - mean.get(1)) * 1 /
              Math.sqrt(variance.get(1) + varianceEpsilon),
      offset.get(0) +
          (x.get(1, 0) - mean.get(0)) * 1 /
              Math.sqrt(variance.get(0) + varianceEpsilon),
      offset.get(1) +
          (x.get(1, 1) - mean.get(1)) * 1 /
              Math.sqrt(variance.get(1) + varianceEpsilon)
    ]);
  });

  it('simple batchnorm2D, 2x2', () => {
    const x = tf.tensor2d([2, 4, 9, 23], [2, 2]);
    const mean = tf.tensor1d([1, 2]);
    const variance = tf.tensor1d([2, 3]);
    const offset = tf.tensor1d([3, 4]);
    const scale = tf.tensor1d([4, 5]);

    const varianceEpsilon = .001;

    const result = tf.batchNormalization2d(
        x, mean, variance, varianceEpsilon, scale, offset);

    expectArraysClose(result, [
      offset.get(0) +
          (x.get(0, 0) - mean.get(0)) * scale.get(0) /
              Math.sqrt(variance.get(0) + varianceEpsilon),
      offset.get(1) +
          (x.get(0, 1) - mean.get(1)) * scale.get(1) /
              Math.sqrt(variance.get(1) + varianceEpsilon),
      offset.get(0) +
          (x.get(1, 0) - mean.get(0)) * scale.get(0) /
              Math.sqrt(variance.get(0) + varianceEpsilon),
      offset.get(1) +
          (x.get(1, 1) - mean.get(1)) * scale.get(1) /
              Math.sqrt(variance.get(1) + varianceEpsilon)
    ]);
  });

  it('simple batchnorm2D gradients, 2x2', () => {
    const x = tf.tensor2d([2, 4, 9, 23], [2, 2]);
    const mean = tf.tensor1d([1, 2]);
    const variance = tf.tensor1d([2, 3]);
    const offset = tf.tensor1d([3, 4]);
    const scale = tf.tensor1d([2, 5]);

    const varianceEpsilon = .001;

    const dy = tf.tensor2d([1, 1, 1, 1], [2, 2]);
    const gradX = tf.grad(
        (x: tf.Tensor2D) => tf.batchNormalization2d(
            x, mean, variance, varianceEpsilon, scale, offset))(x, dy);
    expectArraysClose(gradX, tf.tensor2d([1.414, 2.887, 1.414, 2.887], [2, 2]));
    const gradMean = tf.grad(
        (mean: tf.Tensor1D) => tf.batchNormalization2d(
            x, mean, variance, varianceEpsilon, scale, offset))(mean, dy);
    expectArraysClose(gradMean, tf.tensor1d([-2.828, -5.773]));
    const gradVariance = tf.grad(
        (variance: tf.Tensor1D) => tf.batchNormalization2d(
            x, mean, variance, varianceEpsilon, scale, offset))(variance, dy);
    expectArraysClose(gradVariance, tf.tensor1d([-3.180, -11.060]));
    const gradOffset = tf.grad(
        (offset: tf.Tensor1D) => tf.batchNormalization2d(
            x, mean, variance, varianceEpsilon, scale, offset))(offset, dy);
    expectArraysClose(gradOffset, tf.onesLike(offset).mul(tf.scalar(2)));
    const gradScale = tf.grad(
        (scale: tf.Tensor1D) => tf.batchNormalization2d(
            x, mean, variance, varianceEpsilon, scale, offset))(scale, dy);
    expectArraysClose(gradScale, tf.tensor1d([6.362, 13.277]));
  });

  it('batchnorm2D gradients, same shapes in x, mean and variance', () => {
    const x = tf.tensor2d([10, 20, 30, 40], [2, 2]);
    const mean = tf.tensor2d([0, 5, 10, 15], [2, 2]);
    const variance = tf.tensor2d([2, 4, 6, 8], [2, 2]);
    const scale = tf.tensor2d([2, 5, 2, 5], [2, 2]);
    const offset = tf.tensor2d([0, 0, 0, 0], [2, 2]);

    const varianceEpsilon = .001;

    const dy = tf.tensor2d([1, 1, 1, 1], [2, 2]);
    const gradX = tf.grad(
        (x: tf.Tensor2D) => tf.batchNormalization2d(
            x, mean, variance, varianceEpsilon, scale, offset))(x, dy);
    expectArraysClose(gradX, tf.tensor2d([1.414, 2.500, 0.816, 1.768], [2, 2]));
    const gradMean = tf.grad(
        (mean: tf.Tensor2D) => tf.batchNormalization2d(
            x, mean, variance, varianceEpsilon, scale, offset))(mean, dy);
    expectArraysClose(
        gradMean, tf.tensor2d([-1.414, -2.500, -0.816, -1.768], [2, 2]));
    const gradVariance = tf.grad(
        (variance: tf.Tensor2D) => tf.batchNormalization2d(
            x, mean, variance, varianceEpsilon, scale, offset))(variance, dy);
    expectArraysClose(
        gradVariance, tf.tensor2d([-3.533, -4.686, -1.360, -2.762], [2, 2]));
    const gradOffset = tf.grad(
        (offset: tf.Tensor2D) => tf.batchNormalization2d(
            x, mean, variance, varianceEpsilon, scale, offset))(offset, dy);
    expectArraysClose(gradOffset, tf.onesLike(offset));
    const gradScale = tf.grad(
        (scale: tf.Tensor2D) => tf.batchNormalization2d(
            x, mean, variance, varianceEpsilon, scale, offset))(scale, dy);
    expectArraysClose(
        gradScale, tf.tensor2d([7.069, 7.499, 8.164, 8.838], [2, 2]));
  });

  it('batchnorm2D matches tensorflow, 3x3', () => {
    const x = tf.tensor2d(
        [
          0.3136892, 0.92389025, 0.594782, 0.05021042, 0.67545404, 0.93910035,
          0.13277993, 0.96474269, 0.88608916
        ],
        [3, 3]);
    const mean = tf.tensor1d([0.19526312, 0.74857256, 0.45166398]);
    const variance = tf.tensor1d([0.22963001, 0.61521992, 0.46623685]);
    const offset = tf.tensor1d([0.43098484, 0.77712237, 0.47916298]);
    const scale = tf.tensor1d([0.62186907, 0.85673736, 0.19201061]);
    const varianceEpsilon = .001;

    const result = tf.batchNormalization2d(
        x, mean, variance, varianceEpsilon, scale, offset);

    expectArraysClose(result, [
      0.58433646, 0.96846228, 0.51936529, 0.24315402, 0.69732157, 0.61608542,
      0.35007446, 1.01304821, 0.60119441
    ]);
  });

  it('throws when passed x as a non-tensor', () => {
    const mean = tf.tensor1d([1, 2]);
    const variance = tf.tensor1d([2, 3]);

    expect(() => tf.batchNormalization({} as tf.Tensor, mean, variance))
        .toThrowError(
            /Argument 'x' passed to 'batchNormalization' must be a Tensor/);
  });
  it('throws when passed mean as a non-tensor', () => {
    const x = tf.tensor4d([2, 4, 9, 23], [2, 1, 1, 2]);
    const variance = tf.tensor1d([2, 3]);

    expect(() => tf.batchNormalization(x, {} as tf.Tensor, variance))
        .toThrowError(
            /Argument 'mean' passed to 'batchNormalization' must be a Tensor/);
  });
  it('throws when passed variance as a non-tensor', () => {
    const x = tf.tensor4d([2, 4, 9, 23], [2, 1, 1, 2]);
    const mean = tf.tensor1d([1, 2]);

    const e =
        /Argument 'variance' passed to 'batchNormalization' must be a Tensor/;
    expect(() => tf.batchNormalization(x, mean, {} as tf.Tensor))
        .toThrowError(e);
  });
  it('throws when passed scale as a non-tensor', () => {
    const x = tf.tensor4d([2, 4, 9, 23], [2, 1, 1, 2]);
    const mean = tf.tensor1d([1, 2]);
    const variance = tf.tensor1d([2, 3]);
    const epsilon = .001;

    expect(
        () =>
            tf.batchNormalization(x, mean, variance, epsilon, {} as tf.Tensor))
        .toThrowError(
            /Argument 'scale' passed to 'batchNormalization' must be a Tensor/);
  });
  it('throws when passed offset as a non-tensor', () => {
    const x = tf.tensor4d([2, 4, 9, 23], [2, 1, 1, 2]);
    const mean = tf.tensor1d([1, 2]);
    const variance = tf.tensor1d([2, 3]);
    const epsilon = .001;
    const scale = tf.tensor1d([0.62186907, 0.85673736, 0.19201061]);

    const e =
        /Argument 'offset' passed to 'batchNormalization' must be a Tensor/;
    expect(
        () => tf.batchNormalization(
            x, mean, variance, epsilon, scale, {} as tf.Tensor))
        .toThrowError(e);
  });
});
