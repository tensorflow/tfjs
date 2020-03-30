/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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
import {expectArraysClose} from '../test_util';

describeWithFlags('batchNorm3D', ALL_ENVS, () => {
  it('simple batchnorm3D, no offset or scale, 2x1x2', async () => {
    const xT = tf.tensor3d([2, 4, 9, 23], [2, 1, 2]);
    const meanT = tf.tensor1d([1, 2]);
    const varianceT = tf.tensor1d([2, 3]);
    const varianceEpsilon = .001;

    const result = tf.batchNorm3d(
      xT, meanT, varianceT, undefined, undefined, varianceEpsilon);
    const x = await xT.buffer();
    const mean = await meanT.buffer();
    const variance = await varianceT.buffer();
    expectArraysClose(await result.data(), [
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

  it('simple batchnorm3D, no offset, 2x1x2', async () => {
    const xT = tf.tensor3d([2, 4, 9, 23], [2, 1, 2]);
    const meanT = tf.tensor1d([1, 2]);
    const varianceT = tf.tensor1d([2, 3]);
    const scaleT = tf.tensor1d([4, 5]);
    const varianceEpsilon = .001;

    const result = tf.batchNorm3d(
      xT, meanT, varianceT, undefined, scaleT, varianceEpsilon);

    const x = await xT.buffer();
    const mean = await meanT.buffer();
    const variance = await varianceT.buffer();
    const scale = await scaleT.buffer();
    expectArraysClose(await result.data(), [
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

  it('simple batchnorm3D, no scale, 2x1x2', async () => {
    const xT = tf.tensor3d([2, 4, 9, 23], [2, 1, 2]);
    const meanT = tf.tensor1d([1, 2]);
    const varianceT = tf.tensor1d([2, 3]);
    const offsetT = tf.tensor1d([4, 5]);

    const varianceEpsilon = .001;

    const result = tf.batchNorm3d(
      xT, meanT, varianceT, offsetT, undefined, varianceEpsilon);

    const x = await xT.buffer();
    const mean = await meanT.buffer();
    const variance = await varianceT.buffer();
    const offset = await offsetT.buffer();
    expectArraysClose(await result.data(), [
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

  it('simple batchnorm3D, 2x1x2', async () => {
    const xT = tf.tensor3d([2, 4, 9, 23], [2, 1, 2]);
    const meanT = tf.tensor1d([1, 2]);
    const varianceT = tf.tensor1d([2, 3]);
    const offsetT = tf.tensor1d([3, 4]);
    const scaleT = tf.tensor1d([4, 5]);

    const varianceEpsilon = .001;

    const result =
      tf.batchNorm3d(xT, meanT, varianceT, offsetT, scaleT, varianceEpsilon);
    const x = await xT.buffer();
    const mean = await meanT.buffer();
    const variance = await varianceT.buffer();
    const offset = await offsetT.buffer();
    const scale = await scaleT.buffer();

    expectArraysClose(await result.data(), [
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

  it('accepts a tensor-like object', async () => {
    const x = [[[2, 4]], [[9, 23]]];  // 2x1x2
    const mean = [1, 2];
    const variance = [2, 3];
    const offset = [3, 4];
    const scale = [4, 5];

    const varianceEpsilon = .001;

    const result =
      tf.batchNorm3d(x, mean, variance, offset, scale, varianceEpsilon);

    expectArraysClose(await result.data(), [
      offset[0] +
      (x[0][0][0] - mean[0]) * scale[0] /
      Math.sqrt(variance[0] + varianceEpsilon),
      offset[1] +
      (x[0][0][1] - mean[1]) * scale[1] /
      Math.sqrt(variance[1] + varianceEpsilon),
      offset[0] +
      (x[1][0][0] - mean[0]) * scale[0] /
      Math.sqrt(variance[0] + varianceEpsilon),
      offset[1] +
      (x[1][0][1] - mean[1]) * scale[1] /
      Math.sqrt(variance[1] + varianceEpsilon)
    ]);
  });

  it('batchnorm3D, x,mean,var,offset,scale are all 3D', async () => {
    const shape: [number, number, number] = [2, 1, 2];
    const xT = tf.tensor3d([2, 4, 9, 23], shape);
    const meanT = tf.tensor3d([1, 2, 3, 4], shape);
    const varianceT = tf.tensor3d([2, 3, 4, 5], shape);
    const offsetT = tf.tensor3d([3, 4, 5, 6], shape);
    const scaleT = tf.tensor3d([4, 5, 6, 7], shape);

    const varianceEpsilon = .001;

    const result =
      tf.batchNorm3d(xT, meanT, varianceT, offsetT, scaleT, varianceEpsilon);

    const x = await xT.buffer();
    const mean = await meanT.buffer();
    const variance = await varianceT.buffer();
    const offset = await offsetT.buffer();
    const scale = await scaleT.buffer();
    expectArraysClose(await result.data(), [
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

  it('simple batchnorm3D gradients, 2x1x2', async () => {
    const x = tf.tensor3d([2, 4, 9, 23], [2, 1, 2]);
    const mean = tf.tensor1d([1, 2]);
    const variance = tf.tensor1d([2, 3]);
    const offset = tf.tensor1d([3, 4]);
    const scale = tf.tensor1d([2, 5]);

    const varianceEpsilon = .001;

    const dy = tf.tensor3d([1, 1, 1, 1], [2, 1, 2]);
    const gradX = tf.grad(
      (x: tf.Tensor3D) => tf.batchNorm3d(
        x, mean, variance, offset, scale, varianceEpsilon))(x, dy);
    expectArraysClose(await gradX.data(), [1.414, 2.887, 1.414, 2.887]);
    expect(gradX.shape).toEqual([2, 1, 2]);
    const gradMean = tf.grad(
      (mean: tf.Tensor1D) => tf.batchNorm3d(
        x, mean, variance, offset, scale, varianceEpsilon))(mean, dy);
    expectArraysClose(await gradMean.data(), [-2.828, -5.773]);
    expect(gradMean.shape).toEqual([2]);
    const gradVariance = tf.grad(
      (variance: tf.Tensor1D) => tf.batchNorm3d(
        x, mean, variance, offset, scale, varianceEpsilon))(variance, dy);
    expectArraysClose(await gradVariance.data(), [-3.180, -11.060]);
    expect(gradVariance.shape).toEqual([2]);
    const gradOffset = tf.grad(
      (offset: tf.Tensor1D) => tf.batchNorm3d(
        x, mean, variance, offset, scale, varianceEpsilon))(offset, dy);
    expectArraysClose(await gradOffset.data(), [2, 2]);
    expect(gradOffset.shape).toEqual([2]);
    const gradScale = tf.grad(
      (scale: tf.Tensor1D) => tf.batchNorm3d(
        x, mean, variance, offset, scale, varianceEpsilon))(scale, dy);
    expectArraysClose(await gradScale.data(), [6.362, 13.277]);
    expect(gradScale.shape).toEqual([2]);
  });

  it('batchnorm3D gradients, same shapes in x, mean and variance', async () => {
    const x = tf.tensor3d([10, 20, 30, 40], [2, 1, 2]);
    const mean = tf.tensor3d([0, 5, 10, 15], [2, 1, 2]);
    const variance = tf.tensor3d([2, 4, 6, 8], [2, 1, 2]);
    const scale = tf.tensor3d([2, 5, 2, 5], [2, 1, 2]);
    const offset = tf.tensor3d([0, 0, 0, 0], [2, 1, 2]);

    const varianceEpsilon = .001;

    const dy = tf.tensor3d([1, 1, 1, 1], [2, 1, 2]);
    const gradX = tf.grad(
      (x: tf.Tensor3D) => tf.batchNorm3d(
        x, mean, variance, offset, scale, varianceEpsilon))(x, dy);
    expectArraysClose(await gradX.data(), [1.414, 2.500, 0.816, 1.768]);
    expect(gradX.shape).toEqual([2, 1, 2]);
    const gradMean = tf.grad(
      (mean: tf.Tensor3D) => tf.batchNorm3d(
        x, mean, variance, offset, scale, varianceEpsilon))(mean, dy);
    expectArraysClose(await gradMean.data(), [-1.414, -2.500, -0.816, -1.768]);
    expect(gradMean.shape).toEqual([2, 1, 2]);
    const gradVariance = tf.grad(
      (variance: tf.Tensor3D) => tf.batchNorm3d(
        x, mean, variance, offset, scale, varianceEpsilon))(variance, dy);
    expectArraysClose(
      await gradVariance.data(), [-3.533, -4.686, -1.360, -2.762]);
    expect(gradVariance.shape).toEqual([2, 1, 2]);
    const gradOffset = tf.grad(
      (offset: tf.Tensor3D) => tf.batchNorm3d(
        x, mean, variance, offset, scale, varianceEpsilon))(offset, dy);
    expectArraysClose(await gradOffset.data(), [1, 1, 1, 1]);
    expect(gradOffset.shape).toEqual([2, 1, 2]);
    const gradScale = tf.grad(
      (scale: tf.Tensor3D) => tf.batchNorm3d(
        x, mean, variance, offset, scale, varianceEpsilon))(scale, dy);
    expectArraysClose(await gradScale.data(), [7.069, 7.499, 8.164, 8.838]);
    expect(gradScale.shape).toEqual([2, 1, 2]);
  });

  it('batchnorm matches tensorflow, 2x3x3', async () => {
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

    const result =
      tf.batchNorm3d(x, mean, variance, offset, scale, varianceEpsilon);

    expectArraysClose(await result.data(), [
      0.59352049, -0.66135202, 0.5610874, -0.92077015, -1.45341019, 1.52106473,
      -0.07704776, 0.26144429, 1.28010017, -1.14422404, -1.15776136, 1.15425493,
      1.82644104, -0.52249442, 1.04803919, 0.74932291, 0.40568101, 1.2844412
    ]);
  });
});
