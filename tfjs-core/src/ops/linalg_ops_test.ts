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
import {Tensor1D, Tensor2D} from '../tensor';
import {expectArraysClose} from '../test_util';

import {scalar, tensor1d, tensor2d, tensor3d, tensor4d} from './ops';

describeWithFlags('gramSchmidt-tiny', ALL_ENVS, () => {
  it('2x2, Array of Tensor1D', async () => {
    const xs: Tensor1D[] = [
      tf.randomNormal([2], 0, 1, 'float32', 1),
      tf.randomNormal([2], 0, 1, 'float32', 2)
    ];
    const ys = tf.linalg.gramSchmidt(xs) as Tensor1D[];
    const y = tf.stack(ys) as Tensor2D;
    // Test that the results are orthogonalized and normalized.
    expectArraysClose(
        await y.transpose().matMul(y).array(), await tf.eye(2).array());
    // Test angle between xs[0] and ys[0] is zero, i.e., the orientation of the
    // first vector is kept.
    expectArraysClose(
        await tf.sum(xs[0].mul(ys[0])).array(),
        await tf.norm(xs[0]).mul(tf.norm(ys[0])).array());
  });

  it('3x3, Array of Tensor1D', async () => {
    const xs: Tensor1D[] = [
      tf.randomNormal([3], 0, 1, 'float32', 1),
      tf.randomNormal([3], 0, 1, 'float32', 2),
      tf.randomNormal([3], 0, 1, 'float32', 3)
    ];
    const ys = tf.linalg.gramSchmidt(xs) as Tensor1D[];
    const y = tf.stack(ys) as Tensor2D;
    expectArraysClose(
        await y.transpose().matMul(y).array(), await tf.eye(3).array());
    expectArraysClose(
        await tf.sum(xs[0].mul(ys[0])).array(),
        await tf.norm(xs[0]).mul(tf.norm(ys[0])).array());
  });

  it('3x3, Matrix', async () => {
    const xs = tf.randomNormal([3, 3], 0, 1, 'float32', 1) as Tensor2D;
    const y = tf.linalg.gramSchmidt(xs) as Tensor2D;
    expectArraysClose(
        await y.transpose().matMul(y).array(), await tf.eye(3).array());
  });

  it('2x3, Matrix', async () => {
    const xs = tf.randomNormal([2, 3], 0, 1, 'float32', 1) as Tensor2D;
    const y = tf.linalg.gramSchmidt(xs) as Tensor2D;
    expectArraysClose(
        await y.matMul(y.transpose()).array(), await tf.eye(2).array());
  });

  it('3x2 Matrix throws Error', () => {
    const xs = tf.tensor2d([[1, 2], [3, -1], [5, 1]]);
    expect(() => tf.linalg.gramSchmidt(xs))
        .toThrowError(
            /Number of vectors \(3\) exceeds number of dimensions \(2\)/);
  });

  it('Mismatching dimensions input throws Error', () => {
    const xs: Tensor1D[] =
        [tf.tensor1d([1, 2, 3]), tf.tensor1d([-1, 5, 1]), tf.tensor1d([0, 0])];

    expect(() => tf.linalg.gramSchmidt(xs)).toThrowError(/Non-unique/);
  });

  it('Empty input throws Error', () => {
    expect(() => tf.linalg.gramSchmidt([])).toThrowError(/empty/);
  });
});

describeWithFlags('qr', ALL_ENVS, () => {
  it('1x1', async () => {
    const x = tensor2d([[10]], [1, 1]);
    const [q, r] = tf.linalg.qr(x);
    expectArraysClose(await q.array(), [[-1]]);
    expectArraysClose(await r.array(), [[-10]]);
  });

  it('2x2', async () => {
    const x = tensor2d([[1, 3], [-2, -4]], [2, 2]);
    const [q, r] = tf.linalg.qr(x);
    expectArraysClose(await q.array(), [[-0.4472, -0.8944], [0.8944, -0.4472]]);
    expectArraysClose(await r.array(), [[-2.2361, -4.9193], [0, -0.8944]]);
  });

  it('2x2x2', async () => {
    const x = tensor3d([[[-1, -3], [2, 4]], [[1, 3], [-2, -4]]], [2, 2, 2]);
    const [q, r] = tf.linalg.qr(x);
    expectArraysClose(await q.array(), [
      [[-0.4472, -0.8944], [0.8944, -0.4472]],
      [[-0.4472, -0.8944], [0.8944, -0.4472]]
    ]);
    expectArraysClose(
        await r.array(),
        [[[2.2361, 4.9193], [0, 0.8944]], [[-2.2361, -4.9193], [0, -0.8944]]]);
  });

  it('2x1x2x2', async () => {
    const x =
        tensor4d([[[[-1, -3], [2, 4]]], [[[1, 3], [-2, -4]]]], [2, 1, 2, 2]);
    const [q, r] = tf.linalg.qr(x);
    expectArraysClose(await q.array(), [
      [[[-0.4472, -0.8944], [0.8944, -0.4472]]],
      [[[-0.4472, -0.8944], [0.8944, -0.4472]]],
    ]);
    expectArraysClose(await r.array(), [
      [[[2.2361, 4.9193], [0, 0.8944]]], [[[-2.2361, -4.9193], [0, -0.8944]]]
    ]);
  });

  it('3x3', async () => {
    const x = tensor2d([[1, 3, 2], [-2, 0, 7], [8, -9, 4]], [3, 3]);
    const [q, r] = tf.linalg.qr(x);
    expectArraysClose(await q.array(), [
      [-0.1204, 0.8729, 0.4729], [0.2408, -0.4364, 0.8669],
      [-0.9631, -0.2182, 0.1576]
    ]);
    expectArraysClose(
        await r.array(),
        [[-8.3066, 8.3066, -2.4077], [0, 4.5826, -2.1822], [0, 0, 7.6447]]);
  });

  it('3x3, zero on diagonal', async () => {
    const x = tensor2d([[0, 2, 2], [1, 1, 1], [0, 1, 2]], [3, 3]);
    const [q, r] = tf.linalg.qr(x);
    expectArraysClose(await q.data(), [
      [0., -0.89442719, 0.4472136], [1., 0., 0.], [0., -0.4472136, -0.89442719]
    ]);
    expectArraysClose(
        await r.data(),
        [[1., 1., 1.], [0., -2.23606798, -2.68328157], [0., 0., -0.89442719]]);
  });

  it('3x2, fullMatrices = default false', async () => {
    const x = tensor2d([[1, 2], [3, -3], [-2, 1]], [3, 2]);
    const [q, r] = tf.linalg.qr(x);
    expectArraysClose(
        await q.array(),
        [[-0.2673, 0.9221], [-0.8018, -0.3738], [0.5345, -0.0997]]);
    expectArraysClose(await r.array(), [[-3.7417, 2.4054], [0, 2.8661]]);
  });

  it('3x2, fullMatrices = true', async () => {
    const x = tensor2d([[1, 2], [3, -3], [-2, 1]], [3, 2]);
    const [q, r] = tf.linalg.qr(x, true);
    expectArraysClose(await q.array(), [
      [-0.2673, 0.9221, 0.2798], [-0.8018, -0.3738, 0.4663],
      [0.5345, -0.0997, 0.8393]
    ]);
    expectArraysClose(
        await r.array(), [[-3.7417, 2.4054], [0, 2.8661], [0, 0]]);
  });

  it('2x3, fullMatrices = default false', async () => {
    const x = tensor2d([[1, 2, 3], [-3, -2, 1]], [2, 3]);
    const [q, r] = tf.linalg.qr(x);
    expectArraysClose(
        await q.array(), [[-0.3162278, -0.9486833], [0.9486833, -0.31622773]]);
    expectArraysClose(
        await r.array(),
        [[-3.162, -2.5298, -2.3842e-07], [0, -1.2649, -3.162]]);
  });

  it('2x3, fullMatrices = true', async () => {
    const x = tensor2d([[1, 2, 3], [-3, -2, 1]], [2, 3]);
    const [q, r] = tf.linalg.qr(x, true);
    expectArraysClose(
        await q.array(), [[-0.3162278, -0.9486833], [0.9486833, -0.31622773]]);
    expectArraysClose(
        await r.array(),
        [[-3.162, -2.5298, -2.3842e-07], [0, -1.2649, -3.162]]);
  });

  it('Does not leak memory', () => {
    const x = tensor2d([[1, 3], [-2, -4]], [2, 2]);
    // The first call to qr creates and keeps internal singleton tensors.
    // Subsequent calls should always create exactly two tensors.
    tf.linalg.qr(x);
    // Count before real call.
    const numTensors = tf.memory().numTensors;
    tf.linalg.qr(x);
    expect(tf.memory().numTensors).toEqual(numTensors + 2);
  });

  it('Insuffient input tensor rank leads to error', () => {
    const x1 = scalar(12);
    expect(() => tf.linalg.qr(x1)).toThrowError(/rank >= 2.*got rank 0/);
    const x2 = tensor1d([12]);
    expect(() => tf.linalg.qr(x2)).toThrowError(/rank >= 2.*got rank 1/);
  });
});
