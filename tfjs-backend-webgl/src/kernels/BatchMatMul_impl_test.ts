/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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
import {test_util} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {MathBackendWebGL} from '../backend_webgl';
import {WEBGL_ENVS} from '../backend_webgl_test_registry';

import {batchMatMulImpl, MATMUL_SHARED_DIM_THRESHOLD} from './BatchMatMul_impl';
import {transpose} from './Transpose';

const {expectArraysClose} = test_util;

describeWithFlags('batchMatMulImpl', WEBGL_ENVS, () => {
  it('(A x B).T batch=1', async () => {
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [1, 4, 3]);
    const b = tf.tensor3d([0, 1, -3, 2, 2, 1], [1, 3, 2]);
    const transposeA = false;
    const transposeB = false;
    const transposeProduct = true;

    const result = batchMatMulImpl({
      a,
      b,
      transposeA,
      transposeB,
      backend: tf.backend() as MathBackendWebGL,
      transposeProduct,
    });

    expect(result.shape).toEqual([1, 2, 4]);
    expectArraysClose(
        tf.backend().readSync(result.dataId), [0, -3, -6, -9, 8, 20, 32, 44]);
  });

  it('(A.T x B).T batch=1', async () => {
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [1, 3, 4]);
    const b = tf.tensor3d([0, 1, -3, 2, 2, 1], [1, 3, 2]);
    const transposeA = true;
    const transposeB = false;
    const transposeProduct = true;

    const result = batchMatMulImpl({
      a,
      b,
      transposeA,
      transposeB,
      backend: tf.backend() as MathBackendWebGL,
      transposeProduct,
    });

    expect(result.shape).toEqual([1, 2, 4]);
    expectArraysClose(
        tf.backend().readSync(result.dataId), [3, 2, 1, 0, 20, 24, 28, 32]);
  });

  it('(A x B.T).T batch=1', async () => {
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [1, 4, 3]);
    const b = tf.tensor3d([0, 1, -3, 2, 2, 1], [1, 2, 3]);
    const transposeA = false;
    const transposeB = true;
    const transposeProduct = true;

    const result = batchMatMulImpl({
      a,
      b,
      transposeA,
      transposeB,
      backend: tf.backend() as MathBackendWebGL,
      transposeProduct,
    });

    expect(result.shape).toEqual([1, 2, 4]);
    expectArraysClose(
        tf.backend().readSync(result.dataId),
        [-7, -13, -19, -25, 9, 24, 39, 54]);
  });

  it('(A.T x B.T).T batch=1', async () => {
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [1, 3, 4]);
    const b = tf.tensor3d([0, 1, -3, 2, 2, 1], [1, 2, 3]);
    const transposeA = true;
    const transposeB = true;
    const transposeProduct = true;

    const result = batchMatMulImpl({
      a,
      b,
      transposeA,
      transposeB,
      backend: tf.backend() as MathBackendWebGL,
      transposeProduct,
    });

    expect(result.shape).toEqual([1, 2, 4]);
    expectArraysClose(
        tf.backend().readSync(result.dataId),
        [-22, -24, -26, -28, 21, 26, 31, 36]);
  });

  it('(A x B).T batch=1, 1-D bias', async () => {
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [1, 4, 3]);
    const b = tf.tensor3d([0, 1, -3, 2, 2, 1], [1, 3, 2]);
    const bias = tf.tensor1d([1, 2, 3, 4]);
    const transposeA = false;
    const transposeB = false;
    const transposeProduct = true;

    const result = batchMatMulImpl({
      a,
      b,
      transposeA,
      transposeB,
      backend: tf.backend() as MathBackendWebGL,
      bias,
      transposeProduct,
    });

    expect(result.shape).toEqual([1, 2, 4]);
    expectArraysClose(
        tf.backend().readSync(result.dataId), [1, -1, -3, -5, 9, 22, 35, 48]);
  });

  it('(A x B).T batch=1, 1-D preluActivationWeights', async () => {
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [1, 4, 3]);
    const b = tf.tensor3d([0, 1, -3, 2, 2, 1], [1, 3, 2]);
    const alpha = tf.tensor1d([1, 2, 3, 4]);
    const transposeA = false;
    const transposeB = false;
    const transposeProduct = true;

    const result = batchMatMulImpl({
      a,
      b,
      transposeA,
      transposeB,
      backend: tf.backend() as MathBackendWebGL,
      activation: 'prelu',
      preluActivationWeights: alpha,
      transposeProduct,
    });

    expect(result.shape).toEqual([1, 2, 4]);
    expectArraysClose(
        tf.backend().readSync(result.dataId), [0, -6, -18, -36, 8, 20, 32, 44]);
  });

  it('(A x B).T batch=1, 2-D preluActivationWeights', async () => {
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [1, 4, 3]);
    const b = tf.tensor3d([0, 1, -3, -2, -2, -1], [1, 3, 2]);
    const alpha = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
    const transposeA = false;
    const transposeB = false;
    const transposeProduct = true;

    const result = batchMatMulImpl({
      a,
      b,
      transposeA,
      transposeB,
      backend: tf.backend() as MathBackendWebGL,
      activation: 'prelu',
      preluActivationWeights: alpha,
      transposeProduct,
    });

    expect(result.shape).toEqual([1, 2, 4]);
    expectArraysClose(
        tf.backend().readSync(result.dataId),
        [-12, -54, -126, -228, -30, -72, -126, -192]);
  });

  it('(A x B).T batch=2', async () => {
    const a = tf.tensor3d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
          13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
        ],
        [2, 4, 3]);
    const b = tf.tensor3d([0, 1, -3, 2, 2, 1], [1, 3, 2]);
    const transposeA = false;
    const transposeB = false;
    const transposeProduct = true;

    const result = batchMatMulImpl({
      a,
      b,
      transposeA,
      transposeB,
      backend: tf.backend() as MathBackendWebGL,
      transposeProduct,
    });

    expect(result.shape).toEqual([2, 2, 4]);
    expectArraysClose(
        tf.backend().readSync(result.dataId),
        [0, -3, -6, -9, 8, 20, 32, 44, -12, -15, -18, -21, 56, 68, 80, 92]);
  });

  it('(A x B).T batch=1, using mul().sum()', async () => {
    const outerShapeA = 2;
    const outerShapeB = 1;
    const sharedDim = MATMUL_SHARED_DIM_THRESHOLD + 1;

    const a = tf.tensor3d(
        new Array(sharedDim * outerShapeA).fill(null).map((_, i) => i),
        [1, outerShapeA, sharedDim]);
    const b = tf.tensor3d(
        new Array(sharedDim * outerShapeB).fill(null).map((_, i) => i),
        [1, sharedDim, outerShapeB]);
    const transposeA = false;
    const transposeB = false;

    const actualResult = batchMatMulImpl({
      a,
      b,
      transposeA,
      transposeB,
      backend: tf.backend() as MathBackendWebGL,
      transposeProduct: true,
    });

    const product = batchMatMulImpl({
      a,
      b,
      transposeA,
      transposeB,
      backend: tf.backend() as MathBackendWebGL,
      transposeProduct: false,
    });

    const expectedResult = transpose({
      inputs: {x: product},
      backend: tf.backend() as MathBackendWebGL,
      attrs: {perm: [0, 2, 1]}
    });

    expect(actualResult.shape).toEqual([1, outerShapeB, outerShapeA]);
    expectArraysClose(
        tf.backend().readSync(actualResult.dataId),
        tf.backend().readSync(expectedResult.dataId));
  });
});
