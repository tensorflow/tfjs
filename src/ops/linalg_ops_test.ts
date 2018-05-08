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
import {describeWithFlags} from '../jasmine_util';
import {Tensor1D, Tensor2D} from '../tensor';
import {ALL_ENVS, expectArraysClose, WEBGL_ENVS} from '../test_util';

describeWithFlags('gramSchmidt-tiny', ALL_ENVS, () => {
  it('2x2, Array of Tensor1D', () => {
    const xs: Tensor1D[] = [tf.randomNormal([2]), tf.randomNormal([2])];
    const ys = tf.linalg.gramSchmidt(xs) as Tensor1D[];
    const y = tf.stack(ys) as Tensor2D;
    // Test that the results are orthogonalized and normalized.
    expectArraysClose(y.transpose().matMul(y), tf.eye(2));
    // Test angle between xs[0] and ys[0] is zero, i.e., the orientation of the
    // first vector is kept.
    expectArraysClose(
        tf.sum(xs[0].mul(ys[0])), tf.norm(xs[0]).mul(tf.norm(ys[0])));
  });

  it('3x3, Array of Tensor1D', () => {
    const xs: Tensor1D[] =
        [tf.randomNormal([3]), tf.randomNormal([3]), tf.randomNormal([3])];
    const ys = tf.linalg.gramSchmidt(xs) as Tensor1D[];
    const y = tf.stack(ys) as Tensor2D;
    expectArraysClose(y.transpose().matMul(y), tf.eye(3));
    expectArraysClose(
        tf.sum(xs[0].mul(ys[0])), tf.norm(xs[0]).mul(tf.norm(ys[0])));
  });

  it('3x3, Matrix', () => {
    const xs = tf.randomNormal([3, 3]) as Tensor2D;
    const y = tf.linalg.gramSchmidt(xs) as Tensor2D;
    expectArraysClose(y.transpose().matMul(y), tf.eye(3));
  });

  it('2x3, Matrix', () => {
    const xs = tf.randomNormal([2, 3]) as Tensor2D;
    const y = tf.linalg.gramSchmidt(xs) as Tensor2D;
    expectArraysClose(y.matMul(y.transpose()), tf.eye(2));
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

// For operations on non-trivial matrix sizes, we skip the CPU-only ENV and use
// only WebGL ENVs.
describeWithFlags('gramSchmidt-non-tiny', WEBGL_ENVS, () => {
  it('32x512', () => {
    // Part of this test's point is that operation on a matrix of this size
    // can complete in the timeout limit of the unit test.
    const xs = tf.randomUniform([32, 512]) as Tensor2D;
    const y = tf.linalg.gramSchmidt(xs) as Tensor2D;
    expectArraysClose(y.matMul(y.transpose()), tf.eye(32));
  });
});
