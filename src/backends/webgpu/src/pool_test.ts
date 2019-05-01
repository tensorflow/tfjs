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

import * as tf from '@tensorflow/tfjs-core';

import * as tfwebgpu from './index';

describe('pool', () => {
  beforeAll(async () => tfwebgpu.ready);

  it('x=[1,1,1] f=[1,1] s=1 [0] => [0]', async () => {
    const x = tf.tensor3d([0], [1, 1, 1]);

    const result = tf.maxPool(x, 1, 1, 0);
    const resultData = await result.data();

    tf.test_util.expectArraysClose(resultData, new Float32Array([0]));
  });

  it('x=[3,3,1] f=[2,2] s=1', async () => {
    const x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 9, 8], [3, 3, 1]);

    const result = tf.maxPool(x, 2, 1, 'same');
    const resultData = await result.data();

    tf.test_util.expectArraysClose(
        resultData, new Float32Array([5, 6, 6, 9, 9, 8, 9, 9, 8]));
  });

  it('x=[2,3,3,1] f=[2,2] s=1', async () => {
    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 9, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 3, 3, 1]);

    const result = tf.maxPool(x, 2, 1, 0);
    const resultData = await result.data();

    tf.test_util.expectArraysClose(
        resultData, new Float32Array([5, 6, 9, 9, 5, 6, 8, 9]));
  });

  it('x=[3,3,2] f=[2,2] s=1', async () => {
    const x = tf.tensor3d(
        [1, 99, 2, 88, 3, 77, 4, 66, 5, 55, 6, 44, 7, 33, 9, 22, 8, 11],
        [3, 3, 2]);

    const result = tf.maxPool(x, 2, 1, 0);
    const resultData = await result.data();

    expect(result.shape).toEqual([2, 2, 2]);
    tf.test_util.expectArraysClose(
        resultData, new Float32Array([5, 99, 6, 88, 9, 66, 9, 55]));
  });

  it('x=[4,4,1] f=[2,2] s=2', async () => {
    const x = tf.tensor3d(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [4, 4, 1]);

    const result = tf.maxPool(x, 2, 2, 0);
    const resultData = await result.data();

    expect(result.shape).toEqual([2, 2, 1]);
    tf.test_util.expectArraysClose(
        resultData, new Float32Array([5, 7, 13, 15]));
  });

  it('x=[2,2,1] f=[2,2] s=1 p=same', async () => {
    const x = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
    const fSize = 2;
    const strides = 1;
    const result = tf.maxPool(x, fSize, strides, 'same');
    const resultData = await result.data();
    expect(result.shape).toEqual([2, 2, 1]);
    tf.test_util.expectArraysClose(resultData, new Float32Array([4, 4, 4, 4]));
  });

  it('x=[2,2,3] f=[1,1] s=2 p=1 dimRoundingMode=floor', () => {
    const x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 2, 3]);
    const result = tf.maxPool(x, 1, 2, 1, 'floor');

    expect(result.shape).toEqual([2, 2, 3]);
  });
});