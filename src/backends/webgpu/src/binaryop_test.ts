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

describe('Binary ops', () => {
  beforeAll(async () => tfwebgpu.ready);

  it('A * B', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    const b = tf.tensor1d([3, 4, 5]);
    const c = tf.mul(a, b);

    const cData = await c.data();

    tf.test_util.expectArraysClose(cData, new Float32Array([3, 8, 15]));
  });

  it('A * B >1D', async () => {
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6], [1, 2, 3]);
    const b = tf.tensor3d([1, 2, 3, 4, 5, 6], [1, 2, 3]);
    const c = tf.mul(a, b);

    const cData = await c.data();

    tf.test_util.expectArraysClose(
        cData, new Float32Array([1, 4, 9, 16, 25, 36]));
  });

  it('A + B', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    const b = tf.tensor1d([3, 4, 5]);
    const c = tf.add(a, b);

    const cData = await c.data();

    tf.test_util.expectArraysClose(cData, new Float32Array([4, 6, 8]));
  });

  it('broadcasts 3D and 1D', async () => {
    const a = tf.tensor3d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [2, 3, 2]);
    const b = tf.tensor1d([2]);
    const c = tf.mul(a, b);

    const cData = await c.data();

    tf.test_util.expectArraysClose(
        cData, new Float32Array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]));
  });

  it('floor division', async () => {
    const a = tf.tensor1d([-6, -6, -5, -4, -3, -3, 3, 3, 2]);
    const c = tf.tensor1d([-2, 2, 3, 2, -3, 3, 2, 3, 2]);

    const r = tf.floorDiv(a, c);

    const rData = await r.data();
    tf.test_util.expectArraysClose(
        rData, new Float32Array([3, -3, -2, -2, 1, -1, 1, 1, 1]));
  });

  it('floor division broadcasts', async () => {
    const a = tf.tensor1d([-5, -4, 3, 2]);
    const c = tf.scalar(2);

    const r = tf.floorDiv(a, c);

    const rData = await r.data();
    tf.test_util.expectArraysClose(rData, new Float32Array([-3, -2, 1, 1]));
  });
});
