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

describe('matMul', () => {
  beforeAll(async () => tfwebgpu.ready);

  it('matMul A x B odd shared dim', async () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);

    const c = tf.matMul(a, b);
    const cData = await c.data();

    expect(c.shape).toEqual([2, 2]);
    tf.test_util.expectArraysClose(cData, new Float32Array([0, 8, -3, 20]));
  });

  it('matMul A x B multiple tiles', async () => {
    const a = tf.tensor2d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 19, 20, 21, 22, 23, 24, 1, 2,  3,  4,  5,  6,  7,  8
        ],
        [8, 4]);
    const b = tf.tensor2d(
        [
          0,  1,  -3, 2, 1,  -1, 0, 5,  6, 7, 8, 0, -2, -2, 1, 9,
          11, 10, 0,  1, -3, 2,  1, -1, 1, 2, 3, 4, 5,  6,  7, 8
        ],
        [4, 8]);

    const c = tf.matMul(a, b);
    const cData = await c.data();

    expect(c.shape).toEqual([8, 8]);
    tf.test_util.expectArraysClose(
        cData, new Float32Array([
          49,  53,  25,  21,  8,   25,  33,  52,  121, 133, 57,  49,  12,
          45,  69,  136, 193, 213, 89,  77,  16,  65,  105, 220, 265, 293,
          121, 105, 20,  85,  141, 304, 337, 373, 153, 133, 24,  105, 177,
          388, 409, 453, 185, 161, 28,  125, 213, 472, 49,  53,  25,  21,
          8,   25,  33,  52,  121, 133, 57,  49,  12,  45,  69,  136
        ]));
  });

  it('matmul A x B asymmetric', async () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const b = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);

    const c = tf.matMul(a, b);
    const cData = await c.data();

    expect(c.shape).toEqual([2, 3]);
    tf.test_util.expectArraysClose(
        cData, new Float32Array([9, 12, 15, 19, 26, 33]));
  });

  it('works when chained', async () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const b = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);

    const c = tf.matMul(a, b);

    const f = tf.tensor2d([0, 1, 0.5, 0, 0.25, 2], [2, 3]);
    const d = tf.mul(c, f);

    const dData = await d.data();

    expect(d.shape).toEqual([2, 3]);
    tf.test_util.expectArraysClose(
        dData, new Float32Array([0, 12, 7.5, 0, 6.5, 66]));
  });

  it('it works in graph mode.', async () => {
    const savedFlag = tf.ENV.get('WEBGPU_IMMEDIATE_EXECUTION_ENABLED');
    tf.ENV.set('WEBGPU_IMMEDIATE_EXECUTION_ENABLED', true);
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const b = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);

    const c = tf.matMul(a, b);

    const f = tf.tensor2d([0, 1, 0.5, 0, 0.25, 2], [2, 3]);
    const d = tf.mul(c, f);

    const dData = await d.data();
    tf.test_util.expectArraysClose(
        dData, new Float32Array([0, 12, 7.5, 0, 6.5, 66]));
    tf.ENV.set('WEBGPU_IMMEDIATE_EXECUTION_ENABLED', savedFlag);
  });
});
