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

describe('transpose', () => {
  beforeAll(async () => tfwebgpu.ready);

  it('2D (transpose)', async () => {
    const t = tf.tensor2d([1, 11, 2, 22, 3, 33, 4, 44], [2, 4]);
    const t2 = tf.transpose(t, [1, 0]);

    expect(t2.shape).toEqual([4, 2]);
    const result = await t2.data();
    tf.test_util.expectArraysClose(
        result, new Float32Array([1, 3, 11, 33, 2, 4, 22, 44]));
  });

  it('3D [r, c, d] => [d, r, c]', async () => {
    const t = tf.tensor3d([1, 11, 2, 22, 3, 33, 4, 44], [2, 2, 2]);
    const t2 = tf.transpose(t, [2, 0, 1]);

    expect(t2.shape).toEqual([2, 2, 2]);
    const result = await t2.data();
    tf.test_util.expectArraysClose(
        result, new Float32Array([1, 2, 3, 4, 11, 22, 33, 44]));
  });
});