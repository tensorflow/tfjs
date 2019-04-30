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

describe('pad', () => {
  beforeAll(async () => tfwebgpu.ready);

  it('Should pad 1D arrays', async () => {
    const a = tf.tensor1d([1, 2, 3, 4, 5, 6], 'int32');
    const b = tf.pad1d(a, [2, 3]);
    const bData = await b.data();

    tf.test_util.expectArraysClose(
        bData, new Float32Array([0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0]));
  });

  it('Should pad 2D arrays', async () => {
    const a = tf.tensor2d([[1], [2]], [2, 1], 'int32');
    const b = tf.pad2d(a, [[1, 1], [1, 1]]);
    // 0, 0, 0
    // 0, 1, 0
    // 0, 2, 0
    // 0, 0, 0
    const bData = await b.data();
    tf.test_util.expectArraysClose(
        bData, new Float32Array([0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0]));
  });

  it('should pad larger 2D arrays', async () => {
    const a = tf.tensor2d([[1, 2, 3], [4, 5, 6]], [2, 3], 'int32');
    const b = tf.pad2d(a, [[2, 2], [1, 1]]);
    const bData = await b.data();
    // 0, 0, 0, 0, 0
    // 0, 0, 0, 0, 0
    // 0, 1, 2, 3, 0
    // 0, 4, 5, 6, 0
    // 0, 0, 0, 0, 0
    // 0, 0, 0, 0, 0
    tf.test_util.expectArraysClose(
        bData, new Float32Array([
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0,
          0, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]));
  });

  it('should pad 3D arrays', async () => {
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6], [3, 1, 2]);
    const b = tf.pad(a, [[1, 0], [0, 1], [2, 2]]);
    const bData = await b.data();

    tf.test_util.expectArraysClose(
        bData, new Float32Array([
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0
        ]));
  });
});
