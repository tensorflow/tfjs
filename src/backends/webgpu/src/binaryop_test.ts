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
  beforeAll(async () => await tfwebgpu.ready);

  it('A * B', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    const b = tf.tensor1d([3, 4, 5]);
    const c = tf.mul(a, b);

    const cData = await c.data();

    tf.test_util.expectArraysClose(cData, new Float32Array([3, 8, 15]));
  });

  it('A + B', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    const b = tf.tensor1d([3, 4, 5]);
    const c = tf.add(a, b);

    const cData = await c.data();

    tf.test_util.expectArraysClose(cData, new Float32Array([4, 6, 8]));
  });
});