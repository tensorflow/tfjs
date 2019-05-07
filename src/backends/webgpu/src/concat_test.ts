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

describe('concat', () => {
  beforeAll(async () => tfwebgpu.ready);

  it('3 + 5', async () => {
    const a = tf.tensor1d([3]);
    const b = tf.tensor1d([5]);

    const result = tf.concat1d([a, b]);
    const expected = [3, 5];
    const resultData = await result.data();
    tf.test_util.expectArraysClose(resultData, new Float32Array(expected));
  });

  it('[[3]] + [[5]], axis=0', async () => {
    const axis = 0;
    const a = tf.tensor2d([3], [1, 1]);
    const b = tf.tensor2d([5], [1, 1]);

    const result = tf.concat2d([a, b], axis);
    const expected = [3, 5];

    expect(result.shape).toEqual([2, 1]);
    const resultData = await result.data();
    tf.test_util.expectArraysClose(resultData, new Float32Array(expected));
  });

  it('[[1, 2],[3, 4]] + [[5, 6],[7, 8]] + [[9, 10],[11, 12]], axis=1',
     async () => {
       const axis = 1;
       const a = tf.tensor2d([[1, 2], [3, 4]]);
       const b = tf.tensor2d([[5, 6], [7, 8]]);
       const c = tf.tensor2d([[9, 10], [11, 12]]);

       const result = tf.concat2d([a, b, c], axis);
       const expected = [1, 2, 5, 6, 9, 10, 3, 4, 7, 8, 11, 12];

       expect(result.shape).toEqual([2, 6]);
       const resultData = await result.data();
       tf.test_util.expectArraysClose(resultData, new Float32Array(expected));
     });

  it('concat axis=2', async () => {
    const tensor1 = tf.tensor3d([1, 11, 2, 22, 3, 33, 4, 44], [2, 2, 2]);
    const tensor2 = tf.tensor3d(
        [5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888], [2, 2, 3]);
    const values = tf.concat3d([tensor1, tensor2], 2);
    expect(values.shape).toEqual([2, 2, 5]);
    const valuesData = await values.data();
    tf.test_util.expectArraysClose(valuesData, new Float32Array([
                                     1, 11, 5, 55, 555, 2, 22, 6, 66, 666,
                                     3, 33, 7, 77, 777, 4, 44, 8, 88, 888
                                   ]));
  });
});