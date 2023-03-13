/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
import {expectArraysClose} from '../test_util';

describeWithFlags('slice4d', ALL_ENVS, () => {
  it('slices 1x1x1x1 into shape 1x1x1x1 (effectively a copy)', async () => {
    const a = tf.tensor4d([[[[5]]]], [1, 1, 1, 1]);
    const result = tf.slice4d(a, [0, 0, 0, 0], [1, 1, 1, 1]);

    expect(result.shape).toEqual([1, 1, 1, 1]);
    expectArraysClose(await result.data(), [5]);
  });

  it('slices 2x2x2x2 array into 1x2x2x2 starting at [1, 0, 0, 0]', async () => {
    const a = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 11, 22, 33, 44, 55, 66, 77, 88],
        [2, 2, 2, 2],
    );
    const result = tf.slice4d(a, [1, 0, 0, 0], [1, 2, 2, 2]);

    expect(result.shape).toEqual([1, 2, 2, 2]);
    expectArraysClose(await result.data(), [11, 22, 33, 44, 55, 66, 77, 88]);
  });

  it('slices 2x2x2x2 array into 2x1x1x1 starting at [0, 1, 1, 1]', async () => {
    const a = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 11, 22, 33, 44, 55, 66, 77, 88], [2, 2, 2, 2]);
    const result = tf.slice4d(a, [0, 1, 1, 1], [2, 1, 1, 1]);

    expect(result.shape).toEqual([2, 1, 1, 1]);
    expectArraysClose(await result.data(), [8, 88]);
  });

  it('accepts a tensor-like object', async () => {
    const a = [[[[5]]]];  // 1x1x1x1
    const result = tf.slice4d(a, [0, 0, 0, 0], [1, 1, 1, 1]);

    expect(result.shape).toEqual([1, 1, 1, 1]);
    expectArraysClose(await result.data(), [5]);
  });
});
