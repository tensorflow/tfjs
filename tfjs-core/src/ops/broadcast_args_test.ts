/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
import {expectArraysEqual} from '../test_util';

describeWithFlags('broadcastArgs', ALL_ENVS, () => {
  it('([1,1], [1,1]) -> [1,1]', async () => {
    const s1 = tf.tensor1d([1, 1], 'int32');
    const s2 = tf.tensor1d([1, 1], 'int32');
    const expected = [1, 1];

    expectArraysEqual(expected, await tf.broadcastArgs(s1, s2).array());
  });

  it('([1,1], [4,2]) -> [4,2]', async () => {
    const s1 = tf.tensor1d([1, 1], 'int32');
    const s2 = tf.tensor1d([4, 2], 'int32');
    const expected = [4, 2];

    expectArraysEqual(expected, await tf.broadcastArgs(s1, s2).array());
  });

  it('([1,6], [3,1]) -> [3,6]', async () => {
    const s1 = tf.tensor1d([1, 6], 'int32');
    const s2 = tf.tensor1d([3, 1], 'int32');
    const expected = [3, 6];

    expectArraysEqual(expected, await tf.broadcastArgs(s1, s2).array());
  });

  it('([1,6], [3,1,1,1]) -> [3,1,1,6]', async () => {
    const s1 = tf.tensor1d([1, 6], 'int32');
    const s2 = tf.tensor1d([3, 1, 1, 1], 'int32');
    const expected = [3, 1, 1, 6];

    expectArraysEqual(expected, await tf.broadcastArgs(s1, s2).array());
  });

  it('([1,6,-1], [3,1,1,1]) -> [3,1,6,-1]', async () => {
    const s1 = tf.tensor1d([1, 6, -1], 'int32');
    const s2 = tf.tensor1d([3, 1, 1, 1], 'int32');
    const expected = [3, 1, 6, -1];

    expectArraysEqual(expected, await tf.broadcastArgs(s1, s2).array());
  });

  it('([1,2], [1,3]) -> error', async () => {
    const s1 = tf.tensor1d([1, 2], 'int32');
    const s2 = tf.tensor1d([1, 3], 'int32');

    expect(() => tf.broadcastArgs(s1, s2).arraySync()).toThrowError();
  });

  it('([[1,1],[1,1]], [[1,1],[1,1]]) -> error', async () => {
    const s1 = tf.tensor2d([[1, 1], [1, 1]], [2, 2], 'int32');
    const s2 = tf.tensor2d([[1, 1], [1, 1]], [2, 2], 'int32');

    expect(() => tf.broadcastArgs(s1, s2).arraySync()).toThrowError();
  });
});
