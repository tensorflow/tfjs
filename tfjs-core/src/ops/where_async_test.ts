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

describeWithFlags('whereAsync', ALL_ENVS, () => {
  it('1d tensor', async () => {
    const condition = tf.tensor1d([true, false, true, true], 'bool');
    const res = await tf.whereAsync(condition);
    expect(res.dtype).toBe('int32');
    expect(res.shape).toEqual([3, 1]);
    expectArraysClose(await res.data(), [0, 2, 3]);
  });

  it('2d tensor', async () => {
    const condition = tf.tensor2d(
        [[true, false, false], [false, true, true]], [2, 3], 'bool');
    const res = await tf.whereAsync(condition);
    expect(res.dtype).toBe('int32');
    expect(res.shape).toEqual([3, 2]);
    expectArraysClose(await res.data(), [0, 0, 1, 1, 1, 2]);
  });

  it('3d tensor', async () => {
    const condition = tf.tensor3d(
        [
          [[true, false, false], [false, true, true]],
          [[false, false, false], [true, true, false]]
        ],
        [2, 2, 3], 'bool');
    const res = await tf.whereAsync(condition);
    expect(res.dtype).toBe('int32');
    expect(res.shape).toEqual([5, 3]);
    expectArraysClose(
        await res.data(), [0, 0, 0, 0, 1, 1, 0, 1, 2, 1, 1, 0, 1, 1, 1]);
  });

  it('accepts a tensor-like object', async () => {
    const condition = [true, false, true];
    const res = await tf.whereAsync(condition);
    expect(res.dtype).toBe('int32');
    expect(res.shape).toEqual([2, 1]);
    expectArraysClose(await res.data(), [0, 2]);
  });

  it('throws error if condition is not of type bool', async () => {
    const condition = tf.tensor1d([1, 0, 1]);
    // expect(...).toThrowError() does not support async functions.
    // See https://github.com/jasmine/jasmine/issues/1410
    try {
      await tf.whereAsync(condition);
      throw new Error('The line above should have thrown an error');
    } catch (ex) {
      expect(ex.message)
          .toMatch(/Argument 'condition' passed to 'whereAsync' must be bool/);
    }
  });

  it('returns tensor with 0 in shape when no values are true', async () => {
    const condition = [[[false]], [[false]], [[false]]];
    const res = await tf.whereAsync(condition);
    expect(res.dtype).toBe('int32');
    expect(res.shape).toEqual([0, 3]);
    expectArraysClose(await res.data(), []);
  });
});
