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
import {expectArraysClose, expectArraysEqual} from '../test_util';

describeWithFlags('prod', ALL_ENVS, () => {
  it('basic', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const result = tf.prod(a);
    expectArraysClose(await result.data(), 0);
  });

  it('propagates NaNs', async () => {
    const a = tf.tensor2d([1, 2, 3, NaN, 0, 1], [3, 2]);
    expectArraysEqual(await tf.prod(a).data(), NaN);
  });

  it('prod over dtype int32', async () => {
    const a = tf.tensor1d([1, 5, 7, 3], 'int32');
    const prod = tf.prod(a);
    expectArraysEqual(await prod.data(), 105);
  });

  it('prod over dtype bool', async () => {
    const a = tf.tensor1d([true, false, false, true, true], 'bool');
    const prod = tf.prod(a);
    expectArraysEqual(await prod.data(), 0);
  });

  it('prods all values in 2D array with keep dim', async () => {
    const a = tf.tensor2d([1, 2, 3, 1, 0, 1], [3, 2]);
    const res = tf.prod(a, null, true /* keepDims */);

    expect(res.shape).toEqual([1, 1]);
    expectArraysClose(await res.data(), 0);
  });

  it('prods across axis=0 in 2D array', async () => {
    const a = tf.tensor2d([1, 2, 3, 1, 0, 1], [3, 2]);
    const res = tf.prod(a, [0]);

    expect(res.shape).toEqual([2]);
    expectArraysClose(await res.data(), [0, 2]);
  });

  it('prods across axis=0 in 2D array, keepDims', async () => {
    const a = tf.tensor2d([1, 2, 3, 1, 0, 1], [3, 2]);
    const res = tf.prod(a, [0], true /* keepDims */);

    expect(res.shape).toEqual([1, 2]);
    expectArraysClose(await res.data(), [0, 2]);
  });

  it('prods across axis=1 in 2D array', async () => {
    const a = tf.tensor2d([1, 2, 3, 1, 1, 1], [3, 2]);
    const res = tf.prod(a, [1]);

    expect(res.shape).toEqual([3]);
    expectArraysClose(await res.data(), [2, 3, 1]);
  });

  it('2D, axis=1 provided as number', async () => {
    const a = tf.tensor2d([1, 2, 3, 1, 1, 1], [2, 3]);
    const res = tf.prod(a, 1);

    expect(res.shape).toEqual([2]);
    expectArraysClose(await res.data(), [6, 1]);
  });

  it('2D, axis = -1 provided as number', async () => {
    const a = tf.tensor2d([1, 2, 3, 1, 1, 1], [2, 3]);
    const res = tf.prod(a, -1);

    expect(res.shape).toEqual([2]);
    expectArraysClose(await res.data(), [6, 1]);
  });

  it('prods across axis=0,1 in 2D array', async () => {
    const a = tf.tensor2d([1, 2, 3, 1, 1, 1], [3, 2]);
    const res = tf.prod(a, [0, 1]);

    expect(res.shape).toEqual([]);
    expectArraysClose(await res.data(), [6]);
  });

  it('2D, axis=[-1,-2] in 2D array', async () => {
    const a = tf.tensor2d([1, 2, 3, 1, 1, 1], [3, 2]);
    const res = tf.prod(a, [-1, -2]);

    expect(res.shape).toEqual([]);
    expectArraysClose(await res.data(), [6]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.prod({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'prod' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const result = tf.prod([[1, 2], [3, 1], [1, 1]]);
    expectArraysClose(await result.data(), 6);
  });

  it('throws error for string tensor', () => {
    expect(() => tf.prod(['a']))
        .toThrowError(/Argument 'x' passed to 'prod' must be numeric tensor/);
  });
});
