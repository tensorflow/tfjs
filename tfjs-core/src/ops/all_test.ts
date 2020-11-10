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

describeWithFlags('all', ALL_ENVS, () => {
  it('Tensor1D', async () => {
    let a = tf.tensor1d([0, 0, 0], 'bool');
    expectArraysClose(await tf.all(a).data(), 0);

    a = tf.tensor1d([1, 0, 1], 'bool');
    expectArraysClose(await tf.all(a).data(), 0);

    a = tf.tensor1d([1, 1, 1], 'bool');
    expectArraysClose(await tf.all(a).data(), 1);
  });

  it('ignores NaNs', async () => {
    const a = tf.tensor1d([1, NaN, 1], 'bool');
    expectArraysEqual(await tf.all(a).data(), 1);
  });

  it('2D', async () => {
    const a = tf.tensor2d([1, 1, 0, 0], [2, 2], 'bool');
    expectArraysClose(await tf.all(a).data(), 0);
  });

  it('2D axis=[0,1]', async () => {
    const a = tf.tensor2d([1, 1, 0, 0, 1, 0], [2, 3], 'bool');
    expectArraysClose(await tf.all(a, [0, 1]).data(), 0);
  });

  it('2D, axis=0', async () => {
    const a = tf.tensor2d([1, 1, 0, 0], [2, 2], 'bool');
    let r = tf.all(a, 0);

    expect(r.shape).toEqual([2]);
    expectArraysClose(await r.data(), [0, 0]);

    r = tf.all(a, 1);

    expect(r.shape).toEqual([2]);
    expectArraysClose(await r.data(), [1, 0]);
  });

  it('2D, axis=0, keepDims', async () => {
    const a = tf.tensor2d([1, 1, 0, 0, 1, 0], [2, 3], 'bool');
    const r = a.all(0, true /* keepDims */);

    expect(r.shape).toEqual([1, 3]);
    expectArraysClose(await r.data(), [0, 1, 0]);
  });

  it('2D, axis=1 provided as a number', async () => {
    const a = tf.tensor2d([1, 1, 0, 0, 1, 0], [2, 3], 'bool');
    const r = tf.all(a, 1);
    expectArraysClose(await r.data(), [0, 0]);
  });

  it('2D, axis = -1 provided as a number', async () => {
    const a = tf.tensor2d([1, 1, 0, 0, 1, 0], [2, 3], 'bool');
    const r = tf.all(a, -1);
    expectArraysClose(await r.data(), [0, 0]);
  });

  it('2D, axis=[1]', async () => {
    const a = tf.tensor2d([1, 1, 0, 0, 1, 0], [2, 3], 'bool');
    const r = tf.all(a, [1]);
    expectArraysClose(await r.data(), [0, 0]);
  });

  it('throws when dtype is not boolean', () => {
    const a = tf.tensor2d([1, 1, 0, 0], [2, 2]);
    expect(() => tf.all(a))
        .toThrowError(
            /Argument 'x' passed to 'all' must be bool tensor, but got float/);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.all({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'all' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const a = [0, 0, 0];
    expectArraysClose(await tf.all(a).data(), 0);
  });

  it('throws error for string tensor', () => {
    expect(() => tf.all(['a']))
        .toThrowError(
            /Argument 'x' passed to 'all' must be bool tensor, but got string/);
  });
});
