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

import * as tf from './index';
import {ALL_ENVS, describeWithFlags, SYNC_BACKEND_ENVS} from './jasmine_util';
import {convertToTensor} from './tensor_util_env';
import {expectArraysClose} from './test_util';

describeWithFlags('debug on', SYNC_BACKEND_ENVS, () => {
  beforeAll(() => {
    // Silence debug warnings.
    spyOn(console, 'warn');
    tf.enableDebugMode();
  });

  it('debug mode does not error when no nans', async () => {
    const a = tf.tensor1d([2, -1, 0, 3]);
    const res = tf.relu(a);
    expectArraysClose(await res.data(), [2, 0, 0, 3]);
  });

  it('debug mode errors when nans in tensor construction, float32', () => {
    const a = () => tf.tensor1d([2, NaN], 'float32');
    expect(a).toThrowError();
  });

  it('debug mode errors when nans in tensor construction, int32', () => {
    const a = () => tf.tensor1d([2, NaN], 'int32');
    expect(a).toThrowError();
  });

  it('debug mode errors when Infinity in tensor construction', () => {
    const a = () => tf.tensor1d([2, Infinity], 'float32');
    expect(a).toThrowError();
  });

  it('debug mode errors when nans in tensor created from TypedArray', () => {
    const a = () => tf.tensor1d(new Float32Array([1, 2, NaN]), 'float32');
    expect(a).toThrowError();
  });

  it('debug mode errors when infinities in op output', async () => {
    const a = tf.tensor1d([1, 2, 3, 4]);
    const b = tf.tensor1d([2, -1, 0, 3]);

    const c = async () => {
      const result = a.div(b);
      // Must await result so we know exception would have happened by the
      // time we call `expect`.
      await result.data();
    };

    await c();

    expect(console.warn).toHaveBeenCalled();
  });

  it('debug mode errors when nans in op output', async () => {
    const a = tf.tensor1d([-1, 2]);
    const b = tf.tensor1d([0.5, 1]);

    const c = async () => {
      const result = a.pow(b);
      await result.data();
    };

    await c();

    expect(console.warn).toHaveBeenCalled();
  });

  it('debug mode errors when nans in oneHot op (tensorlike), int32', () => {
    const f = () => tf.oneHot([2, NaN], 3);
    expect(f).toThrowError();
  });

  it('debug mode errors when nan in convertToTensor, int32', () => {
    const a = () => convertToTensor(NaN, 'a', 'test', 'int32');
    expect(a).toThrowError();
  });

  it('debug mode errors when nan in convertToTensor array input, int32', () => {
    const a = () => convertToTensor([NaN], 'a', 'test', 'int32');
    expect(a).toThrowError();
  });

  // tslint:disable-next-line: ban
  xit('A x B', async () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);

    const c = tf.matMul(a, b);

    expect(c.shape).toEqual([2, 2]);
    expectArraysClose(await c.data(), [0, 8, -3, 20]);
  });
});

describeWithFlags('debug off', ALL_ENVS, () => {
  beforeAll(() => {
    tf.env().set('DEBUG', false);
  });

  it('no errors where there are nans, and debug mode is disabled', async () => {
    const a = tf.tensor1d([2, NaN]);
    const res = tf.relu(a);
    expectArraysClose(await res.data(), [2, NaN]);
  });
});
