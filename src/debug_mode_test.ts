/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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
import {describeWithFlags} from './jasmine_util';
import {convertToTensor} from './tensor_util_env';
import {ALL_ENVS, expectArraysClose} from './test_util';

describeWithFlags('debug on', ALL_ENVS, () => {
  beforeAll(() => {
    tf.ENV.set('DEBUG', true);
  });

  afterAll(() => {
    tf.ENV.set('DEBUG', false);
  });

  it('debug mode does not error when no nans', () => {
    const a = tf.tensor1d([2, -1, 0, 3]);
    const res = tf.relu(a);
    expectArraysClose(res, [2, 0, 0, 3]);
  });

  it('debug mode errors when there are nans, float32', () => {
    const a = tf.tensor1d([2, NaN]);
    const f = () => tf.relu(a);
    expect(f).toThrowError();
  });

  it('debug mode errors when nans in tensor construction, int32', () => {
    const a = () => tf.tensor1d([2, NaN], 'int32');
    expect(a).toThrowError();
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

  it('A x B', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);

    const c = tf.matMul(a, b);

    expect(c.shape).toEqual([2, 2]);
    expectArraysClose(c, [0, 8, -3, 20]);
  });
});

describeWithFlags('debug off', ALL_ENVS, () => {
  beforeAll(() => {
    tf.ENV.set('DEBUG', false);
  });

  it('no errors where there are nans, and debug mode is disabled', () => {
    const a = tf.tensor1d([2, NaN]);
    const res = tf.relu(a);
    expectArraysClose(res, [2, NaN]);
  });
});
