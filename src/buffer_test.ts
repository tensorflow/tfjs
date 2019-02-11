/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import {ALL_ENVS, expectArraysClose, expectArraysEqual} from './test_util';

describeWithFlags('tf.buffer', ALL_ENVS, () => {
  it('float32', () => {
    const buff = tf.buffer([1, 2, 3], 'float32');
    buff.set(1.3, 0, 0, 0);
    buff.set(2.9, 0, 1, 0);
    expect(buff.get(0, 0, 0)).toBeCloseTo(1.3);
    expect(buff.get(0, 0, 1)).toBeCloseTo(0);
    expect(buff.get(0, 0, 2)).toBeCloseTo(0);
    expect(buff.get(0, 1, 0)).toBeCloseTo(2.9);
    expect(buff.get(0, 1, 1)).toBeCloseTo(0);
    expect(buff.get(0, 1, 2)).toBeCloseTo(0);
    expectArraysClose(buff.toTensor(), [1.3, 0, 0, 2.9, 0, 0]);
    expectArraysClose(buff.values, new Float32Array([1.3, 0, 0, 2.9, 0, 0]));
  });

  it('int32', () => {
    const buff = tf.buffer([2, 3], 'int32');
    buff.set(1.3, 0, 0);
    buff.set(2.1, 1, 1);
    expect(buff.get(0, 0)).toEqual(1);
    expect(buff.get(0, 1)).toEqual(0);
    expect(buff.get(0, 2)).toEqual(0);
    expect(buff.get(1, 0)).toEqual(0);
    expect(buff.get(1, 1)).toEqual(2);
    expect(buff.get(1, 2)).toEqual(0);
    expectArraysClose(buff.toTensor(), [1, 0, 0, 0, 2, 0]);
    expectArraysClose(buff.values, new Int32Array([1, 0, 0, 0, 2, 0]));
  });

  it('bool', () => {
    const buff = tf.buffer([4], 'bool');
    buff.set(true, 1);
    buff.set(true, 2);
    expect(buff.get(0)).toBeFalsy();
    expect(buff.get(1)).toBeTruthy();
    expect(buff.get(2)).toBeTruthy();
    expect(buff.get(3)).toBeFalsy();
    expectArraysClose(buff.toTensor(), [0, 1, 1, 0]);
    expectArraysClose(buff.values, new Uint8Array([0, 1, 1, 0]));
  });

  it('string', () => {
    const buff = tf.buffer([2, 2], 'string');
    buff.set('first', 0, 0);
    buff.set('third', 1, 0);
    expect(buff.get(0, 0)).toEqual('first');
    expect(buff.get(0, 1)).toBeFalsy();
    expect(buff.get(1, 0)).toEqual('third');
    expect(buff.get(1, 1)).toBeFalsy();
    expectArraysEqual(buff.toTensor(), ['first', null, 'third', null]);
  });

  it('throws when passed non-integer shape', () => {
    const msg = 'Tensor must have a shape comprised of positive ' +
        'integers but got shape [2,2.2].';
    expect(() => tf.buffer([2, 2.2])).toThrowError(msg);
  });

  it('throws when passed negative shape', () => {
    const msg = 'Tensor must have a shape comprised of positive ' +
        'integers but got shape [2,-2].';
    expect(() => tf.buffer([2, -2])).toThrowError(msg);
  });
});
