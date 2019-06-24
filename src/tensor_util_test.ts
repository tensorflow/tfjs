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
import {ALL_ENVS, describeWithFlags} from './jasmine_util';
import {Tensor} from './tensor';
import {getTensorsInContainer, isTensorInList} from './tensor_util';
import {convertToTensor} from './tensor_util_env';
import {expectArraysClose, expectArraysEqual} from './test_util';

describeWithFlags('tensor_util.isTensorInList', ALL_ENVS, () => {
  it('not in list', () => {
    const a = tf.scalar(1);
    const list: Tensor[] = [tf.scalar(1), tf.tensor1d([1, 2, 3])];

    expect(isTensorInList(a, list)).toBe(false);
  });

  it('in list', () => {
    const a = tf.scalar(1);
    const list: Tensor[] = [tf.scalar(2), tf.tensor1d([1, 2, 3]), a];

    expect(isTensorInList(a, list)).toBe(true);
  });
});

describeWithFlags('getTensorsInContainer', ALL_ENVS, () => {
  it('null input returns empty tensor', () => {
    const results = getTensorsInContainer(null);

    expect(results).toEqual([]);
  });

  it('tensor input returns one element tensor', () => {
    const x = tf.scalar(1);
    const results = getTensorsInContainer(x);

    expect(results).toEqual([x]);
  });

  it('name tensor map returns flattened tensor', () => {
    const x1 = tf.scalar(1);
    const x2 = tf.scalar(3);
    const x3 = tf.scalar(4);
    const results = getTensorsInContainer({x1, x2, x3});

    expect(results).toEqual([x1, x2, x3]);
  });

  it('can extract from arbitrary depth', () => {
    const container = [
      {x: tf.scalar(1), y: tf.scalar(2)}, [[[tf.scalar(3)]], {z: tf.scalar(4)}]
    ];
    const results = getTensorsInContainer(container);
    expect(results.length).toBe(4);
  });

  it('works with loops in container', () => {
    const container = [tf.scalar(1), tf.scalar(2), [tf.scalar(3)]];
    const innerContainer = [container];
    // tslint:disable-next-line:no-any
    container.push(innerContainer as any);
    const results = getTensorsInContainer(container);
    expect(results.length).toBe(3);
  });
});

describeWithFlags('convertToTensor', ALL_ENVS, () => {
  it('primitive integer, NaN converts to zero, no error thrown', async () => {
    const a = () => convertToTensor(NaN, 'a', 'test', 'int32');
    expect(a).not.toThrowError();

    const b = convertToTensor(NaN, 'b', 'test', 'int32');
    expect(b.rank).toBe(0);
    expect(b.dtype).toBe('int32');
    expectArraysClose(await b.data(), 0);
  });

  it('primitive number', async () => {
    const a = convertToTensor(3, 'a', 'test');
    expect(a.rank).toBe(0);
    expect(a.dtype).toBe('float32');
    expectArraysClose(await a.data(), 3);
  });

  it('primitive integer, NaN converts to zero', async () => {
    const a = convertToTensor(NaN, 'a', 'test', 'int32');
    expect(a.rank).toBe(0);
    expect(a.dtype).toBe('int32');
    expectArraysClose(await a.data(), 0);
  });

  it('primitive boolean, parsed as bool tensor', async () => {
    const a = convertToTensor(true, 'a', 'test');
    expect(a.rank).toBe(0);
    expect(a.dtype).toBe('bool');
    expectArraysClose(await a.data(), 1);
  });

  it('primitive boolean, forced to be parsed as bool tensor', async () => {
    const a = convertToTensor(true, 'a', 'test', 'bool');
    expect(a.rank).toBe(0);
    expect(a.dtype).toBe('bool');
    expectArraysEqual(await a.data(), 1);
  });

  it('array1d', async () => {
    const a = convertToTensor([1, 2, 3], 'a', 'test');
    expect(a.rank).toBe(1);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    expectArraysClose(await a.data(), [1, 2, 3]);
  });

  it('array2d', async () => {
    const a = convertToTensor([[1], [2], [3]], 'a', 'test');
    expect(a.rank).toBe(2);
    expect(a.shape).toEqual([3, 1]);
    expect(a.dtype).toBe('float32');
    expectArraysClose(await a.data(), [1, 2, 3]);
  });

  it('array3d', async () => {
    const a = convertToTensor([[[1], [2]], [[3], [4]]], 'a', 'test');
    expect(a.rank).toBe(3);
    expect(a.shape).toEqual([2, 2, 1]);
    expect(a.dtype).toBe('float32');
    expectArraysClose(await a.data(), [1, 2, 3, 4]);
  });

  it('array4d', async () => {
    const a = convertToTensor([[[[1]], [[2]]], [[[3]], [[4]]]], 'a', 'test');
    expect(a.rank).toBe(4);
    expect(a.shape).toEqual([2, 2, 1, 1]);
    expect(a.dtype).toBe('float32');
    expectArraysClose(await a.data(), [1, 2, 3, 4]);
  });

  it('passing a tensor returns the tensor itself', () => {
    const s = tf.scalar(3);
    const res = convertToTensor(s, 'a', 'test');
    expect(res === s).toBe(true);
  });

  it('passing a tensor with wrong type errors', () => {
    const s = tf.scalar(3);
    expect(() => convertToTensor(s, 'p', 'f', 'bool'))
        .toThrowError(
            /Argument 'p' passed to 'f' must be bool tensor, but got float32/);
  });

  it('fails when passed a string and force numeric is true', () => {
    const expectedDtype = 'numeric';
    expect(() => convertToTensor('hello', 'p', 'test', expectedDtype))
        .toThrowError();
  });

  it('force numeric is true by default', () => {
    // Should fail to parse a string tensor since force numeric is true.
    expect(() => convertToTensor('hello', 'p', 'test')).toThrowError();
  });

  it('primitive string, do not force numeric', () => {
    const t = convertToTensor('hello', 'p', 'test', null /* Allow any dtype */);
    expect(t.dtype).toBe('string');
    expect(t.shape).toEqual([]);
  });

  it('string[], do not force numeric', () => {
    const t = convertToTensor(
        ['a', 'b', 'c'], 'p', 'test', null /* Allow any dtype */);
    expect(t.dtype).toBe('string');
    expect(t.shape).toEqual([3]);
  });

  it('string, explicitly parse as bool', () => {
    expect(() => convertToTensor('a', 'argName', 'func', 'bool'))
        .toThrowError(
            'Argument \'argName\' passed to \'func\' must be bool tensor' +
            ', but got string tensor');
  });

  it('fails to convert a dict to tensor', () => {
    expect(() => convertToTensor({} as number, 'a', 'test'))
        .toThrowError(
            'Argument \'a\' passed to \'test\' must be a Tensor ' +
            'or TensorLike, but got \'Object\'');
  });

  it('fails to convert a string to tensor', () => {
    expect(() => convertToTensor('asdf', 'a', 'test'))
        .toThrowError(
            'Argument \'a\' passed to \'test\' must be numeric tensor, ' +
            'but got string tensor');
  });
});

describeWithFlags('convertToTensor debug mode', ALL_ENVS, () => {
  beforeAll(() => {
    tf.ENV.set('DEBUG', true);
  });

  it('fails to convert a non-valid shape array to tensor', () => {
    const a = [[1, 2], [3], [4, 5, 6]];  // 2nd element has only 1 entry.
    expect(() => convertToTensor(a, 'a', 'test'))
        .toThrowError(
            'Element arr[1] should have 2 elements, but has 1 elements');
  });
});
