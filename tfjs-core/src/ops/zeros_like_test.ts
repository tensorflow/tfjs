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

describeWithFlags('zerosLike', ALL_ENVS, () => {
  it('1D default dtype', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    expectArraysClose(await b.data(), [0, 0, 0]);
  });

  it('chainable 1D default dtype', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    const b = a.zerosLike();
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    expectArraysClose(await b.data(), [0, 0, 0]);
  });

  it('1D float32 dtype', async () => {
    const a = tf.tensor1d([1, 2, 3], 'float32');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    expectArraysClose(await b.data(), [0, 0, 0]);
  });

  it('1D int32 dtype', async () => {
    const a = tf.tensor1d([1, 2, 3], 'int32');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([3]);
    expectArraysEqual(await b.data(), [0, 0, 0]);
  });

  it('1D bool dtype', async () => {
    const a = tf.tensor1d([1, 2, 3], 'bool');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([3]);
    expectArraysEqual(await b.data(), [0, 0, 0]);
  });

  it('2D default dtype', async () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    expectArraysClose(await b.data(), [0, 0, 0, 0]);
  });

  it('2D float32 dtype', async () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2], 'float32');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    expectArraysClose(await b.data(), [0, 0, 0, 0]);
  });

  it('2D int32 dtype', async () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2], 'int32');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2]);
    expectArraysEqual(await b.data(), [0, 0, 0, 0]);
  });

  it('2D bool dtype', async () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2], 'bool');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2]);
    expectArraysEqual(await b.data(), [0, 0, 0, 0]);
  });

  it('3D default dtype', async () => {
    const a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysClose(await b.data(), [0, 0, 0, 0]);
  });

  it('3D float32 dtype', async () => {
    const a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1], 'float32');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysClose(await b.data(), [0, 0, 0, 0]);
  });

  it('3D int32 dtype', async () => {
    const a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1], 'int32');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysEqual(await b.data(), [0, 0, 0, 0]);
  });

  it('3D bool dtype', async () => {
    const a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1], 'bool');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysEqual(await b.data(), [0, 0, 0, 0]);
  });

  it('4D default dtype', async () => {
    const a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1]);
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysClose(await b.data(), [0, 0, 0, 0]);
  });

  it('4D float32 dtype', async () => {
    const a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'float32');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysClose(await b.data(), [0, 0, 0, 0]);
  });

  it('4D int32 dtype', async () => {
    const a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'int32');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysEqual(await b.data(), [0, 0, 0, 0]);
  });

  it('4D bool dtype', async () => {
    const a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'bool');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysEqual(await b.data(), [0, 0, 0, 0]);
  });

  it('4D default dtype', async () => {
    const a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1]);
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysClose(await b.data(), [0, 0, 0, 0]);
  });

  it('5D float32 dtype', async () => {
    const a = tf.tensor5d([1, 2, 3, 4], [1, 2, 2, 1, 1], 'float32');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([1, 2, 2, 1, 1]);
    expectArraysClose(await b.data(), [0, 0, 0, 0]);
  });

  it('5D int32 dtype', async () => {
    const a = tf.tensor5d([1, 2, 3, 4], [1, 2, 2, 1, 1], 'int32');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([1, 2, 2, 1, 1]);
    expectArraysEqual(await b.data(), [0, 0, 0, 0]);
  });

  it('5D bool dtype', async () => {
    const a = tf.tensor5d([1, 2, 3, 4], [1, 2, 2, 1, 1], 'bool');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([1, 2, 2, 1, 1]);
    expectArraysEqual(await b.data(), [0, 0, 0, 0]);
  });

  it('5D default dtype', async () => {
    const a = tf.tensor5d([1, 2, 3, 4], [1, 2, 2, 1, 1]);
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([1, 2, 2, 1, 1]);
    expectArraysClose(await b.data(), [0, 0, 0, 0]);
  });

  it('6D float32 dtype', async () => {
    const a = tf.tensor6d([1, 2, 3, 4], [1, 2, 2, 1, 1, 1], 'float32');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([1, 2, 2, 1, 1, 1]);
    expectArraysClose(await b.data(), [0, 0, 0, 0]);
  });

  it('6D int32 dtype', async () => {
    const a = tf.tensor6d([1, 2, 3, 4], [1, 2, 2, 1, 1, 1], 'int32');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual(a.shape);
    expectArraysEqual(await b.data(), [0, 0, 0, 0]);
  });

  it('6D bool dtype', async () => {
    const a = tf.tensor6d([1, 2, 3, 4], [1, 2, 2, 1, 1, 1], 'bool');
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual(a.shape);
    expectArraysEqual(await b.data(), [0, 0, 0, 0]);
  });

  it('6D default dtype', async () => {
    const a = tf.tensor6d([1, 2, 3, 4], [1, 2, 2, 1, 1, 1]);
    const b = tf.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual(a.shape);
    expectArraysClose(await b.data(), [0, 0, 0, 0]);
  });

  it('zerosLike gradient', async () => {
    const x = tf.tensor2d([[0, 1, 2], [4, 5, 6]]);
    const gradients = tf.grad(x => tf.zerosLike(x))(x);
    expect(gradients.shape).toEqual([2, 3]);
    expectArraysEqual(await gradients.data(), [0, 0, 0, 0, 0, 0]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.zerosLike({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'zerosLike' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const res = tf.zerosLike([[1, 2], [3, 4]]);
    expect(res.shape).toEqual([2, 2]);
    expectArraysEqual(await res.data(), [0, 0, 0, 0]);
  });
});
