/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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

describeWithFlags('clone', ALL_ENVS, () => {
  it('returns a tensor with the same shape and value', async () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3]);
    const aPrime = tf.clone(a);
    expect(aPrime.shape).toEqual(a.shape);
    expectArraysClose(await aPrime.data(), await a.data());
    expect(aPrime.shape).toEqual(a.shape);
  });

  it('accepts a tensor-like object', async () => {
    const res = tf.clone([[1, 2, 3], [4, 5, 6]]);
    expect(res.dtype).toBe('float32');
    expect(res.shape).toEqual([2, 3]);
    expectArraysClose(await res.data(), [1, 2, 3, 4, 5, 6]);
  });
});

describeWithFlags('clone', ALL_ENVS, () => {
  it('1D default dtype', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    const b = tf.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    expectArraysClose(await b.data(), [1, 2, 3]);
  });

  it('1D float32 dtype', async () => {
    const a = tf.tensor1d([1, 2, 3], 'float32');
    const b = tf.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    expectArraysClose(await b.data(), [1, 2, 3]);
  });

  it('1D int32 dtype', async () => {
    const a = tf.tensor1d([1, 2, 3], 'int32');
    const b = tf.clone(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([3]);
    expectArraysEqual(await b.data(), [1, 2, 3]);
  });

  it('1D bool dtype', async () => {
    const a = tf.tensor1d([1, 1, 0], 'bool');
    const b = tf.clone(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([3]);
    expectArraysEqual(await b.data(), [1, 1, 0]);
  });

  it('1D complex64 dtype', async () => {
    const a = tf.complex([1], [1]);
    const b = tf.clone(a);
    expect(b.dtype).toBe('complex64');
    expect(b.shape).toEqual([1]);
    expectArraysEqual(await b.data(), [1, 1]);
  });

  it('1D string dtype', async () => {
    const a = tf.tensor1d(['a', 'b', 'c'], 'string');
    const b = tf.clone(a);
    expect(b.dtype).toBe('string');
    expect(b.shape).toEqual([3]);
    expectArraysEqual(await b.data(), ['a', 'b', 'c']);
  });

  it('2D default dtype', async () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const b = tf.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    expectArraysClose(await b.data(), [1, 2, 3, 4]);
  });

  it('2D float32 dtype', async () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2], 'float32');
    const b = tf.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    expectArraysClose(await b.data(), [1, 2, 3, 4]);
  });

  it('2D int32 dtype', async () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2], 'int32');
    const b = tf.clone(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2]);
    expectArraysEqual(await b.data(), [1, 2, 3, 4]);
  });

  it('2D bool dtype', async () => {
    const a = tf.tensor2d([1, 1, 1, 0], [2, 2], 'bool');
    const b = tf.clone(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2]);
    expectArraysEqual(await b.data(), [1, 1, 1, 0]);
  });

  it('2D complex64 dtype', async () => {
    const a = tf.complex([[1, 3], [5, 7]], [[2, 4], [6, 8]]);
    const b = tf.clone(a);
    expect(b.dtype).toBe('complex64');
    expect(b.shape).toEqual([2, 2]);
    expectArraysEqual(await b.data(), [1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it('2D string dtype', async () => {
    const a = tf.tensor2d(['a', 'b', 'c', 'd'], [2, 2], 'string');
    const b = tf.clone(a);
    expect(b.dtype).toBe('string');
    expect(b.shape).toEqual([2, 2]);
    expectArraysEqual(await b.data(), ['a', 'b', 'c', 'd']);
  });

  it('3D default dtype', async () => {
    const a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
    const b = tf.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysClose(await b.data(), [1, 2, 3, 4]);
  });

  it('3D float32 dtype', async () => {
    const a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1], 'float32');
    const b = tf.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysClose(await b.data(), [1, 2, 3, 4]);
  });

  it('3D int32 dtype', async () => {
    const a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1], 'int32');
    const b = tf.clone(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysEqual(await b.data(), [1, 2, 3, 4]);
  });

  it('3D bool dtype', async () => {
    const a = tf.tensor3d([1, 1, 1, 0], [2, 2, 1], 'bool');
    const b = tf.clone(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysEqual(await b.data(), [1, 1, 1, 0]);
  });

  it('3D complex64 dtype', async () => {
    const a = tf.complex([[[1], [3]], [[5], [7]]], [[[2], [4]], [[6], [8]]]);
    const b = tf.clone(a);
    expect(b.dtype).toBe('complex64');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysEqual(await b.data(), [1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it('3D string dtype', async () => {
    const a = tf.tensor3d(['a', 'b', 'c', 'd'], [2, 2, 1], 'string');
    const b = tf.clone(a);
    expect(b.dtype).toBe('string');
    expect(b.shape).toEqual([2, 2, 1]);
    expectArraysEqual(await b.data(), ['a', 'b', 'c', 'd']);
  });

  it('4D default dtype', async () => {
    const a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1]);
    const b = tf.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysClose(await b.data(), [1, 2, 3, 4]);
  });

  it('4D float32 dtype', async () => {
    const a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'float32');
    const b = tf.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysClose(await b.data(), [1, 2, 3, 4]);
  });

  it('4D int32 dtype', async () => {
    const a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'int32');
    const b = tf.clone(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysEqual(await b.data(), [1, 2, 3, 4]);
  });

  it('4D bool dtype', async () => {
    const a = tf.tensor4d([1, 1, 1, 0], [2, 2, 1, 1], 'bool');
    const b = tf.clone(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysEqual(await b.data(), [1, 1, 1, 0]);
  });

  it('4D complex64 dtype', async () => {
    const a = tf.complex(
        [[[[1]], [[3]]], [[[5]], [[7]]]], [[[[2]], [[4]]], [[[6]], [[8]]]]);
    const b = tf.clone(a);
    expect(b.dtype).toBe('complex64');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysEqual(await b.data(), [1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it('4D string dtype', async () => {
    const a = tf.tensor4d(['a', 'b', 'c', 'd'], [2, 2, 1, 1], 'string');
    const b = tf.clone(a);
    expect(b.dtype).toBe('string');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expectArraysEqual(await b.data(), ['a', 'b', 'c', 'd']);
  });

  it('gradient: 1D', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    const dy = tf.tensor1d([4, 5, 6]);
    const da = tf.grad(x => tf.clone(x))(a, dy);

    expect(da.dtype).toBe('float32');
    expect(da.shape).toEqual([3]);
    expectArraysClose(await da.data(), [4, 5, 6]);
  });

  it('gradient with clones', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    const dy = tf.tensor1d([4, 5, 6]);
    const da = tf.grad(x => tf.clone(x.clone()).clone())(a, dy);

    expect(da.dtype).toBe('float32');
    expect(da.shape).toEqual([3]);
    expectArraysClose(await da.data(), [4, 5, 6]);
  });

  it('gradient: 1D string throws error with string dy', () => {
    const a = tf.tensor1d(['a', 'b', 'c'], 'string');
    const dy = tf.tensor1d(['d', 'e', 'f']);
    expect(() => tf.grad(x => tf.clone(x))(a, dy)).toThrowError();
  });

  it('gradient: 1D string throws error with bool dy', () => {
    const a = tf.tensor1d(['a', 'b', 'c'], 'string');
    const dy = tf.tensor1d([false, true, false], 'bool');
    expect(() => tf.grad(x => tf.clone(x))(a, dy)).toThrowError();
  });

  it('gradient: 1D string throws error with int32 dy', () => {
    const a = tf.tensor1d(['a', 'b', 'c'], 'string');
    const dy = tf.tensor1d([4, 5, 6], 'int32');
    expect(() => tf.grad(x => tf.clone(x))(a, dy)).toThrowError();
  });

  it('gradient: 1D string works with float32 dy', async () => {
    const a = tf.tensor1d(['a', 'b', 'c'], 'string');
    const dy = tf.tensor1d([4, 5, 6]);
    const da = tf.grad(x => tf.clone(x))(a, dy);

    expect(da.dtype).toBe('float32');
    expect(da.shape).toEqual([3]);
    expectArraysClose(await da.data(), [4, 5, 6]);
  });

  it('gradient: 2D int32', async () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2], 'int32');
    const dy = tf.tensor2d([5, 6, 7, 8], [2, 2], 'float32');
    const da = tf.grad(x => tf.clone(x))(a, dy);

    expect(da.dtype).toBe('float32');
    expect(da.shape).toEqual([2, 2]);
    expectArraysEqual(await da.data(), [5, 6, 7, 8]);
  });

  it('gradient: 4D bool', async () => {
    const a = tf.tensor4d([1, 1, 1, 0], [2, 2, 1, 1], 'bool');
    const dy = tf.tensor4d([5, 6, 7, 8], [2, 2, 1, 1], 'float32');
    const da = tf.grad(x => tf.clone(x))(a, dy);

    expect(da.dtype).toBe('float32');
    expect(da.shape).toEqual([2, 2, 1, 1]);
    expectArraysEqual(await da.data(), [5, 6, 7, 8]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.clone({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'clone' must be a Tensor/);
  });
});
