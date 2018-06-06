/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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
import {Scalar, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D} from './tensor';
// tslint:disable-next-line:max-line-length
import {ALL_ENVS, expectArraysClose, expectArraysEqual, expectNumbersClose} from './test_util';
import {DType, Rank} from './types';

describeWithFlags('tensor', ALL_ENVS, () => {
  it('Tensors of arbitrary size', () => {
    // [1, 2, 3]
    let t: Tensor = tf.tensor1d([1, 2, 3]);
    expect(t.rank).toBe(1);
    expect(t.size).toBe(3);
    expectArraysClose(t, [1, 2, 3]);
    // Out of bounds indexing.
    expect(t.get(4)).toBeUndefined();

    // [[1, 2, 3]]
    t = tf.tensor2d([1, 2, 3], [1, 3]);
    expect(t.rank).toBe(2);
    expect(t.size).toBe(3);
    expectArraysClose(t, [1, 2, 3]);
    // Out of bounds indexing.
    expect(t.get(0, 4)).toBeUndefined();

    // [[1, 2, 3],
    //  [4, 5, 6]]
    t = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    expect(t.rank).toBe(2);
    expect(t.size).toBe(6);

    expectArraysClose(t, [1, 2, 3, 4, 5, 6]);

    // Out of bounds indexing.
    expect(t.get(5, 3)).toBeUndefined();

    // Shape mismatch with the values.
    expect(() => tf.tensor2d([1], [1, 2])).toThrowError();
  });

  it('Tensors of explicit size', () => {
    const t = tf.tensor1d([5, 3, 2]);
    expect(t.rank).toBe(1);
    expect(t.shape).toEqual([3]);
    expectNumbersClose(t.get(1), 3);

    // tslint:disable-next-line:no-any
    expect(() => tf.tensor3d([1, 2], [1, 2, 3, 5] as any)).toThrowError();

    const t4 = tf.tensor4d([1, 2, 3, 4], [1, 2, 1, 2]);
    expectNumbersClose(t4.get(0, 0, 0, 0), 1);
    expectNumbersClose(t4.get(0, 0, 0, 1), 2);
    expectNumbersClose(t4.get(0, 1, 0, 0), 3);
    expectNumbersClose(t4.get(0, 1, 0, 1), 4);

    // Tensor of ones.
    const x = tf.ones<Rank.R3>([3, 4, 2]);
    expect(x.rank).toBe(3);
    expect(x.size).toBe(24);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 4; j++) {
        for (let k = 0; k < 2; k++) {
          expectNumbersClose(x.get(i, j, k), 1);
        }
      }
    }

    // Tensor of zeros.
    const z = tf.zeros<Rank.R3>([3, 4, 2]);
    expect(z.rank).toBe(3);
    expect(z.size).toBe(24);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 4; j++) {
        for (let k = 0; k < 2; k++) {
          expectNumbersClose(z.get(i, j, k), 0);
        }
      }
    }

    // Reshaping tensors.
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    expectNumbersClose(a.get(1, 2), 6);
  });

  it('Tensor dataSync CPU --> GPU', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
    expectArraysClose(a.dataSync(), new Float32Array([1, 2, 3, 4, 5, 6]));
  });

  it('Tensor.data() CPU --> GPU', async () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
    expectArraysClose(await a.data(), new Float32Array([1, 2, 3, 4, 5, 6]));
  });

  it('Scalar basic methods', () => {
    const a = tf.scalar(5);
    expectNumbersClose(a.get(), 5);
    expectArraysClose(a, [5]);
    expect(a.rank).toBe(0);
    expect(a.size).toBe(1);
    expect(a.shape).toEqual([]);
  });

  it('indexToLoc Scalar', () => {
    const a = tf.scalar(0).buffer();
    expect(a.indexToLoc(0)).toEqual([]);

    const b = tf.zeros<Rank.R0>([]).buffer();
    expect(b.indexToLoc(0)).toEqual([]);
  });

  it('indexToLoc Tensor1D', () => {
    const a = tf.zeros([3]).buffer();
    expect(a.indexToLoc(0)).toEqual([0]);
    expect(a.indexToLoc(1)).toEqual([1]);
    expect(a.indexToLoc(2)).toEqual([2]);

    const b = tf.zeros<Rank.R1>([3]).buffer();
    expect(b.indexToLoc(0)).toEqual([0]);
    expect(b.indexToLoc(1)).toEqual([1]);
    expect(b.indexToLoc(2)).toEqual([2]);
  });

  it('indexToLoc Tensor2D', () => {
    const a = tf.zeros([3, 2]).buffer();
    expect(a.indexToLoc(0)).toEqual([0, 0]);
    expect(a.indexToLoc(1)).toEqual([0, 1]);
    expect(a.indexToLoc(2)).toEqual([1, 0]);
    expect(a.indexToLoc(3)).toEqual([1, 1]);
    expect(a.indexToLoc(4)).toEqual([2, 0]);
    expect(a.indexToLoc(5)).toEqual([2, 1]);

    const b = tf.zeros<Rank.R2>([3, 2]).buffer();
    expect(b.indexToLoc(0)).toEqual([0, 0]);
    expect(b.indexToLoc(1)).toEqual([0, 1]);
    expect(b.indexToLoc(2)).toEqual([1, 0]);
    expect(b.indexToLoc(3)).toEqual([1, 1]);
    expect(b.indexToLoc(4)).toEqual([2, 0]);
    expect(b.indexToLoc(5)).toEqual([2, 1]);
  });

  it('indexToLoc Tensor3D', () => {
    const a = tf.zeros([3, 2, 2]).buffer();
    expect(a.indexToLoc(0)).toEqual([0, 0, 0]);
    expect(a.indexToLoc(1)).toEqual([0, 0, 1]);
    expect(a.indexToLoc(2)).toEqual([0, 1, 0]);
    expect(a.indexToLoc(3)).toEqual([0, 1, 1]);
    expect(a.indexToLoc(4)).toEqual([1, 0, 0]);
    expect(a.indexToLoc(5)).toEqual([1, 0, 1]);
    expect(a.indexToLoc(11)).toEqual([2, 1, 1]);

    const b = tf.zeros<Rank.R3>([3, 2, 2]).buffer();
    expect(b.indexToLoc(0)).toEqual([0, 0, 0]);
    expect(b.indexToLoc(1)).toEqual([0, 0, 1]);
    expect(b.indexToLoc(2)).toEqual([0, 1, 0]);
    expect(b.indexToLoc(3)).toEqual([0, 1, 1]);
    expect(b.indexToLoc(4)).toEqual([1, 0, 0]);
    expect(b.indexToLoc(5)).toEqual([1, 0, 1]);
    expect(b.indexToLoc(11)).toEqual([2, 1, 1]);
  });

  it('indexToLoc Tensor 5D', () => {
    const values = new Float32Array([1, 2, 3, 4]);
    const a = Tensor.make([2, 1, 1, 1, 2], {values}).buffer();
    expect(a.indexToLoc(0)).toEqual([0, 0, 0, 0, 0]);
    expect(a.indexToLoc(1)).toEqual([0, 0, 0, 0, 1]);
    expect(a.indexToLoc(2)).toEqual([1, 0, 0, 0, 0]);
    expect(a.indexToLoc(3)).toEqual([1, 0, 0, 0, 1]);
  });

  it('locToIndex Scalar', () => {
    const a = tf.scalar(0).buffer();
    expect(a.locToIndex([])).toEqual(0);

    const b = tf.zeros<Rank.R0>([]).buffer();
    expect(b.locToIndex([])).toEqual(0);
  });

  it('locToIndex Tensor1D', () => {
    const a = tf.zeros<Rank.R1>([3]).buffer();
    expect(a.locToIndex([0])).toEqual(0);
    expect(a.locToIndex([1])).toEqual(1);
    expect(a.locToIndex([2])).toEqual(2);

    const b = tf.zeros<Rank.R1>([3]).buffer();
    expect(b.locToIndex([0])).toEqual(0);
    expect(b.locToIndex([1])).toEqual(1);
    expect(b.locToIndex([2])).toEqual(2);
  });

  it('locToIndex Tensor2D', () => {
    const a = tf.zeros<Rank.R2>([3, 2]).buffer();
    expect(a.locToIndex([0, 0])).toEqual(0);
    expect(a.locToIndex([0, 1])).toEqual(1);
    expect(a.locToIndex([1, 0])).toEqual(2);
    expect(a.locToIndex([1, 1])).toEqual(3);
    expect(a.locToIndex([2, 0])).toEqual(4);
    expect(a.locToIndex([2, 1])).toEqual(5);

    const b = tf.zeros<Rank.R2>([3, 2]).buffer();
    expect(b.locToIndex([0, 0])).toEqual(0);
    expect(b.locToIndex([0, 1])).toEqual(1);
    expect(b.locToIndex([1, 0])).toEqual(2);
    expect(b.locToIndex([1, 1])).toEqual(3);
    expect(b.locToIndex([2, 0])).toEqual(4);
    expect(b.locToIndex([2, 1])).toEqual(5);
  });

  it('locToIndex Tensor3D', () => {
    const a = tf.zeros<Rank.R3>([3, 2, 2]).buffer();
    expect(a.locToIndex([0, 0, 0])).toEqual(0);
    expect(a.locToIndex([0, 0, 1])).toEqual(1);
    expect(a.locToIndex([0, 1, 0])).toEqual(2);
    expect(a.locToIndex([0, 1, 1])).toEqual(3);
    expect(a.locToIndex([1, 0, 0])).toEqual(4);
    expect(a.locToIndex([1, 0, 1])).toEqual(5);
    expect(a.locToIndex([2, 1, 1])).toEqual(11);

    const b = tf.zeros<Rank.R3>([3, 2, 2]).buffer();
    expect(b.locToIndex([0, 0, 0])).toEqual(0);
    expect(b.locToIndex([0, 0, 1])).toEqual(1);
    expect(b.locToIndex([0, 1, 0])).toEqual(2);
    expect(b.locToIndex([0, 1, 1])).toEqual(3);
    expect(b.locToIndex([1, 0, 0])).toEqual(4);
    expect(b.locToIndex([1, 0, 1])).toEqual(5);
    expect(b.locToIndex([2, 1, 1])).toEqual(11);
  });

  it('Tensor assignability (asserts compiler)', () => {
    // This test asserts compilation, not doing any run-time assertion.
    const a: Tensor<Rank.R0> = null;
    const b: Scalar = a;
    expect(b).toBeNull();

    const a1: Tensor<Rank.R1> = null;
    const b1: Tensor1D = a1;
    expect(b1).toBeNull();

    const a2: Tensor<Rank.R2> = null;
    const b2: Tensor2D = a2;
    expect(b2).toBeNull();

    const a3: Tensor<Rank.R3> = null;
    const b3: Tensor3D = a3;
    expect(b3).toBeNull();

    const a4: Tensor<Rank.R4> = null;
    const b4: Tensor4D = a4;
    expect(b4).toBeNull();
  });

  it('tf.tensor1d() from number[]', () => {
    const a = tf.tensor1d([1, 2, 3]);
    expectArraysClose(a, [1, 2, 3]);
  });

  it('tf.tensor1d() from number[][], shape mismatch', () => {
    // tslint:disable-next-line:no-any
    expect(() => tf.tensor1d([[1], [2], [3]] as any)).toThrowError();
  });

  it('tf.tensor2d() from number[][]', () => {
    const a = tf.tensor2d([[1, 2, 3], [4, 5, 6]], [2, 3]);
    expectArraysClose(a, [1, 2, 3, 4, 5, 6]);
  });

  it('tf.tensor2d() requires shape to be of length 2', () => {
    // tslint:disable-next-line:no-any
    const shape: any = [4];
    expect(() => tf.tensor2d([1, 2, 3, 4], shape)).toThrowError();
  });

  it('tf.tensor2d() from number[][], but shape does not match', () => {
    // Actual shape is [2, 3].
    expect(() => tf.tensor2d([[1, 2, 3], [4, 5, 6]], [3, 2])).toThrowError();
  });

  it('tf.tensor2d() from number[], but no shape throws error', () => {
    expect(() => tf.tensor2d([1, 2, 3, 4])).toThrowError();
  });

  it('tensor3d() from number[][][]', () => {
    const a = tf.tensor3d([[[1], [2], [3]], [[4], [5], [6]]], [2, 3, 1]);
    expectArraysClose(a, [1, 2, 3, 4, 5, 6]);
  });

  it('tensor3d() from number[][][], but shape does not match', () => {
    const values = [[[1], [2], [3]], [[4], [5], [6]]];
    // Actual shape is [2, 3, 1].
    expect(() => tf.tensor3d(values, [3, 2, 1])).toThrowError();
  });

  it('tf.tensor3d() from number[], but no shape throws error', () => {
    expect(() => tf.tensor3d([1, 2, 3, 4])).toThrowError();
  });

  it('tf.tensor3d() requires shape to be of length 3', () => {
    // tslint:disable-next-line:no-any
    const shape: any = [4, 1];
    expect(() => tf.tensor3d([1, 2, 3, 4], shape)).toThrowError();
  });

  it('tensor4d() from number[][][][]', () => {
    const a = tf.tensor4d([[[[1]], [[2]]], [[[4]], [[5]]]], [2, 2, 1, 1]);
    expectArraysClose(a, [1, 2, 4, 5]);
  });

  it('tensor4d() from number[][][][], but shape does not match', () => {
    const f = () => {
      // Actual shape is [2, 2, 1, 1].
      tf.tensor4d([[[[1]], [[2]]], [[[4]], [[5]]]], [2, 1, 2, 1]);
    };
    expect(f).toThrowError();
  });

  it('tf.tensor4d() from number[], but no shape throws error', () => {
    expect(() => tf.tensor4d([1, 2, 3, 4])).toThrowError();
  });

  it('tf.tensor4d() requires shape to be of length 4', () => {
    // tslint:disable-next-line:no-any
    const shape: any = [4, 1];
    expect(() => tf.tensor4d([1, 2, 3, 4], shape)).toThrowError();
  });

  it('default dtype', () => {
    const a = tf.scalar(3);
    expect(a.dtype).toBe('float32');
    expectArraysClose(a, [3]);
  });

  it('float32 dtype', () => {
    const a = tf.scalar(3, 'float32');
    expect(a.dtype).toBe('float32');
    expectArraysClose(a, [3]);
  });

  it('int32 dtype', () => {
    const a = tf.scalar(3, 'int32');
    expect(a.dtype).toBe('int32');
    expectArraysEqual(a, [3]);
  });

  it('int32 dtype, 3.9 => 3, like numpy', () => {
    const a = tf.scalar(3.9, 'int32');
    expect(a.dtype).toBe('int32');
    expectArraysEqual(a, [3]);
  });

  it('int32 dtype, -3.9 => -3, like numpy', () => {
    const a = tf.scalar(-3.9, 'int32');
    expect(a.dtype).toBe('int32');
    expectArraysEqual(a, [-3]);
  });

  it('bool dtype, 3 => true, like numpy', () => {
    const a = tf.scalar(3, 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.get()).toBe(1);
  });

  it('bool dtype, -2 => true, like numpy', () => {
    const a = tf.scalar(-2, 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.get()).toBe(1);
  });

  it('bool dtype, 0 => false, like numpy', () => {
    const a = tf.scalar(0, 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.get()).toBe(0);
  });

  it('bool dtype from boolean', () => {
    const a = tf.scalar(false, 'bool');
    expect(a.get()).toBe(0);
    expect(a.dtype).toBe('bool');

    const b = tf.scalar(true, 'bool');
    expect(b.get()).toBe(1);
    expect(b.dtype).toBe('bool');
  });

  it('int32 dtype from boolean', () => {
    const a = tf.scalar(true, 'int32');
    expect(a.get()).toBe(1);
    expect(a.dtype).toBe('int32');
  });

  it('default dtype from boolean', () => {
    const a = tf.scalar(false);
    expectNumbersClose(a.get(), 0);
    expect(a.dtype).toBe('float32');
  });

  it('default dtype', () => {
    const a = tf.tensor1d([1, 2, 3]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    expectArraysClose(a, [1, 2, 3]);
  });

  it('float32 dtype', () => {
    const a = tf.tensor1d([1, 2, 3], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    expectArraysClose(a, [1, 2, 3]);
  });

  it('int32 dtype', () => {
    const a = tf.tensor1d([1, 2, 3], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3]);
    expectArraysEqual(a, [1, 2, 3]);
  });

  it('int32 dtype, non-ints get floored, like numpy', () => {
    const a = tf.tensor1d([1.1, 2.5, 3.9], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3]);
    expectArraysEqual(a, [1, 2, 3]);
  });

  it('int32 dtype, negative non-ints get ceiled, like numpy', () => {
    const a = tf.tensor1d([-1.1, -2.5, -3.9], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3]);
    expectArraysEqual(a, [-1, -2, -3]);
  });

  it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', () => {
    const a = tf.tensor1d([1, -2, 0, 3], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([4]);
    expect(a.get(0)).toBe(1);
    expect(a.get(1)).toBe(1);
    expect(a.get(2)).toBe(0);
    expect(a.get(3)).toBe(1);
  });

  it('default dtype from boolean[]', () => {
    const a = tf.tensor1d([false, false, true]);
    expect(a.dtype).toBe('float32');
    expectArraysClose(a, [0, 0, 1]);
  });

  it('int32 dtype from boolean[]', () => {
    const a = tf.tensor1d([false, false, true], 'int32');
    expect(a.dtype).toBe('int32');
    expectArraysEqual(a, [0, 0, 1]);
  });

  it('bool dtype from boolean[]', () => {
    const a = tf.tensor1d([false, false, true], 'bool');
    expect(a.dtype).toBe('bool');
    expectArraysEqual(a, [0, 0, 1]);
  });

  it('default dtype', () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2]);
    expectArraysClose(a, [1, 2, 3, 4]);
  });

  it('float32 dtype', () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2]);
    expectArraysClose(a, [1, 2, 3, 4]);
  });

  it('int32 dtype', () => {
    const a = tf.tensor2d([[1, 2], [3, 4]], [2, 2], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2]);
    expectArraysEqual(a, [1, 2, 3, 4]);
  });

  it('int32 dtype, non-ints get floored, like numpy', () => {
    const a = tf.tensor2d([1.1, 2.5, 3.9, 4.0], [2, 2], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2]);
    expectArraysEqual(a, [1, 2, 3, 4]);
  });

  it('int32 dtype, negative non-ints get ceiled, like numpy', () => {
    const a = tf.tensor2d([-1.1, -2.5, -3.9, -4.0], [2, 2], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2]);
    expectArraysEqual(a, [-1, -2, -3, -4]);
  });

  it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', () => {
    const a = tf.tensor2d([1, -2, 0, 3], [2, 2], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([2, 2]);
    expect(a.get(0, 0)).toBe(1);
    expect(a.get(0, 1)).toBe(1);
    expect(a.get(1, 0)).toBe(0);
    expect(a.get(1, 1)).toBe(1);
  });

  it('default dtype from boolean[]', () => {
    const a = tf.tensor2d([[false, false], [true, false]], [2, 2]);
    expect(a.dtype).toBe('float32');
    expectArraysClose(a, [0, 0, 1, 0]);
  });

  it('int32 dtype from boolean[]', () => {
    const a = tf.tensor2d([[false, false], [true, false]], [2, 2], 'int32');
    expect(a.dtype).toBe('int32');
    expectArraysEqual(a, [0, 0, 1, 0]);
  });

  it('bool dtype from boolean[]', () => {
    const a = tf.tensor2d([[false, false], [true, false]], [2, 2], 'bool');
    expect(a.dtype).toBe('bool');
    expectArraysEqual(a, [0, 0, 1, 0]);
  });
  it('default dtype', () => {
    const a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 1]);
    expectArraysClose(a, [1, 2, 3, 4]);
  });

  it('float32 dtype', () => {
    const a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 1]);
    expectArraysClose(a, [1, 2, 3, 4]);
  });

  it('int32 dtype', () => {
    const a = tf.tensor3d([[[1], [2]], [[3], [4]]], [2, 2, 1], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 1]);
    expectArraysEqual(a, [1, 2, 3, 4]);
  });

  it('int32 dtype, non-ints get floored, like numpy', () => {
    const a = tf.tensor3d([1.1, 2.5, 3.9, 4.0], [2, 2, 1], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 1]);
    expectArraysEqual(a, [1, 2, 3, 4]);
  });

  it('int32 dtype, negative non-ints get ceiled, like numpy', () => {
    const a = tf.tensor3d([-1.1, -2.5, -3.9, -4.0], [2, 2, 1], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 1]);
    expectArraysEqual(a, [-1, -2, -3, -4]);
  });

  it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', () => {
    const a = tf.tensor3d([1, -2, 0, 3], [2, 2, 1], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([2, 2, 1]);
    expect(a.get(0, 0, 0)).toBe(1);
    expect(a.get(0, 1, 0)).toBe(1);
    expect(a.get(1, 0, 0)).toBe(0);
    expect(a.get(1, 1, 0)).toBe(1);
  });

  it('default dtype from boolean[]', () => {
    const a = tf.tensor3d([[[false], [false]], [[true], [false]]], [2, 2, 1]);
    expect(a.dtype).toBe('float32');
    expectArraysClose(a, [0, 0, 1, 0]);
  });

  it('int32 dtype from boolean[]', () => {
    const a = tf.tensor3d(
        [[[false], [false]], [[true], [false]]], [2, 2, 1], 'int32');
    expect(a.dtype).toBe('int32');
    expectArraysEqual(a, [0, 0, 1, 0]);
  });

  it('bool dtype from boolean[]', () => {
    const a =
        tf.tensor3d([[[false], [false]], [[true], [false]]], [2, 2, 1], 'bool');
    expect(a.dtype).toBe('bool');
    expectArraysEqual(a, [0, 0, 1, 0]);
  });

  it('default dtype', () => {
    const a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    expectArraysClose(a, [1, 2, 3, 4]);
  });

  it('float32 dtype', () => {
    const a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    expectArraysClose(a, [1, 2, 3, 4]);
  });

  it('int32 dtype', () => {
    const a =
        tf.tensor4d([[[[1]], [[2]]], [[[3]], [[4]]]], [2, 2, 1, 1], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    expectArraysEqual(a, [1, 2, 3, 4]);
  });

  it('int32 dtype, non-ints get floored, like numpy', () => {
    const a = tf.tensor4d([1.1, 2.5, 3.9, 4.0], [2, 2, 1, 1], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    expectArraysEqual(a, [1, 2, 3, 4]);
  });

  it('int32 dtype, negative non-ints get ceiled, like numpy', () => {
    const a = tf.tensor4d([-1.1, -2.5, -3.9, -4.0], [2, 2, 1, 1], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    expectArraysEqual(a, [-1, -2, -3, -4]);
  });

  it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', () => {
    const a = tf.tensor4d([1, -2, 0, 3], [2, 2, 1, 1], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    expect(a.get(0, 0, 0, 0)).toBe(1);
    expect(a.get(0, 1, 0, 0)).toBe(1);
    expect(a.get(1, 0, 0, 0)).toBe(0);
    expect(a.get(1, 1, 0, 0)).toBe(1);
  });

  it('default dtype from boolean[]', () => {
    const a =
        tf.tensor4d([[[[false], [false]], [[true], [false]]]], [1, 2, 2, 1]);
    expect(a.dtype).toBe('float32');
    expectArraysClose(a, [0, 0, 1, 0]);
  });

  it('int32 dtype from boolean[]', () => {
    const a = tf.tensor4d(
        [[[[false], [false]], [[true], [false]]]], [1, 2, 2, 1], 'int32');
    expect(a.dtype).toBe('int32');
    expectArraysEqual(a, [0, 0, 1, 0]);
  });

  it('bool dtype from boolean[]', () => {
    const a = tf.tensor4d(
        [[[[false], [false]], [[true], [false]]]], [1, 2, 2, 1], 'bool');
    expect(a.dtype).toBe('bool');
    expectArraysEqual(a, [0, 0, 1, 0]);
  });

  it('Scalar default dtype', () => {
    const a = tf.scalar(4);
    const b = a.reshape([1, 1]);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([1, 1]);
  });

  it('Scalar bool dtype', () => {
    const a = tf.scalar(4, 'bool');
    const b = a.reshape([1, 1, 1]);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([1, 1, 1]);
  });

  it('Tensor1D default dtype', () => {
    const a = tf.tensor1d([1, 2, 3, 4]);
    const b = a.reshape([2, 2]);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
  });

  it('Tensor1D int32 dtype', () => {
    const a = tf.tensor1d([1, 2, 3, 4], 'int32');
    const b = a.reshape([2, 2]);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2]);
  });

  it('Tensor2D default dtype', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = a.reshape([6]);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([6]);
  });

  it('Tensor2D bool dtype', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3], 'bool');
    const b = a.reshape([6]);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([6]);
  });

  it('Tensor3D default dtype', () => {
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6], [2, 3, 1]);
    const b = a.reshape([6]);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([6]);
  });

  it('Tensor3D bool dtype', () => {
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6], [2, 3, 1], 'bool');
    const b = a.reshape([6]);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([6]);
  });

  it('Tensor4D default dtype', () => {
    const a = tf.tensor4d([1, 2, 3, 4, 5, 6], [2, 3, 1, 1]);
    const b = a.reshape([2, 3]);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 3]);
  });

  it('Tensor4D int32 dtype', () => {
    const a = tf.tensor4d([1, 2, 3, 4, 5, 6], [2, 3, 1, 1], 'int32');
    const b = a.reshape([3, 2]);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([3, 2]);
  });

  it('reshape is functional', () => {
    const a = tf.scalar(2.4);
    const b = a.reshape([]);
    expect(a.id).not.toBe(b.id);
    b.dispose();
    expectArraysClose(a, [2.4]);
  });

  it('reshape throws when passed a non-tensor', () => {
    // tslint:disable-next-line:no-any
    expect(() => tf.reshape({} as any, []))
        .toThrowError(/Argument 'x' passed to 'reshape' must be a Tensor/);
  });

  it('cast bool -> bool', () => {
    const a = tf.tensor1d([1, 0], 'bool');
    expect(a.cast('bool').dtype).toEqual('bool');
  });

  it('cast bool -> int32', () => {
    const a = tf.tensor1d([1, 0], 'bool');
    expect(a.cast('int32').dtype).toEqual('int32');
  });

  it('cast bool -> float32', () => {
    const a = tf.tensor1d([1, 0], 'bool');
    expect(a.cast('float32').dtype).toEqual('float32');
  });

  it('cast int32 -> bool', () => {
    const a = tf.tensor1d([1, 0], 'int32');
    expect(a.cast('bool').dtype).toEqual('bool');
  });

  it('cast int32 -> int32', () => {
    const a = tf.tensor1d([1, 2], 'int32');
    expect(a.cast('int32').dtype).toEqual('int32');
  });

  it('cast int32 -> float32', () => {
    const a = tf.tensor1d([1, 2], 'int32');
    expect(a.cast('float32').dtype).toEqual('float32');
  });

  it('cast float32 -> bool', () => {
    const a = tf.tensor1d([1.0, 0.0]);
    expect(a.cast('bool').dtype).toEqual('bool');
  });

  it('cast float32 -> int32', () => {
    const a = tf.tensor1d([1.0, 2.0]);
    expect(a.cast('int32').dtype).toEqual('int32');
  });

  it('cast float32 -> float32', () => {
    const a = tf.tensor1d([1.0, 2.0]);
    expect(a.cast('float32').dtype).toEqual('float32');
  });

  it('cast throws when passed a non-tensor', () => {
    expect(() => tf.cast({} as tf.Tensor, 'float32'))
        .toThrowError(/Argument 'x' passed to 'cast' must be a Tensor/);
  });

  it('scalar bool -> int32', () => {
    const a = tf.scalar(true, 'bool').toInt();
    expect(a.dtype).toBe('int32');
    expect(a.get()).toBe(1);
  });

  it('Tensor1D float32 -> int32', () => {
    const a = tf.tensor1d([1.1, 3.9, -2.9, 0]).toInt();
    expect(a.dtype).toBe('int32');
    expectArraysEqual(a, [1, 3, -2, 0]);
  });

  it('Tensor2D float32 -> bool', () => {
    const a = tf.tensor2d([1.1, 3.9, -2.9, 0], [2, 2]).asType(DType.bool);
    expect(a.dtype).toBe('bool');
    expect(a.get(0, 0)).toBe(1);
    expect(a.get(0, 1)).toBe(1);
    expect(a.get(1, 0)).toBe(1);
    expect(a.get(1, 1)).toBe(0);
  });

  it('Tensor2D int32 -> bool', () => {
    const a = tf.tensor2d([1, 3, 0, -1], [2, 2], 'int32').toBool();
    expect(a.dtype).toBe('bool');
    expect(a.get(0, 0)).toBe(1);
    expect(a.get(0, 1)).toBe(1);
    expect(a.get(1, 0)).toBe(0);
    expect(a.get(1, 1)).toBe(1);
  });

  it('Tensor3D bool -> float32', () => {
    const a =
        tf.tensor3d([true, false, false, true], [2, 2, 1], 'bool').toFloat();
    expect(a.dtype).toBe('float32');
    expectArraysClose(a, [1, 0, 0, 1]);
  });

  it('bool CPU -> GPU -> CPU', () => {
    const a = tf.tensor1d([1, 2, 0, 0, 5], 'bool');
    expectArraysEqual(a, [1, 1, 0, 0, 1]);
  });

  it('int32 CPU -> GPU -> CPU', () => {
    const a = tf.tensor1d([1, 2, 0, 0, 5], 'int32');
    expectArraysEqual(a, [1, 2, 0, 0, 5]);
  });

  it('asType is functional', () => {
    const a = tf.scalar(2.4, 'float32');
    const b = a.toFloat();
    expect(a.id).not.toBe(b.id);
    b.dispose();
    expectArraysClose(a, [2.4]);
  });

  it('squeeze no axis', () => {
    const a = tf.tensor2d([4, 2, 1], [3, 1], 'bool');
    const b = a.squeeze();
    expect(b.shape).toEqual([3]);
  });

  it('squeeze with axis', () => {
    const a = tf.tensor3d([4, 2, 1], [3, 1, 1], 'bool');
    const b = a.squeeze([1]);
    expect(b.shape).toEqual([3, 1]);
  });

  it('squeeze wrong axis', () => {
    const a = tf.tensor3d([4, 2, 1], [3, 1, 1], 'bool');
    expect(() => a.squeeze([0, 1])).toThrowError();
  });

  it('squeeze throws when passed a non-tensor', () => {
    expect(() => tf.squeeze({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'squeeze' must be a Tensor/);
  });

  it('scalar -> 2d', () => {
    const a = tf.scalar(4, 'int32');
    const b = a.as2D(1, 1);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([1, 1]);
  });

  it('1d -> 2d', () => {
    const a = tf.tensor1d([4, 2, 1], 'bool');
    const b = a.as2D(3, 1);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([3, 1]);
  });

  it('2d -> 4d', () => {
    const a = tf.tensor2d([4, 2, 1, 3], [2, 2]);
    const b = a.as4D(1, 1, 2, 2);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([1, 1, 2, 2]);
  });

  it('3d -> 2d', () => {
    const a = tf.tensor3d([4, 2, 1, 3], [2, 2, 1], 'float32');
    const b = a.as2D(2, 2);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
  });

  it('4d -> 1d', () => {
    const a = tf.tensor4d([4, 2, 1, 3], [2, 2, 1, 1], 'bool');
    const b = a.as1D();
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([4]);
  });
});

describe('tensor.toString', () => {
  it('scalar verbose', () => {
    const verbose = true;
    const str = tf.scalar(5).toString(verbose);
    expect(str).toEqual(
        'Tensor\n' +
        '  dtype: float32\n' +
        '  rank: 0\n' +
        '  shape: []\n' +
        '  values:\n' +
        '    5');
  });

  it('1d tensor verbose', () => {
    const verbose = true;
    const str = tf.zeros([4]).toString(verbose);
    expect(str).toEqual(
        'Tensor\n' +
        '  dtype: float32\n' +
        '  rank: 1\n' +
        '  shape: [4]\n' +
        '  values:\n' +
        '    [0, 0, 0, 0]');
  });

  it('2d tensor verbose', () => {
    const verbose = true;
    const str = tf.zeros([3, 3]).toString(verbose);
    expect(str).toEqual(
        'Tensor\n' +
        '  dtype: float32\n' +
        '  rank: 2\n' +
        '  shape: [3,3]\n' +
        '  values:\n' +
        '    [[0, 0, 0],\n' +
        '     [0, 0, 0],\n' +
        '     [0, 0, 0]]');
  });

  it('3d tensor verbose', () => {
    const verbose = true;
    const str = tf.zeros([3, 3, 2]).toString(verbose);
    expect(str).toEqual(
        'Tensor\n' +
        '  dtype: float32\n' +
        '  rank: 3\n' +
        '  shape: [3,3,2]\n' +
        '  values:\n' +
        '    [[[0, 0],\n' +
        '      [0, 0],\n' +
        '      [0, 0]],\n\n' +
        '     [[0, 0],\n' +
        '      [0, 0],\n' +
        '      [0, 0]],\n\n' +
        '     [[0, 0],\n' +
        '      [0, 0],\n' +
        '      [0, 0]]]');
  });

  it('1d long tensor verbose', () => {
    const verbose = true;
    const str = tf.zeros([100]).toString(verbose);
    expect(str).toEqual(
        'Tensor\n' +
        '  dtype: float32\n' +
        '  rank: 1\n' +
        '  shape: [100]\n' +
        '  values:\n' +
        '    [0, 0, 0, ..., 0, 0, 0]');
  });

  it('2d long tensor verbose', () => {
    const verbose = true;
    const str = tf.zeros([100, 100]).toString(verbose);
    expect(str).toEqual(
        'Tensor\n' +
        '  dtype: float32\n' +
        '  rank: 2\n' +
        '  shape: [100,100]\n' +
        '  values:\n' +
        '    [[0, 0, 0, ..., 0, 0, 0],\n' +
        '     [0, 0, 0, ..., 0, 0, 0],\n' +
        '     [0, 0, 0, ..., 0, 0, 0],\n' +
        '     ...,\n' +
        '     [0, 0, 0, ..., 0, 0, 0],\n' +
        '     [0, 0, 0, ..., 0, 0, 0],\n' +
        '     [0, 0, 0, ..., 0, 0, 0]]');
  });

  it('2d with padding to align columns verbose', () => {
    const verbose = true;
    const str = tf.tensor([
                    [0.8597712, 3, 0.2740789], [0.6696132, 0.4825962, 2.75],
                    [1.991, 0.0640865, 0.2983858]
                  ]).toString(verbose);
    expect(str).toEqual(
        'Tensor\n' +
        '  dtype: float32\n' +
        '  rank: 2\n' +
        '  shape: [3,3]\n' +
        '  values:\n' +
        '    [[0.8597712, 3        , 0.2740789],\n' +
        '     [0.6696132, 0.4825962, 2.75     ],\n' +
        '     [1.9910001, 0.0640865, 0.2983858]]');
  });

  it('scalar', () => {
    const str = tf.scalar(5).toString();
    expect(str).toEqual(
        'Tensor\n' +
        '    5');
  });

  it('1d tensor', () => {
    const str = tf.zeros([4]).toString();
    expect(str).toEqual(
        'Tensor\n' +
        '    [0, 0, 0, 0]');
  });

  it('2d tensor', () => {
    const str = tf.zeros([3, 3]).toString();
    expect(str).toEqual(
        'Tensor\n' +
        '    [[0, 0, 0],\n' +
        '     [0, 0, 0],\n' +
        '     [0, 0, 0]]');
  });

  it('3d tensor', () => {
    const str = tf.zeros([3, 3, 2]).toString();
    expect(str).toEqual(
        'Tensor\n' +
        '    [[[0, 0],\n' +
        '      [0, 0],\n' +
        '      [0, 0]],\n\n' +
        '     [[0, 0],\n' +
        '      [0, 0],\n' +
        '      [0, 0]],\n\n' +
        '     [[0, 0],\n' +
        '      [0, 0],\n' +
        '      [0, 0]]]');
  });

  it('1d long tensor', () => {
    const str = tf.zeros([100]).toString();
    expect(str).toEqual(
        'Tensor\n' +
        '    [0, 0, 0, ..., 0, 0, 0]');
  });

  it('2d long tensor', () => {
    const str = tf.zeros([100, 100]).toString();
    expect(str).toEqual(
        'Tensor\n' +
        '    [[0, 0, 0, ..., 0, 0, 0],\n' +
        '     [0, 0, 0, ..., 0, 0, 0],\n' +
        '     [0, 0, 0, ..., 0, 0, 0],\n' +
        '     ...,\n' +
        '     [0, 0, 0, ..., 0, 0, 0],\n' +
        '     [0, 0, 0, ..., 0, 0, 0],\n' +
        '     [0, 0, 0, ..., 0, 0, 0]]');
  });

  it('2d with padding to align columns', () => {
    const str = tf.tensor([
                    [0.8597712, 3, 0.2740789], [0.6696132, 0.4825962, 2.75],
                    [1.991, 0.0640865, 0.2983858]
                  ]).toString();
    expect(str).toEqual(
        'Tensor\n' +
        '    [[0.8597712, 3        , 0.2740789],\n' +
        '     [0.6696132, 0.4825962, 2.75     ],\n' +
        '     [1.9910001, 0.0640865, 0.2983858]]');
  });
});

describeWithFlags('tensor grad', ALL_ENVS, () => {
  it('grad with second derivative', () => {
    // f(x) = x ^ 3
    const f = (x: Tensor) => x.pow(tf.scalar(3, 'int32'));
    // f'(x) = 3x ^ 2
    const g = tf.grad(f);
    // f''(x) = 6x
    const gg = tf.grad(g);
    const x = tf.tensor1d([2, 3]);
    const data = gg(x);
    expectArraysClose(data, [12, 18]);
  });
});

describeWithFlags('tensor.data', ALL_ENVS, () => {
  it('.data() postpones disposal of tensor', done => {
    expect(tf.memory().numTensors).toBe(0);
    tf.tidy(() => {
      const a = tf.scalar(5);
      expect(tf.memory().numTensors).toBe(1);
      a.square();  // Uploads it on GPU.
      a.data().then(vals => {
        // The tidy above should not dispose the scalar since there is
        // a pending data read.
        expectNumbersClose(vals[0], 5);
      });
    });

    // tidy ends immediately, but should not dispose the scalar.

    setTimeout(() => {
      // tidy should dispose the tensor.
      expect(tf.memory().numTensors).toBe(0);
      done();
    });
  });

  it('calling .data() twice works (2 subscribers to a single read)', done => {
    tf.tidy(() => {
      const a = tf.scalar(5);
      a.square();  // Uploads it on GPU.
      a.data().then(vals => {
        expectNumbersClose(vals[0], 5);
      });
      a.data()
          .then(vals => {
            expectNumbersClose(vals[0], 5);
          })
          .then(done);
    });
    // tidy ends immediately, but should not dispose the scalar since there is
    // a pending data read.
  });
});
