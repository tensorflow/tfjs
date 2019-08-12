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
import {ALL_ENVS, describeWithFlags, SYNC_BACKEND_ENVS} from './jasmine_util';
import {Scalar, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D} from './tensor';
import {expectArraysClose, expectArraysEqual, expectNumbersClose} from './test_util';
import {Rank, RecursiveArray, TensorLike1D, TensorLike2D, TensorLike3D, TensorLike4D, TypedArray} from './types';
import {encodeString} from './util';

/** Private method used by these tests. Encodes strings into utf-8 bytes. */
function encodeStrings(a: RecursiveArray<{}>): RecursiveArray<Uint8Array> {
  for (let i = 0; i < (a as Array<{}>).length; i++) {
    const val = a[i];
    if (Array.isArray(val)) {
      encodeStrings(val);
    } else {
      a[i] = encodeString(val as string);
    }
  }
  return a;
}

describeWithFlags('tensor', ALL_ENVS, () => {
  it('Tensors of arbitrary size', async () => {
    // [1, 2, 3]
    let t: Tensor = tf.tensor1d([1, 2, 3]);
    expect(t.rank).toBe(1);
    expect(t.size).toBe(3);
    expectArraysClose(await t.data(), [1, 2, 3]);

    // [[1, 2, 3]]
    t = tf.tensor2d([1, 2, 3], [1, 3]);
    expect(t.rank).toBe(2);
    expect(t.size).toBe(3);
    expectArraysClose(await t.data(), [1, 2, 3]);

    // [[1, 2, 3],
    //  [4, 5, 6]]
    t = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    expect(t.rank).toBe(2);
    expect(t.size).toBe(6);

    expectArraysClose(await t.data(), [1, 2, 3, 4, 5, 6]);

    // Shape mismatch with the values.
    expect(() => tf.tensor2d([1], [1, 2])).toThrowError();
  });

  it('Tensors of explicit size', async () => {
    const t = tf.tensor1d([5, 3, 2]);
    expect(t.rank).toBe(1);
    expect(t.shape).toEqual([3]);

    // tslint:disable-next-line:no-any
    expect(() => tf.tensor3d([1, 2], [1, 2, 3, 5] as any)).toThrowError();

    const t4 = tf.tensor4d([1, 2, 3, 4], [1, 2, 1, 2]);
    expectArraysClose(await t4.data(), [1, 2, 3, 4]);

    // Tensor of ones.
    const x = tf.ones<Rank.R3>([3, 4, 2]);
    expect(x.rank).toBe(3);
    expect(x.size).toBe(24);
    expectArraysClose(await x.data(), [
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ]);

    // Tensor of zeros.
    const z = tf.zeros<Rank.R3>([3, 4, 2]);
    expect(z.rank).toBe(3);
    expect(z.size).toBe(24);
    expectArraysClose(await z.data(), [
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
  });

  it('Tensor dataSync CPU --> GPU', async () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
    expectArraysClose(await a.data(), new Float32Array([1, 2, 3, 4, 5, 6]));
  });

  it('Tensor.data() CPU --> GPU', async () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
    expectArraysClose(await a.data(), new Float32Array([1, 2, 3, 4, 5, 6]));
  });

  it('Tensor.data() packed CPU --> GPU', async () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
    tf.matMul(a, tf.tensor2d([1, 2], [2, 1]));
    expectArraysClose(await a.data(), new Float32Array([1, 2, 3, 4, 5, 6]));
  });

  it('Scalar basic methods', async () => {
    const a = tf.scalar(5);
    expectArraysClose(await a.data(), [5]);
    expect(a.rank).toBe(0);
    expect(a.size).toBe(1);
    expect(a.shape).toEqual([]);
  });

  it('indexToLoc Scalar', async () => {
    const a = await tf.scalar(0).buffer();
    expect(a.indexToLoc(0)).toEqual([]);

    const b = await tf.zeros<Rank.R0>([]).buffer();
    expect(b.indexToLoc(0)).toEqual([]);
  });

  it('indexToLoc Tensor1D', async () => {
    const a = await tf.zeros([3]).buffer();
    expect(a.indexToLoc(0)).toEqual([0]);
    expect(a.indexToLoc(1)).toEqual([1]);
    expect(a.indexToLoc(2)).toEqual([2]);

    const b = await tf.zeros<Rank.R1>([3]).buffer();
    expect(b.indexToLoc(0)).toEqual([0]);
    expect(b.indexToLoc(1)).toEqual([1]);
    expect(b.indexToLoc(2)).toEqual([2]);
  });

  it('indexToLoc Tensor2D', async () => {
    const a = await tf.zeros([3, 2]).buffer();
    expect(a.indexToLoc(0)).toEqual([0, 0]);
    expect(a.indexToLoc(1)).toEqual([0, 1]);
    expect(a.indexToLoc(2)).toEqual([1, 0]);
    expect(a.indexToLoc(3)).toEqual([1, 1]);
    expect(a.indexToLoc(4)).toEqual([2, 0]);
    expect(a.indexToLoc(5)).toEqual([2, 1]);

    const b = await tf.zeros<Rank.R2>([3, 2]).buffer();
    expect(b.indexToLoc(0)).toEqual([0, 0]);
    expect(b.indexToLoc(1)).toEqual([0, 1]);
    expect(b.indexToLoc(2)).toEqual([1, 0]);
    expect(b.indexToLoc(3)).toEqual([1, 1]);
    expect(b.indexToLoc(4)).toEqual([2, 0]);
    expect(b.indexToLoc(5)).toEqual([2, 1]);
  });

  it('indexToLoc Tensor3D', async () => {
    const a = await tf.zeros([3, 2, 2]).buffer();
    expect(a.indexToLoc(0)).toEqual([0, 0, 0]);
    expect(a.indexToLoc(1)).toEqual([0, 0, 1]);
    expect(a.indexToLoc(2)).toEqual([0, 1, 0]);
    expect(a.indexToLoc(3)).toEqual([0, 1, 1]);
    expect(a.indexToLoc(4)).toEqual([1, 0, 0]);
    expect(a.indexToLoc(5)).toEqual([1, 0, 1]);
    expect(a.indexToLoc(11)).toEqual([2, 1, 1]);

    const b = await tf.zeros<Rank.R3>([3, 2, 2]).buffer();
    expect(b.indexToLoc(0)).toEqual([0, 0, 0]);
    expect(b.indexToLoc(1)).toEqual([0, 0, 1]);
    expect(b.indexToLoc(2)).toEqual([0, 1, 0]);
    expect(b.indexToLoc(3)).toEqual([0, 1, 1]);
    expect(b.indexToLoc(4)).toEqual([1, 0, 0]);
    expect(b.indexToLoc(5)).toEqual([1, 0, 1]);
    expect(b.indexToLoc(11)).toEqual([2, 1, 1]);
  });

  it('indexToLoc Tensor 5D', async () => {
    const values = new Float32Array([1, 2, 3, 4]);
    const a = await Tensor.make([2, 1, 1, 1, 2], {values}).buffer();
    expect(a.indexToLoc(0)).toEqual([0, 0, 0, 0, 0]);
    expect(a.indexToLoc(1)).toEqual([0, 0, 0, 0, 1]);
    expect(a.indexToLoc(2)).toEqual([1, 0, 0, 0, 0]);
    expect(a.indexToLoc(3)).toEqual([1, 0, 0, 0, 1]);
  });

  it('locToIndex Scalar', async () => {
    const a = await tf.scalar(0).buffer();
    expect(a.locToIndex([])).toEqual(0);

    const b = await tf.zeros<Rank.R0>([]).buffer();
    expect(b.locToIndex([])).toEqual(0);
  });

  it('locToIndex Tensor1D', async () => {
    const a = await tf.zeros<Rank.R1>([3]).buffer();
    expect(a.locToIndex([0])).toEqual(0);
    expect(a.locToIndex([1])).toEqual(1);
    expect(a.locToIndex([2])).toEqual(2);

    const b = await tf.zeros<Rank.R1>([3]).buffer();
    expect(b.locToIndex([0])).toEqual(0);
    expect(b.locToIndex([1])).toEqual(1);
    expect(b.locToIndex([2])).toEqual(2);
  });

  it('locToIndex Tensor2D', async () => {
    const a = await tf.zeros<Rank.R2>([3, 2]).buffer();
    expect(a.locToIndex([0, 0])).toEqual(0);
    expect(a.locToIndex([0, 1])).toEqual(1);
    expect(a.locToIndex([1, 0])).toEqual(2);
    expect(a.locToIndex([1, 1])).toEqual(3);
    expect(a.locToIndex([2, 0])).toEqual(4);
    expect(a.locToIndex([2, 1])).toEqual(5);

    const b = await tf.zeros<Rank.R2>([3, 2]).buffer();
    expect(b.locToIndex([0, 0])).toEqual(0);
    expect(b.locToIndex([0, 1])).toEqual(1);
    expect(b.locToIndex([1, 0])).toEqual(2);
    expect(b.locToIndex([1, 1])).toEqual(3);
    expect(b.locToIndex([2, 0])).toEqual(4);
    expect(b.locToIndex([2, 1])).toEqual(5);
  });

  it('locToIndex Tensor3D', async () => {
    const a = await tf.zeros<Rank.R3>([3, 2, 2]).buffer();
    expect(a.locToIndex([0, 0, 0])).toEqual(0);
    expect(a.locToIndex([0, 0, 1])).toEqual(1);
    expect(a.locToIndex([0, 1, 0])).toEqual(2);
    expect(a.locToIndex([0, 1, 1])).toEqual(3);
    expect(a.locToIndex([1, 0, 0])).toEqual(4);
    expect(a.locToIndex([1, 0, 1])).toEqual(5);
    expect(a.locToIndex([2, 1, 1])).toEqual(11);

    const b = await tf.zeros<Rank.R3>([3, 2, 2]).buffer();
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

  it('tf.tensor1d() from number[]', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    expectArraysClose(await a.data(), [1, 2, 3]);
  });

  it('tf.tensor1d() throw error with null input value', () => {
    expect(() => tf.tensor1d(null))
        .toThrowError(
            'The input to the tensor constructor ' +
            'must be a non-null value.');
  });

  it('tf.tensor1d() from string[]', async () => {
    const a = tf.tensor1d(['aa', 'bb', 'cc']);
    expect(a.dtype).toBe('string');
    expect(a.shape).toEqual([3]);
    expectArraysEqual(await a.data(), ['aa', 'bb', 'cc']);
  });

  it('tf.tensor1d() from encoded strings', async () => {
    const bytes = encodeStrings(['aa', 'bb', 'cc']) as TensorLike1D;
    const a = tf.tensor1d(bytes, 'string');
    expect(a.dtype).toBe('string');
    expect(a.shape).toEqual([3]);
    expectArraysEqual(await a.data(), ['aa', 'bb', 'cc']);
  });

  it('tf.tensor1d() from encoded strings without dtype errors', async () => {
    // We do not want to infer 'string' when the user passes Uint8Array in order
    // to be forward compatible in the future when we add uint8 dtype.
    const bytes = encodeStrings(['aa', 'bb', 'cc']) as TensorLike1D;
    expect(() => tf.tensor1d(bytes)).toThrowError();
  });

  it('tf.tensor1d() from encoded strings, shape mismatch', () => {
    const bytes = encodeStrings([['aa'], ['bb'], ['cc']]) as TensorLike1D;
    expect(() => tf.tensor1d(bytes)).toThrowError();
  });

  it('tf.tensor1d() from number[][], shape mismatch', () => {
    // tslint:disable-next-line:no-any
    expect(() => tf.tensor1d([[1], [2], [3]] as any)).toThrowError();
  });

  it('tf.tensor1d() from string[][], shape mismatch', () => {
    // tslint:disable-next-line:no-any
    expect(() => tf.tensor1d([['a'], ['b'], ['c']] as any)).toThrowError();
  });

  it('tf.tensor2d() from number[][]', async () => {
    const a = tf.tensor2d([[1, 2, 3], [4, 5, 6]], [2, 3]);
    expectArraysClose(await a.data(), [1, 2, 3, 4, 5, 6]);
  });

  it('tf.tensor2d() from string[][]', async () => {
    const a = tf.tensor2d([['aa', 'bb'], ['cc', 'dd']]);
    expect(a.dtype).toBe('string');
    expect(a.shape).toEqual([2, 2]);
    expectArraysEqual(await a.data(), ['aa', 'bb', 'cc', 'dd']);
  });

  it('tf.tensor2d() from encoded strings', async () => {
    const bytes = encodeStrings([['aa', 'bb'], ['cc', 'dd']]) as TensorLike2D;
    const a = tf.tensor2d(bytes, [2, 2], 'string');
    expect(a.dtype).toBe('string');
    expect(a.shape).toEqual([2, 2]);
    expectArraysEqual(await a.data(), ['aa', 'bb', 'cc', 'dd']);
  });

  it('tf.tensor2d() from encoded strings without dtype errors', async () => {
    // We do not want to infer 'string' when the user passes Uint8Array in order
    // to be forward compatible in the future when we add uint8 dtype.
    const bytes = encodeStrings([['aa', 'bb'], ['cc', 'dd']]) as TensorLike2D;
    expect(() => tf.tensor2d(bytes)).toThrowError();
  });

  it('tf.tensor2d() from encoded strings, shape mismatch', () => {
    const bytes = encodeStrings([['aa', 'bb'], ['cc', 'dd']]) as TensorLike2D;
    expect(() => tf.tensor2d(bytes, [3, 2], 'string')).toThrowError();
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

  it('tf.tensor2d() from string[][], but shape does not match', () => {
    // Actual shape is [2, 3].
    const vals = [['a', 'b', 'c'], ['d', 'e', 'f']];
    expect(() => tf.tensor2d(vals, [3, 2])).toThrowError();
  });

  it('tf.tensor2d() from number[], but no shape throws error', () => {
    expect(() => tf.tensor2d([1, 2, 3, 4])).toThrowError();
  });

  it('tf.tensor2d() from string[], but no shape throws error', () => {
    expect(() => tf.tensor2d(['a', 'b', 'c', 'd'])).toThrowError();
  });

  it('tf.tensor2d() throw error with null input value', () => {
    expect(() => tf.tensor2d(null))
        .toThrowError(
            'The input to the tensor constructor ' +
            'must be a non-null value.');
  });

  it('tensor3d() from number[][][]', async () => {
    const a = tf.tensor3d([[[1], [2], [3]], [[4], [5], [6]]], [2, 3, 1]);
    expectArraysClose(await a.data(), [1, 2, 3, 4, 5, 6]);
  });

  it('tensor3d() from string[][][]', async () => {
    const vals = [[['a'], ['b'], ['c']], [['d'], ['e'], ['f']]];
    const a = tf.tensor3d(vals, [2, 3, 1]);
    expect(a.dtype).toBe('string');
    expect(a.shape).toEqual([2, 3, 1]);
    expectArraysEqual(await a.data(), ['a', 'b', 'c', 'd', 'e', 'f']);
  });

  it('tf.tensor3d() from encoded strings', async () => {
    const bytes = encodeStrings([[['a'], ['b'], ['c']], [['d'], ['e'], ['f']]]);
    const a = tf.tensor3d(bytes as TensorLike3D, [2, 3, 1], 'string');
    expect(a.dtype).toBe('string');
    expect(a.shape).toEqual([2, 3, 1]);
    expectArraysEqual(await a.data(), ['a', 'b', 'c', 'd', 'e', 'f']);
  });

  it('tf.tensor3d() from encoded strings without dtype errors', async () => {
    // We do not want to infer 'string' when the user passes Uint8Array in order
    // to be forward compatible in the future when we add uint8 dtype.
    const bytes = encodeStrings([[['a'], ['b'], ['c']], [['d'], ['e'], ['f']]]);
    expect(() => tf.tensor3d(bytes as TensorLike3D)).toThrowError();
  });

  it('tf.tensor3d() from encoded strings, shape mismatch', () => {
    const bytes = encodeStrings([[['a'], ['b'], ['c']], [['d'], ['e'], ['f']]]);
    // Actual shape is [2, 3, 1].
    expect(() => tf.tensor3d(bytes as TensorLike3D, [3, 2, 1], 'string'))
        .toThrowError();
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

  it('tf.tensor3d() throw error with null input value', () => {
    expect(() => tf.tensor3d(null))
        .toThrowError(
            'The input to the tensor constructor ' +
            'must be a non-null value.');
  });

  it('tensor4d() from number[][][][]', async () => {
    const a = tf.tensor4d([[[[1]], [[2]]], [[[4]], [[5]]]], [2, 2, 1, 1]);
    expectArraysClose(await a.data(), [1, 2, 4, 5]);
  });

  it('tensor4d() from string[][][][]', async () => {
    const vals = [[[['a']], [['b']]], [[['c']], [['d']]]];
    const a = tf.tensor4d(vals, [2, 2, 1, 1]);
    expect(a.dtype).toBe('string');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    expectArraysEqual(await a.data(), ['a', 'b', 'c', 'd']);
  });

  it('tf.tensor4d() from encoded strings', async () => {
    const bytes = encodeStrings([[[['a']], [['b']]], [[['c']], [['d']]]]);
    const a = tf.tensor4d(bytes as TensorLike4D, [2, 2, 1, 1], 'string');
    expect(a.dtype).toBe('string');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    expectArraysEqual(await a.data(), ['a', 'b', 'c', 'd']);
  });

  it('tf.tensor4d() from encoded strings without dtype errors', async () => {
    // We do not want to infer 'string' when the user passes Uint8Array in order
    // to be forward compatible in the future when we add uint8 dtype.
    const bytes = encodeStrings([[[['a']], [['b']]], [[['c']], [['d']]]]);
    expect(() => tf.tensor4d(bytes as TensorLike4D)).toThrowError();
  });

  it('tf.tensor4d() from encoded strings, shape mismatch', () => {
    const bytes = encodeStrings([[[['a']], [['b']]], [[['c']], [['d']]]]);
    // Actual shape is [2, 2, 1. 1].
    expect(() => tf.tensor4d(bytes as TensorLike4D, [2, 1, 2, 1], 'string'))
        .toThrowError();
  });

  it('tensor4d() from string[][][][] infer shape', async () => {
    const vals = [[[['a']], [['b']]], [[['c']], [['d']]]];
    const a = tf.tensor4d(vals);
    expect(a.dtype).toBe('string');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    expectArraysEqual(await a.data(), ['a', 'b', 'c', 'd']);
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

  it('tf.tensor4d() throw error with null input value', () => {
    expect(() => tf.tensor4d(null))
        .toThrowError(
            'The input to the tensor constructor ' +
            'must be a non-null value.');
  });

  it('tf.tensor5d() throw error with null input value', () => {
    expect(() => tf.tensor5d(null))
        .toThrowError(
            'The input to the tensor constructor ' +
            'must be a non-null value.');
  });

  it('tf.tensor6d() throw error with null input value', () => {
    expect(() => tf.tensor6d(null))
        .toThrowError(
            'The input to the tensor constructor ' +
            'must be a non-null value.');
  });

  it('default dtype', async () => {
    const a = tf.scalar(3);
    expect(a.dtype).toBe('float32');
    expectArraysClose(await a.data(), 3);
  });

  it('float32 dtype', async () => {
    const a = tf.scalar(3, 'float32');
    expect(a.dtype).toBe('float32');
    expectArraysClose(await a.data(), 3);
  });

  it('int32 dtype', async () => {
    const a = tf.scalar(3, 'int32');
    expect(a.dtype).toBe('int32');
    expectArraysEqual(await a.data(), 3);
  });

  it('int32 dtype, 3.9 => 3, like numpy', async () => {
    const a = tf.scalar(3.9, 'int32');
    expect(a.dtype).toBe('int32');
    expectArraysEqual(await a.data(), 3);
  });

  it('int32 dtype, -3.9 => -3, like numpy', async () => {
    const a = tf.scalar(-3.9, 'int32');
    expect(a.dtype).toBe('int32');
    expectArraysEqual(await a.data(), -3);
  });

  it('bool dtype, 3 => true, like numpy', async () => {
    const a = tf.scalar(3, 'bool');
    expect(a.dtype).toBe('bool');
    expectArraysEqual(await a.data(), 1);
  });

  it('bool dtype, -2 => true, like numpy', async () => {
    const a = tf.scalar(-2, 'bool');
    expect(a.dtype).toBe('bool');
    expectArraysEqual(await a.data(), 1);
  });

  it('bool dtype, 0 => false, like numpy', async () => {
    const a = tf.scalar(0, 'bool');
    expect(a.dtype).toBe('bool');
    expectArraysEqual(await a.data(), 0);
  });

  it('bool dtype from boolean', async () => {
    const a = tf.scalar(false, 'bool');
    expectArraysEqual(await a.data(), 0);
    expect(a.dtype).toBe('bool');

    const b = tf.scalar(true, 'bool');
    expectArraysEqual(await a.data(), 0);
    expect(b.dtype).toBe('bool');
  });

  it('int32 dtype from boolean', async () => {
    const a = tf.scalar(true, 'int32');
    expectArraysEqual(await a.data(), 1);
    expect(a.dtype).toBe('int32');
  });

  it('default dtype from boolean', async () => {
    const a = tf.scalar(false);
    expectArraysEqual(await a.data(), 0);
    expect(a.dtype).toBe('bool');
  });

  it('default dtype', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    expectArraysClose(await a.data(), [1, 2, 3]);
  });

  it('float32 dtype', async () => {
    const a = tf.tensor1d([1, 2, 3], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    expectArraysClose(await a.data(), [1, 2, 3]);
  });

  it('int32 dtype', async () => {
    const a = tf.tensor1d([1, 2, 3], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3]);
    expectArraysEqual(await a.data(), [1, 2, 3]);
  });

  it('int32 dtype, non-ints get floored, like numpy', async () => {
    const a = tf.tensor1d([1.1, 2.5, 3.9], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3]);
    expectArraysEqual(await a.data(), [1, 2, 3]);
  });

  it('int32 dtype, negative non-ints get ceiled, like numpy', async () => {
    const a = tf.tensor1d([-1.1, -2.5, -3.9], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3]);
    expectArraysEqual(await a.data(), [-1, -2, -3]);
  });

  it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', async () => {
    const a = tf.tensor1d([1, -2, 0, 3], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([4]);
    expectArraysEqual(await a.data(), [1, 1, 0, 1]);
  });

  it('default dtype from boolean[]', async () => {
    const a = tf.tensor1d([false, false, true]);
    expect(a.dtype).toBe('bool');
    expectArraysClose(await a.data(), [0, 0, 1]);
  });

  it('default dtype from UInt8Array', async () => {
    const a = tf.tensor1d(new Uint8Array([1, 5, 2]));
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3]);
    expectArraysClose(await a.data(), [1, 5, 2]);
  });

  it('default dtype from Int32Array', async () => {
    const a = tf.tensor1d(new Int32Array([1, 5, 2]));
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3]);
    expectArraysClose(await a.data(), [1, 5, 2]);
  });

  it('tf.tensor() from Float32Array and number[]', async () => {
    const a = tf.tensor([
      new Float32Array([1, 2]), new Float32Array([3, 4]),
      new Float32Array([5, 6]), [7, 8]
    ]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([4, 2]);
    expectArraysClose(await a.data(), [1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it('tf.tensor() from Int32Array and number[]', async () => {
    const a = tf.tensor([
      new Int32Array([1, 2]), new Int32Array([3, 4]), new Int32Array([5, 6]),
      [7, 8]
    ]);
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([4, 2]);
    expectArraysClose(await a.data(), [1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it('tf.tensor() from mixed TypedArray', async () => {
    const a = tf.tensor([
      new Float32Array([1, 2]), new Int32Array([3, 4]), new Uint8Array([5, 6]),
      [7, 8]
    ]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([4, 2]);
    expectArraysClose(await a.data(), [1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it('tf.tensor() from TypedArrays which are themselves 3D', () => {
    // 2 tensors, each with shape 20x20x3, as flat Float32Arrays.
    const img1 = new Float32Array(20 * 20 * 3);
    const img2 = new Float32Array(20 * 20 * 3);
    const t = tf.tensor([img1, img2], [2, 20, 20, 3]);
    expect(t.dtype).toBe('float32');
    expect(t.shape).toEqual([2, 20, 20, 3]);
  });

  it('tf.tensor() from TypedArrays which are themselves 3D, wrong shape',
     () => {
       const img1 = new Float32Array(20 * 20 * 3);
       const img2 = new Float32Array(20 * 20 * 3);
       expect(() => tf.tensor([img1, img2], [3, 20, 20, 3])).toThrowError();
     });

  it('default dtype from ascii string', async () => {
    const a = tf.tensor('hello');
    expect(a.dtype).toBe('string');
    expect(a.shape).toEqual([]);
    expectArraysEqual(await a.data(), ['hello']);
  });

  it('default dtype from utf-8 string', async () => {
    const a = tf.tensor('даниел');
    expect(a.dtype).toBe('string');
    expect(a.shape).toEqual([]);
    expectArraysEqual(await a.data(), ['даниел']);
  });

  it('default dtype from empty string', async () => {
    const a = tf.tensor('');
    expect(a.dtype).toBe('string');
    expect(a.shape).toEqual([]);
    expectArraysEqual(await a.data(), ['']);
  });

  it('default dtype from unicode escaped string', async () => {
    const a = tf.tensor('\u0434\u0430\u043d\u0438\u0435\u043b');
    expect(a.dtype).toBe('string');
    expect(a.shape).toEqual([]);
    expectArraysEqual(await a.data(), ['даниел']);
  });

  it('default dtype from string[]', async () => {
    const a = tf.tensor(['a', 'b']);
    expect(a.dtype).toBe('string');
    expect(a.shape).toEqual([2]);
    expectArraysEqual(await a.data(), ['a', 'b']);
  });

  it('float32 dtype from boolean[]', async () => {
    const a = tf.tensor1d([false, false, true], 'float32');
    expect(a.dtype).toBe('float32');
    expectArraysClose(await a.data(), [0, 0, 1]);
  });

  it('int32 dtype from boolean[]', async () => {
    const a = tf.tensor1d([false, false, true], 'int32');
    expect(a.dtype).toBe('int32');
    expectArraysEqual(await a.data(), [0, 0, 1]);
  });

  it('bool dtype from boolean[]', async () => {
    const a = tf.tensor1d([false, false, true], 'bool');
    expect(a.dtype).toBe('bool');
    expectArraysEqual(await a.data(), [0, 0, 1]);
  });

  it('default dtype', async () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2]);
    expectArraysClose(await a.data(), [1, 2, 3, 4]);
  });

  it('float32 dtype', async () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2]);
    expectArraysClose(await a.data(), [1, 2, 3, 4]);
  });

  it('int32 dtype', async () => {
    const a = tf.tensor2d([[1, 2], [3, 4]], [2, 2], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2]);
    expectArraysEqual(await a.data(), [1, 2, 3, 4]);
  });

  it('int32 dtype, non-ints get floored, like numpy', async () => {
    const a = tf.tensor2d([1.1, 2.5, 3.9, 4.0], [2, 2], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2]);
    expectArraysEqual(await a.data(), [1, 2, 3, 4]);
  });

  it('int32 dtype, negative non-ints get ceiled, like numpy', async () => {
    const a = tf.tensor2d([-1.1, -2.5, -3.9, -4.0], [2, 2], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2]);
    expectArraysEqual(await a.data(), [-1, -2, -3, -4]);
  });

  it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', async () => {
    const a = tf.tensor2d([1, -2, 0, 3], [2, 2], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([2, 2]);
    expectArraysEqual(await a.data(), [1, 1, 0, 1]);
  });

  it('default dtype from boolean[]', async () => {
    const a = tf.tensor2d([[false, false], [true, false]], [2, 2]);
    expect(a.dtype).toBe('bool');
    expectArraysClose(await a.data(), [0, 0, 1, 0]);
  });

  it('float32 dtype from boolean[]', async () => {
    const a = tf.tensor2d([[false, false], [true, false]], [2, 2], 'float32');
    expect(a.dtype).toBe('float32');
    expectArraysEqual(await a.data(), [0, 0, 1, 0]);
  });

  it('int32 dtype from boolean[]', async () => {
    const a = tf.tensor2d([[false, false], [true, false]], [2, 2], 'int32');
    expect(a.dtype).toBe('int32');
    expectArraysEqual(await a.data(), [0, 0, 1, 0]);
  });

  it('bool dtype from boolean[]', async () => {
    const a = tf.tensor2d([[false, false], [true, false]], [2, 2], 'bool');
    expect(a.dtype).toBe('bool');
    expectArraysEqual(await a.data(), [0, 0, 1, 0]);
  });

  it('default dtype', async () => {
    const a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 1]);
    expectArraysClose(await a.data(), [1, 2, 3, 4]);
  });

  it('float32 dtype', async () => {
    const a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 1]);
    expectArraysClose(await a.data(), [1, 2, 3, 4]);
  });

  it('int32 dtype', async () => {
    const a = tf.tensor3d([[[1], [2]], [[3], [4]]], [2, 2, 1], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 1]);
    expectArraysEqual(await a.data(), [1, 2, 3, 4]);
  });

  it('int32 dtype, non-ints get floored, like numpy', async () => {
    const a = tf.tensor3d([1.1, 2.5, 3.9, 4.0], [2, 2, 1], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 1]);
    expectArraysEqual(await a.data(), [1, 2, 3, 4]);
  });

  it('int32 dtype, negative non-ints get ceiled, like numpy', async () => {
    const a = tf.tensor3d([-1.1, -2.5, -3.9, -4.0], [2, 2, 1], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 1]);
    expectArraysEqual(await a.data(), [-1, -2, -3, -4]);
  });

  it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', async () => {
    const a = tf.tensor3d([1, -2, 0, 3], [2, 2, 1], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([2, 2, 1]);
    expectArraysEqual(await a.data(), [1, 1, 0, 1]);
  });

  it('default dtype from boolean[]', async () => {
    const a = tf.tensor3d([[[false], [false]], [[true], [false]]], [2, 2, 1]);
    expect(a.dtype).toBe('bool');
    expectArraysClose(await a.data(), [0, 0, 1, 0]);
  });

  it('float32 dtype from boolean[]', async () => {
    const a = tf.tensor3d(
        [[[false], [false]], [[true], [false]]], [2, 2, 1], 'float32');
    expect(a.dtype).toBe('float32');
    expectArraysClose(await a.data(), [0, 0, 1, 0]);
  });

  it('int32 dtype from boolean[]', async () => {
    const a = tf.tensor3d(
        [[[false], [false]], [[true], [false]]], [2, 2, 1], 'int32');
    expect(a.dtype).toBe('int32');
    expectArraysEqual(await a.data(), [0, 0, 1, 0]);
  });

  it('bool dtype from boolean[]', async () => {
    const a =
        tf.tensor3d([[[false], [false]], [[true], [false]]], [2, 2, 1], 'bool');
    expect(a.dtype).toBe('bool');
    expectArraysEqual(await a.data(), [0, 0, 1, 0]);
  });

  it('default dtype', async () => {
    const a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    expectArraysClose(await a.data(), [1, 2, 3, 4]);
  });

  it('float32 dtype', async () => {
    const a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    expectArraysClose(await a.data(), [1, 2, 3, 4]);
  });

  it('int32 dtype', async () => {
    const a =
        tf.tensor4d([[[[1]], [[2]]], [[[3]], [[4]]]], [2, 2, 1, 1], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    expectArraysEqual(await a.data(), [1, 2, 3, 4]);
  });

  it('int32 dtype, non-ints get floored, like numpy', async () => {
    const a = tf.tensor4d([1.1, 2.5, 3.9, 4.0], [2, 2, 1, 1], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    expectArraysEqual(await a.data(), [1, 2, 3, 4]);
  });

  it('int32 dtype, negative non-ints get ceiled, like numpy', async () => {
    const a = tf.tensor4d([-1.1, -2.5, -3.9, -4.0], [2, 2, 1, 1], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    expectArraysEqual(await a.data(), [-1, -2, -3, -4]);
  });

  it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', async () => {
    const a = tf.tensor4d([1, -2, 0, 3], [2, 2, 1, 1], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    expectArraysEqual(await a.data(), [1, 1, 0, 1]);
  });

  it('default dtype from boolean[]', async () => {
    const a =
        tf.tensor4d([[[[false], [false]], [[true], [false]]]], [1, 2, 2, 1]);
    expect(a.dtype).toBe('bool');
    expectArraysClose(await a.data(), [0, 0, 1, 0]);
  });

  it('float32 dtype from boolean[]', async () => {
    const a = tf.tensor4d(
        [[[[false], [false]], [[true], [false]]]], [1, 2, 2, 1], 'float32');
    expect(a.dtype).toBe('float32');
    expectArraysClose(await a.data(), [0, 0, 1, 0]);
  });

  it('int32 dtype from boolean[]', async () => {
    const a = tf.tensor4d(
        [[[[false], [false]], [[true], [false]]]], [1, 2, 2, 1], 'int32');
    expect(a.dtype).toBe('int32');
    expectArraysEqual(await a.data(), [0, 0, 1, 0]);
  });

  it('bool dtype from boolean[]', async () => {
    const a = tf.tensor4d(
        [[[[false], [false]], [[true], [false]]]], [1, 2, 2, 1], 'bool');
    expect(a.dtype).toBe('bool');
    expectArraysEqual(await a.data(), [0, 0, 1, 0]);
  });

  it('Scalar default dtype', async () => {
    const a = tf.scalar(4);
    const b = a.reshape([1, 1]);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([1, 1]);
    expectArraysClose(await a.data(), await b.data());
  });

  it('Scalar float32 dtype', () => {
    const a = tf.scalar(4, 'float32');
    const b = a.reshape([1, 1]);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([1, 1]);
  });

  it('Scalar string dtype', () => {
    const a = tf.scalar('test', 'string');
    const b = a.reshape([1, 1]);
    expect(b.dtype).toBe('string');
    expect(b.shape).toEqual([1, 1]);
  });

  it('scalar from encoded string', async () => {
    const a = tf.scalar(encodeString('hello'), 'string');
    expect(a.dtype).toBe('string');
    expect(a.shape).toEqual([]);
    expectArraysEqual(await a.data(), ['hello']);
  });

  it('scalar from encoded string, but missing dtype', async () => {
    // We do not want to infer 'string' when the user passes Uint8Array in order
    // to be forward compatible in the future when we add uint8 dtype.
    expect(() => tf.scalar(encodeString('hello'))).toThrowError();
  });

  it('scalar from encoded string, but value is not uint8array', async () => {
    // tslint:disable-next-line:no-any
    expect(() => tf.scalar(new Float32Array([1, 2, 3]) as any)).toThrowError();
  });

  it('Scalar inferred dtype from bool', async () => {
    const a = tf.scalar(true);
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([]);
    expectArraysClose(await a.data(), [1]);
  });

  it('Scalar inferred dtype from string', async () => {
    const a = tf.scalar('hello');
    expect(a.dtype).toBe('string');
    expect(a.shape).toEqual([]);
    expectArraysEqual(await a.data(), ['hello']);
  });

  it('Scalar int32 dtype', () => {
    const a = tf.scalar(4, 'int32');
    const b = a.reshape([1, 1]);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([1, 1]);
  });

  it('Scalar bool dtype', async () => {
    const a = tf.scalar(4, 'bool');
    const b = a.reshape([1, 1, 1]);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([1, 1, 1]);
    expectArraysClose(await a.data(), await b.data());
  });

  it('Scalar complex64 dtype', async () => {
    const a = tf.complex(4, 5);
    const b = a.reshape([1, 1]);
    expectArraysClose(await a.data(), [4, 5]);
    expect(b.dtype).toBe('complex64');
    expect(b.shape).toEqual([1, 1]);
    expectArraysClose(await a.data(), await b.data());
  });

  it('Tensor1D default dtype', async () => {
    const a = tf.tensor1d([1, 2, 3, 4]);
    const b = a.reshape([2, 2]);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    expectArraysClose(await a.data(), await b.data());
  });

  it('Tensor1D inferred dtype from bools', async () => {
    const a = tf.tensor1d([true, false, false, true]);
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([4]);
    expectArraysClose(await a.data(), [1, 0, 0, 1]);
  });

  it('Tensor1D inferred dtype from strings', async () => {
    const a = tf.tensor1d(['a', 'b', 'c']);
    expect(a.dtype).toBe('string');
    expect(a.shape).toEqual([3]);
    expectArraysEqual(await a.data(), ['a', 'b', 'c']);
  });

  it('Tensor1D float32 dtype', () => {
    const a = tf.tensor1d([1, 2, 3, 4], 'float32');
    const b = a.reshape([2, 2]);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
  });

  it('Tensor1D int32 dtype', async () => {
    const a = tf.tensor1d([1, 2, 3, 4], 'int32');
    const b = a.reshape([2, 2]);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2]);
    expectArraysClose(await a.data(), await b.data());
  });

  it('Tensor1D complex64 dtype', async () => {
    const a = tf.complex([1, 3, 5, 7], [2, 4, 6, 8]);
    const b = a.reshape([2, 2]);
    expect(b.dtype).toBe('complex64');
    expect(b.shape).toEqual([2, 2]);
    expectArraysClose(await a.data(), await b.data());
  });

  it('Tensor2D default dtype', async () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = a.reshape([6]);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([6]);
    expectArraysClose(await a.data(), await b.data());
  });

  it('Tensor2D float32 dtype', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3], 'float32');
    const b = a.reshape([6]);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([6]);
  });

  it('Tensor2D int32 dtype', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3], 'int32');
    const b = a.reshape([6]);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([6]);
  });

  it('Tensor2D bool dtype', async () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3], 'bool');
    const b = a.reshape([6]);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([6]);
    expectArraysClose(await a.data(), await b.data());
  });

  it('Tensor2D complex64 dtype', async () => {
    const a = tf.complex([[1, 3, 5], [7, 9, 11]], [[2, 4, 6], [8, 10, 12]]);
    const b = a.reshape([6]);
    expect(b.dtype).toBe('complex64');
    expect(b.shape).toEqual([6]);
    expectArraysClose(await a.data(), await b.data());
  });

  it('Tensor3D default dtype', async () => {
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6], [2, 3, 1]);
    const b = a.reshape([6]);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([6]);
    expectArraysClose(await a.data(), await b.data());
  });

  it('Tensor3D float32 dtype', () => {
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6], [2, 3, 1], 'float32');
    const b = a.reshape([6]);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([6]);
  });

  it('Tensor3D int32 dtype', () => {
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6], [2, 3, 1], 'int32');
    const b = a.reshape([6]);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([6]);
  });

  it('Tensor3D bool dtype', async () => {
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6], [2, 3, 1], 'bool');
    const b = a.reshape([6]);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([6]);
    expectArraysClose(await a.data(), await b.data());
  });

  it('Tensor3D complex64 dtype', async () => {
    const a = tf.complex(
        [[[1], [3], [5]], [[7], [9], [11]]],
        [[[2], [4], [6]], [[8], [10], [12]]]);
    const b = a.reshape([6]);
    expect(b.dtype).toBe('complex64');
    expect(b.shape).toEqual([6]);
    expectArraysClose(await a.data(), await b.data());
  });

  it('Tensor4D default dtype', async () => {
    const a = tf.tensor4d([1, 2, 3, 4, 5, 6], [2, 3, 1, 1]);
    const b = a.reshape([2, 3]);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 3]);
    expectArraysClose(await a.data(), await b.data());
  });

  it('Tensor4D float32 dtype', () => {
    const a = tf.tensor4d([1, 2, 3, 4, 5, 6], [2, 3, 1, 1], 'float32');
    const b = a.reshape([2, 3]);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 3]);
  });

  it('Tensor4D int32 dtype', async () => {
    const a = tf.tensor4d([1, 2, 3, 4, 5, 6], [2, 3, 1, 1], 'int32');
    const b = a.reshape([3, 2]);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([3, 2]);
    expectArraysClose(await a.data(), await b.data());
  });

  it('Tensor4D complex64 dtype', async () => {
    const a = tf.complex(
        [[[[1]], [[3]], [[5]]], [[[7]], [[9]], [[11]]]],
        [[[[2]], [[4]], [[6]]], [[[8]], [[10]], [[12]]]]);
    const b = a.reshape([3, 2]);
    expect(b.dtype).toBe('complex64');
    expect(b.shape).toEqual([3, 2]);
    expectArraysClose(await a.data(), await b.data());
  });

  it('Tensor4D bool dtype', () => {
    const a = tf.tensor4d([1, 2, 3, 4, 5, 6], [2, 3, 1, 1], 'bool');
    const b = a.reshape([3, 2]);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([3, 2]);
  });

  it('.data() with casting, string tensor', async () => {
    const a = tf.tensor(['a', 'b']);
    const data: string[] = await a.data<'string'>();
    expect(data).toEqual(['a', 'b']);
  });

  it('reshape is functional', async () => {
    const a = tf.scalar(2.4);
    const b = a.reshape([]);
    expect(a.id).not.toBe(b.id);
    b.dispose();
    expectArraysClose(await a.data(), [2.4]);
  });

  it('reshape a string tensor', async () => {
    const a = tf.tensor(['a', 'b']);
    const b = a.reshape([2, 1, 1]);
    expect(b.dtype).toBe('string');
    expect(b.shape).toEqual([2, 1, 1]);
    expectArraysEqual(await b.data(), ['a', 'b']);
  });

  it('reshape throws when passed a non-tensor', () => {
    // tslint:disable-next-line:no-any
    expect(() => tf.reshape({} as any, []))
        .toThrowError(/Argument 'x' passed to 'reshape' must be a Tensor/);
  });

  it('reshape accepts a tensor-like object', async () => {
    const res = tf.reshape([[1, 2, 3], [4, 5, 6]], [3, 2]);
    expect(res.dtype).toBe('float32');
    expect(res.shape).toEqual([3, 2]);
    expectArraysClose(await res.data(), [1, 2, 3, 4, 5, 6]);
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

  it('cast float32 -> int32. async download', async () => {
    const a = tf.tensor1d([1, 2]);
    const aInt = a.cast('int32');
    expect(aInt.dtype).toEqual('int32');

    const asyncData = await aInt.data();
    expect(asyncData instanceof Int32Array).toEqual(true);
  });

  it('cast float32 -> int32. queued async download', async () => {
    const a = tf.tensor1d([1, 2]);
    const aInt = a.cast('int32');
    expect(aInt.dtype).toEqual('int32');

    const [first, second] = await Promise.all([aInt.data(), aInt.data()]);
    expect(first instanceof Int32Array).toEqual(true);
    expect(second instanceof Int32Array).toEqual(true);
  });

  it('cast float32 -> int32. sync download', async () => {
    const a = tf.tensor1d([1, 2]).cast('int32');
    expect(a.dtype).toEqual('int32');

    const data = await a.data();
    expect(data instanceof Int32Array).toEqual(true);
  });

  it('cast float32 -> float32', () => {
    const a = tf.tensor1d([1.0, 2.0]);
    expect(a.cast('float32').dtype).toEqual('float32');
  });

  it('cast complex64 -> float32', async () => {
    const a = tf.complex([1.0, 2.0], [3.0, 4.0]);
    const result = a.cast('float32');

    expect(result.dtype).toEqual('float32');
    expectArraysClose(await result.data(), [1.0, 2.0]);
  });

  it('cast complex64 -> int32', async () => {
    const a = tf.complex([1.0, 2.0], [3.0, 4.0]);
    const result = a.cast('int32');

    expect(result.dtype).toEqual('int32');
    expectArraysEqual(await result.data(), [1, 2]);
  });

  it('cast complex64 -> bool', async () => {
    const a = tf.complex([1.0, 0.0], [1.0, 1.0]);
    const result = a.cast('bool');

    expect(result.dtype).toEqual('bool');
    expectArraysEqual(await result.data(), [true, false]);
  });

  it('cast throws when passed a non-tensor', () => {
    expect(() => tf.cast({} as tf.Tensor, 'float32'))
        .toThrowError(/Argument 'x' passed to 'cast' must be a Tensor/);
  });

  it('cast accepts a tensor-like object', async () => {
    const a = [1.0, 2.0];
    const res = tf.cast(a, 'int32');
    expect(res.dtype).toEqual('int32');
    expectArraysClose(await res.data(), [1, 2]);
  });

  it('cast string -> !string throws error', () => {
    const a = ['a', 'b'];
    expect(() => tf.cast(a, 'int32')).toThrowError();
    expect(() => tf.cast(a, 'float32')).toThrowError();
    expect(() => tf.cast(a, 'bool')).toThrowError();
    expect(() => tf.cast(a, 'complex64')).toThrowError();
  });

  it('cast !string -> string throws error', () => {
    expect(() => tf.cast(tf.tensor(1, [], 'float32'), 'string')).toThrowError();
    expect(() => tf.cast(tf.tensor(1, [], 'int32'), 'string')).toThrowError();
    expect(() => tf.cast(tf.tensor(1, [], 'bool'), 'string')).toThrowError();
    expect(() => tf.cast(tf.tensor(1, [], 'complex64'), 'string'))
        .toThrowError();
  });

  it('scalar bool -> int32', async () => {
    const a = tf.scalar(true, 'bool').toInt();
    expect(a.dtype).toBe('int32');
    expectArraysEqual(await a.data(), 1);
  });

  it('Tensor1D float32 -> int32', async () => {
    const a = tf.tensor1d([1.1, 3.9, -2.9, 0]).toInt();
    expect(a.dtype).toBe('int32');
    expectArraysEqual(await a.data(), [1, 3, -2, 0]);
  });

  it('Tensor2D float32 -> bool', async () => {
    const a = tf.tensor2d([1.1, 3.9, -2.9, 0], [2, 2]).asType('bool');
    expect(a.dtype).toBe('bool');
    expectArraysEqual(await a.data(), [1, 1, 1, 0]);
  });

  it('Tensor2D int32 -> bool', async () => {
    const a = tf.tensor2d([1, 3, 0, -1], [2, 2], 'int32').toBool();
    expect(a.dtype).toBe('bool');
    expectArraysEqual(await a.data(), [1, 1, 0, 1]);
  });

  it('Tensor3D bool -> float32', async () => {
    const a =
        tf.tensor3d([true, false, false, true], [2, 2, 1], 'bool').toFloat();
    expect(a.dtype).toBe('float32');
    expectArraysClose(await a.data(), [1, 0, 0, 1]);
  });

  it('bool CPU -> GPU -> CPU', async () => {
    const a = tf.tensor1d([1, 2, 0, 0, 5], 'bool');
    expectArraysEqual(await a.data(), [1, 1, 0, 0, 1]);
  });

  it('int32 CPU -> GPU -> CPU', async () => {
    const a = tf.tensor1d([1, 2, 0, 0, 5], 'int32');
    expectArraysEqual(await a.data(), [1, 2, 0, 0, 5]);
  });

  it('asType is functional', async () => {
    const a = tf.scalar(2.4, 'float32');
    const b = a.toFloat();
    expect(a.id).not.toBe(b.id);
    b.dispose();
    expectArraysClose(await a.data(), [2.4]);
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

  it('squeeze with negative axis', () => {
    const a = tf.tensor3d([4, 2, 1], [3, 1, 1], 'bool');
    const b = a.squeeze([-1]);
    expect(b.shape).toEqual([3, 1]);
  });

  it('squeeze with multiple negative axis', () => {
    const a = tf.tensor3d([4, 2, 1], [3, 1, 1], 'bool');
    const b = a.squeeze([-1, -2]);
    expect(b.shape).toEqual([3]);
  });

  it('squeeze wrong axis', () => {
    const a = tf.tensor3d([4, 2, 1], [3, 1, 1], 'bool');
    expect(() => a.squeeze([0, 1])).toThrowError();
  });

  it('squeeze wrong negative axis', () => {
    const a = tf.tensor3d([4, 2, 1], [3, 1, 1], 'bool');
    expect(() => a.squeeze([-3, -2])).toThrowError();
  });

  it('squeeze axis out of range', () => {
    const a = tf.tensor3d([4, 2, 1], [3, 1, 1], 'bool');
    expect(() => a.squeeze([10, 11])).toThrowError();
  });

  it('squeeze negative axis out of range', () => {
    const a = tf.tensor3d([4, 2, 1], [3, 1, 1], 'bool');
    expect(() => a.squeeze([-13, -12])).toThrowError();
  });

  it('squeeze throws when passed a non-tensor', () => {
    expect(() => tf.squeeze({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'squeeze' must be a Tensor/);
  });

  it('squeeze accepts a tensor-like object', async () => {
    const res = tf.squeeze([[[4]], [[2]], [[1]]] /* shape is [3, 1, 1] */);
    expect(res.shape).toEqual([3]);
    expectArraysClose(await res.data(), [4, 2, 1]);
  });

  it('squeeze a zero-sized tensor', () => {
    const a = tf.tensor3d([], [0, 1, 0]);
    const res = tf.squeeze(a);
    expect(res.shape).toEqual([0, 0]);
  });

  it('squeeze a complex64 tensor', async () => {
    const a = tf.complex([[4], [1], [5]], [[2], [3], [6]]);
    const b = a.squeeze();
    expect(b.shape).toEqual([3]);
    expectArraysClose(await b.data(), [4, 2, 1, 3, 5, 6]);
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

  it('throws when passed non-integer shape', () => {
    const msg = 'Tensor must have a shape comprised of positive ' +
        'integers but got shape [2,2.2].';
    expect(() => tf.tensor([1, 2, 3, 4], [2, 2.2])).toThrowError(msg);
  });

  it('throws when passed negative shape', () => {
    const msg = 'Tensor must have a shape comprised of positive ' +
        'integers but got shape [2,-2].';
    expect(() => tf.tensor([1, 2, 3, 4], [2, -2])).toThrowError(msg);
  });

  it('ones with complex type', async () => {
    // Imaginary part should be zero.
    const a = tf.ones([2, 2], 'complex64');
    expectArraysClose(await a.data(), [1, 0, 1, 0, 1, 0, 1, 0]);
  });
});

describeWithFlags('tensor debug mode', ALL_ENVS, () => {
  beforeAll(() => {
    tf.ENV.set('DEBUG', true);
  });

  it('tf.tensor() from TypedArray + number[] fails due to wrong shape', () => {
    expect(() => tf.tensor([
      new Float32Array([1, 2]),
      new Float32Array([3, 4]),
      new Float32Array([5, 6]),
      // Should be of length 4
      [7, 8, 9, 10],
    ]))
        .toThrowError(
            /Element arr\[3\] should have 2 elements, but has 4 elements/);
  });
});

describeWithFlags('tensor dataSync', SYNC_BACKEND_ENVS, () => {
  it('.dataSync() with casting, string tensor', () => {
    const a = tf.tensor(['a', 'b']);
    const data: string[] = a.dataSync<'string'>();
    expect(data).toEqual(['a', 'b']);
  });
});

describeWithFlags('tensor.toString', SYNC_BACKEND_ENVS, () => {
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

  it('string scalar verbose', () => {
    const verbose = true;
    const str = tf.scalar('test').toString(verbose);
    expect(str).toEqual(
        'Tensor\n' +
        '  dtype: string\n' +
        '  rank: 0\n' +
        '  shape: []\n' +
        '  values:\n' +
        '    test');
  });

  it('bool scalar verbose', () => {
    const verbose = true;
    const str = tf.scalar(true).toString(verbose);
    expect(str).toEqual(
        'Tensor\n' +
        '  dtype: bool\n' +
        '  rank: 0\n' +
        '  shape: []\n' +
        '  values:\n' +
        '    true');
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

  it('1d string tensor verbose', () => {
    const verbose = true;
    const str = tf.tensor(['a', 'bb', 'ccc']).toString(verbose);
    expect(str).toEqual(
        'Tensor\n' +
        '  dtype: string\n' +
        '  rank: 1\n' +
        '  shape: [3]\n' +
        '  values:\n' +
        '    [\'a\', \'bb\', \'ccc\']');
  });

  it('1d bool tensor verbose', () => {
    const verbose = true;
    const str = tf.tensor([true, false, true]).toString(verbose);
    expect(str).toEqual(
        'Tensor\n' +
        '  dtype: bool\n' +
        '  rank: 1\n' +
        '  shape: [3]\n' +
        '  values:\n' +
        '    [true, false, true]');
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

  it('2d string tensor verbose', () => {
    const verbose = true;
    const vals = [
      ['a', 'bb', 'ccc'],
      ['d', 'e', 'f'],
      ['g', 'h', 'i'],
    ];
    const str = tf.tensor(vals).toString(verbose);
    expect(str).toEqual(
        'Tensor\n' +
        '  dtype: string\n' +
        '  rank: 2\n' +
        '  shape: [3,3]\n' +
        '  values:\n' +
        '    [[\'a\', \'bb\', \'ccc\'],\n' +
        '     [\'d\', \'e\' , \'f\'  ],\n' +
        '     [\'g\', \'h\' , \'i\'  ]]');
  });

  it('2d bool tensor verbose', () => {
    const verbose = true;
    const str = tf.zeros([3, 3], 'bool').toString(verbose);
    expect(str).toEqual(
        'Tensor\n' +
        '  dtype: bool\n' +
        '  rank: 2\n' +
        '  shape: [3,3]\n' +
        '  values:\n' +
        '    [[false, false, false],\n' +
        '     [false, false, false],\n' +
        '     [false, false, false]]');
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

  it('3d string tensor verbose', () => {
    const verbose = true;
    const vals = [
      [['a', 'bb'], ['ccc', 'dddd']],
      [['e', 'ff'], ['ggg', 'hhhh']],
      [['i', 'jj'], ['kkk', 'llll']],
    ];
    const str = tf.tensor(vals).toString(verbose);
    expect(str).toEqual(
        'Tensor\n' +
        '  dtype: string\n' +
        '  rank: 3\n' +
        '  shape: [3,2,2]\n' +
        '  values:\n' +
        '    [[[\'a\'  , \'bb\'  ],\n' +
        '      [\'ccc\', \'dddd\']],\n\n' +
        '     [[\'e\'  , \'ff\'  ],\n' +
        '      [\'ggg\', \'hhhh\']],\n\n' +
        '     [[\'i\'  , \'jj\'  ],\n' +
        '      [\'kkk\', \'llll\']]]');
  });

  it('3d bool tensor verbose', () => {
    const verbose = true;
    const str = tf.ones([3, 3, 2], 'bool').toString(verbose);
    expect(str).toEqual(
        'Tensor\n' +
        '  dtype: bool\n' +
        '  rank: 3\n' +
        '  shape: [3,3,2]\n' +
        '  values:\n' +
        '    [[[true, true],\n' +
        '      [true, true],\n' +
        '      [true, true]],\n\n' +
        '     [[true, true],\n' +
        '      [true, true],\n' +
        '      [true, true]],\n\n' +
        '     [[true, true],\n' +
        '      [true, true],\n' +
        '      [true, true]]]');
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

  it('1d long string tensor verbose', () => {
    const verbose = true;
    const str = tf.fill([100], 'hi').toString(verbose);
    expect(str).toEqual(
        'Tensor\n' +
        '  dtype: string\n' +
        '  rank: 1\n' +
        '  shape: [100]\n' +
        '  values:\n' +
        '    [\'hi\', \'hi\', \'hi\', ..., \'hi\', \'hi\', \'hi\']');
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

  it('2d long string tensor verbose', () => {
    const verbose = true;
    const str = tf.fill([100, 100], 'a').toString(verbose);
    expect(str).toEqual(
        'Tensor\n' +
        '  dtype: string\n' +
        '  rank: 2\n' +
        '  shape: [100,100]\n' +
        '  values:\n' +
        '    [[\'a\', \'a\', \'a\', ..., \'a\', \'a\', \'a\'],\n' +
        '     [\'a\', \'a\', \'a\', ..., \'a\', \'a\', \'a\'],\n' +
        '     [\'a\', \'a\', \'a\', ..., \'a\', \'a\', \'a\'],\n' +
        '     ...,\n' +
        '     [\'a\', \'a\', \'a\', ..., \'a\', \'a\', \'a\'],\n' +
        '     [\'a\', \'a\', \'a\', ..., \'a\', \'a\', \'a\'],\n' +
        '     [\'a\', \'a\', \'a\', ..., \'a\', \'a\', \'a\']]');
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

  it('2d string tensor with padding verbose', () => {
    const verbose = true;
    const str = tf.tensor([
                    ['abcdef', 'a', 'abcdef'],
                    ['abcdef', 'abcdef', 'abc'],
                    ['abcd', 'abcdef', 'abcdef'],
                  ]).toString(verbose);
    expect(str).toEqual(
        'Tensor\n' +
        '  dtype: string\n' +
        '  rank: 2\n' +
        '  shape: [3,3]\n' +
        '  values:\n' +
        '    [[\'abcdef\', \'a\'     , \'abcdef\'],\n' +
        '     [\'abcdef\', \'abcdef\', \'abc\'   ],\n' +
        '     [\'abcd\'  , \'abcdef\', \'abcdef\']]');
  });

  it('scalar', () => {
    const str = tf.scalar(5).toString();
    expect(str).toEqual(
        'Tensor\n' +
        '    5');
  });

  it('scalar string', () => {
    const str = tf.scalar('hello').toString();
    expect(str).toEqual(
        'Tensor\n' +
        '    hello');
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

  it('scalar complex64 verbose', () => {
    const verbose = true;
    const str = tf.complex(5, 6).toString(verbose);
    expect(str).toEqual(
        'Tensor\n' +
        '  dtype: complex64\n' +
        '  rank: 0\n' +
        '  shape: []\n' +
        '  values:\n' +
        '    5 + 6j');
  });

  it('1d complex64 tensor verbose', () => {
    const verbose = true;
    const str = tf.complex([3, 5], [4, 6]).toString(verbose);
    expect(str).toEqual(
        'Tensor\n' +
        '  dtype: complex64\n' +
        '  rank: 1\n' +
        '  shape: [2]\n' +
        '  values:\n' +
        '    [3 + 4j, 5 + 6j]');
  });

  it('2d complex64 tensor verbose', () => {
    const verbose = true;
    const str = tf.zeros([3, 3], 'complex64').toString(verbose);
    expect(str).toEqual(
        'Tensor\n' +
        '  dtype: complex64\n' +
        '  rank: 2\n' +
        '  shape: [3,3]\n' +
        '  values:\n' +
        '    [[0 + 0j, 0 + 0j, 0 + 0j],\n' +
        '     [0 + 0j, 0 + 0j, 0 + 0j],\n' +
        '     [0 + 0j, 0 + 0j, 0 + 0j]]');
  });

  it('3d complex64 tensor verbose', () => {
    const verbose = true;
    const str = tf.zeros([3, 3, 2], 'complex64').toString(verbose);
    expect(str).toEqual(
        'Tensor\n' +
        '  dtype: complex64\n' +
        '  rank: 3\n' +
        '  shape: [3,3,2]\n' +
        '  values:\n' +
        '    [[[0 + 0j, 0 + 0j],\n' +
        '      [0 + 0j, 0 + 0j],\n' +
        '      [0 + 0j, 0 + 0j]],\n\n' +
        '     [[0 + 0j, 0 + 0j],\n' +
        '      [0 + 0j, 0 + 0j],\n' +
        '      [0 + 0j, 0 + 0j]],\n\n' +
        '     [[0 + 0j, 0 + 0j],\n' +
        '      [0 + 0j, 0 + 0j],\n' +
        '      [0 + 0j, 0 + 0j]]]');
  });

  it('1d long tensor verbose', () => {
    const verbose = true;
    const str = tf.zeros([100], 'complex64').toString(verbose);
    expect(str).toEqual(
        'Tensor\n' +
        '  dtype: complex64\n' +
        '  rank: 1\n' +
        '  shape: [100]\n' +
        '  values:\n' +
        '    [0 + 0j, 0 + 0j, 0 + 0j, ..., 0 + 0j, 0 + 0j, 0 + 0j]');
  });

  it('2d long tensor verbose', () => {
    const verbose = true;
    const str = tf.zeros([100, 100], 'complex64').toString(verbose);
    expect(str).toEqual(
        'Tensor\n' +
        '  dtype: complex64\n' +
        '  rank: 2\n' +
        '  shape: [100,100]\n' +
        '  values:\n' +
        '    [[0 + 0j, 0 + 0j, 0 + 0j, ..., 0 + 0j, 0 + 0j, 0 + 0j],\n' +
        '     [0 + 0j, 0 + 0j, 0 + 0j, ..., 0 + 0j, 0 + 0j, 0 + 0j],\n' +
        '     [0 + 0j, 0 + 0j, 0 + 0j, ..., 0 + 0j, 0 + 0j, 0 + 0j],\n' +
        '     ...,\n' +
        '     [0 + 0j, 0 + 0j, 0 + 0j, ..., 0 + 0j, 0 + 0j, 0 + 0j],\n' +
        '     [0 + 0j, 0 + 0j, 0 + 0j, ..., 0 + 0j, 0 + 0j, 0 + 0j],\n' +
        '     [0 + 0j, 0 + 0j, 0 + 0j, ..., 0 + 0j, 0 + 0j, 0 + 0j]]');
  });

  it('2d complex64 with padding to align columns verbose', () => {
    const verbose = true;

    const str = tf.complex(
                      [
                        [0.8597712, 3, 0.2740789], [0.6696132, 0.4825962, 2.75],
                        [1.991, 0.0640865, 0.2983858]
                      ],
                      [[1, 1.0102332, 3], [2, 5, 2.34424], [1.23, 2, 0.123]])
                    .toString(verbose);
    expect(str).toEqual(
        'Tensor\n' +
        '  dtype: complex64\n' +
        '  rank: 2\n' +
        '  shape: [3,3]\n' +
        '  values:\n' +
        '    [[0.8597712 + 1j   , 3 + 1.0102332j, 0.2740789 + 3j    ],\n' +
        '     [0.6696132 + 2j   , 0.4825962 + 5j, 2.75 + 2.34424j   ],\n' +
        '     [1.9910001 + 1.23j, 0.0640865 + 2j, 0.2983858 + 0.123j]]');
  });

  it('scalar complex64', () => {
    const str = tf.complex(5, 4).toString();
    expect(str).toEqual(
        'Tensor\n' +
        '    5 + 4j');
  });

  it('1d complex64 tensor', () => {
    const str = tf.zeros([4], 'complex64').toString();
    expect(str).toEqual(
        'Tensor\n' +
        '    [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j]');
  });

  it('2d complex64 tensor', () => {
    const str = tf.zeros([3, 3], 'complex64').toString();
    expect(str).toEqual(
        'Tensor\n' +
        '    [[0 + 0j, 0 + 0j, 0 + 0j],\n' +
        '     [0 + 0j, 0 + 0j, 0 + 0j],\n' +
        '     [0 + 0j, 0 + 0j, 0 + 0j]]');
  });

  it('3d complex64 tensor', () => {
    const str = tf.zeros([3, 3, 2], 'complex64').toString();
    expect(str).toEqual(
        'Tensor\n' +
        '    [[[0 + 0j, 0 + 0j],\n' +
        '      [0 + 0j, 0 + 0j],\n' +
        '      [0 + 0j, 0 + 0j]],\n\n' +
        '     [[0 + 0j, 0 + 0j],\n' +
        '      [0 + 0j, 0 + 0j],\n' +
        '      [0 + 0j, 0 + 0j]],\n\n' +
        '     [[0 + 0j, 0 + 0j],\n' +
        '      [0 + 0j, 0 + 0j],\n' +
        '      [0 + 0j, 0 + 0j]]]');
  });

  it('1d long complex64 tensor', () => {
    const str = tf.zeros([100], 'complex64').toString();
    expect(str).toEqual(
        'Tensor\n' +
        '    [0 + 0j, 0 + 0j, 0 + 0j, ..., 0 + 0j, 0 + 0j, 0 + 0j]');
  });

  it('2d long complex64 tensor', () => {
    const str = tf.zeros([100, 100], 'complex64').toString();
    expect(str).toEqual(
        'Tensor\n' +
        '    [[0 + 0j, 0 + 0j, 0 + 0j, ..., 0 + 0j, 0 + 0j, 0 + 0j],\n' +
        '     [0 + 0j, 0 + 0j, 0 + 0j, ..., 0 + 0j, 0 + 0j, 0 + 0j],\n' +
        '     [0 + 0j, 0 + 0j, 0 + 0j, ..., 0 + 0j, 0 + 0j, 0 + 0j],\n' +
        '     ...,\n' +
        '     [0 + 0j, 0 + 0j, 0 + 0j, ..., 0 + 0j, 0 + 0j, 0 + 0j],\n' +
        '     [0 + 0j, 0 + 0j, 0 + 0j, ..., 0 + 0j, 0 + 0j, 0 + 0j],\n' +
        '     [0 + 0j, 0 + 0j, 0 + 0j, ..., 0 + 0j, 0 + 0j, 0 + 0j]]');
  });

  it('2d complex64 with padding to align columns', () => {
    const str = tf.complex(
                      [
                        [0.8597712, 3, 0.2740789], [0.6696132, 0.4825962, 2.75],
                        [1.991, 0.0640865, 0.2983858]
                      ],
                      [[1, 1.0102332, 3], [2, 5, 2.34424], [1.23, 2, 0.123]])
                    .toString();
    expect(str).toEqual(
        'Tensor\n' +
        '    [[0.8597712 + 1j   , 3 + 1.0102332j, 0.2740789 + 3j    ],\n' +
        '     [0.6696132 + 2j   , 0.4825962 + 5j, 2.75 + 2.34424j   ],\n' +
        '     [1.9910001 + 1.23j, 0.0640865 + 2j, 0.2983858 + 0.123j]]');
  });
});

describeWithFlags('tensor grad', ALL_ENVS, () => {
  it('grad with second derivative', async () => {
    // f(x) = x ^ 3
    const f = (x: Tensor) => x.pow(tf.scalar(3, 'int32'));
    // f'(x) = 3x ^ 2
    const g = tf.grad(f);
    // f''(x) = 6x
    const gg = tf.grad(g);
    const x = tf.tensor1d([2, 3]);
    const data = gg(x);
    expectArraysClose(await data.data(), [12, 18]);
  });
});

describeWithFlags('tensor.data', ALL_ENVS, () => {
  it('interleaving .data() and .dataSync()', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    const b = tf.tensor1d([4, 5, 6]);

    const ra = a.square();
    const rb = b.square();

    expectArraysClose(await a.data(), [1, 2, 3]);
    expectArraysClose(await b.data(), [4, 5, 6]);
    expectArraysClose(await rb.data(), [16, 25, 36]);
    expectArraysClose(await ra.data(), [1, 4, 9]);
  });

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

describeWithFlags('x instanceof Tensor', ALL_ENVS, () => {
  it('x: Tensor', () => {
    const t = tf.scalar(1);
    expect(t instanceof Tensor).toBe(true);
  });

  it('x: Tensor-like', () => {
    const t = {shape: [2], dtype: 'float32', dataId: {}};
    expect(t instanceof Tensor).toBe(true);
  });

  it('x: other object, fails', () => {
    const t = {something: 'else'};
    expect(t instanceof Tensor).toBe(false);
  });

  it('x: undefined or null, fails', () => {
    // tslint:disable-next-line:no-any
    expect((undefined as any) instanceof Tensor).toBe(false);
    // tslint:disable-next-line:no-any
    expect((null as any) instanceof Tensor).toBe(false);
  });
});

describeWithFlags('tensor with 0 in shape', ALL_ENVS, () => {
  it('1d of shape [0]', async () => {
    const a = tf.tensor1d([]);
    expect(a.dtype).toBe('float32');
    expect(a.rank).toBe(1);
    expect(a.shape).toEqual([0]);
    expectArraysEqual(await a.data(), []);
  });

  it('1d string tensor of shape [0]', async () => {
    const a = tf.tensor1d([], 'string');
    expect(a.dtype).toBe('string');
    expect(a.rank).toBe(1);
    expect(a.shape).toEqual([0]);
    expectArraysEqual(await a.data(), []);
  });

  it('2d of shape [0, 5]', async () => {
    const a = tf.tensor2d([], [0, 5]);
    expect(a.dtype).toBe('float32');
    expect(a.rank).toBe(2);
    expect(a.shape).toEqual([0, 5]);
    expectArraysEqual(await a.data(), []);
  });

  it('2d string tensor of shape [0, 5]', async () => {
    const a = tf.tensor2d([], [0, 5], 'string');
    expect(a.dtype).toBe('string');
    expect(a.rank).toBe(2);
    expect(a.shape).toEqual([0, 5]);
    expectArraysEqual(await a.data(), []);
  });

  it('2d throws when values are not empty', () => {
    const values = [1, 2, 3, 4];
    expect(() => tf.tensor2d(values, [0, 5], 'float32'))
        .toThrowError(
            'Based on the provided shape, [0,5], the ' +
            'tensor should have 0 values but has 4');
  });

  it('3d of shape [0, 3, 0]', async () => {
    const a = tf.tensor3d([], [0, 3, 0]);
    expect(a.dtype).toBe('float32');
    expect(a.rank).toBe(3);
    expect(a.shape).toEqual([0, 3, 0]);
    expectArraysEqual(await a.data(), []);
  });

  it('3d throws when values are not empty', () => {
    const values = [1, 2, 3];
    expect(() => tf.tensor3d(values, [0, 3, 0], 'float32'))
        .toThrowError(
            'Based on the provided shape, [0,3,0], the ' +
            'tensor should have 0 values but has 3');
  });

  it('4d of shape [1, 3, 0, 5]', async () => {
    const a = tf.tensor4d([], [1, 3, 0, 5]);
    expect(a.dtype).toBe('float32');
    expect(a.rank).toBe(4);
    expect(a.shape).toEqual([1, 3, 0, 5]);
    expectArraysEqual(await a.data(), []);
  });

  it('4d throws when values are not empty', () => {
    const values = [1, 2, 3];
    expect(() => tf.tensor4d(values, [1, 3, 0, 5], 'float32'))
        .toThrowError(
            'Based on the provided shape, [1,3,0,5], the ' +
            'tensor should have 0 values but has 3');
  });

  it('complex64 with 0 in shape', async () => {
    const areal = tf.tensor2d([], [0, 5]);
    const breal = tf.tensor2d([], [0, 5]);
    const a = tf.complex(areal, breal);
    expect(a.dtype).toBe('complex64');
    expect(a.rank).toBe(2);
    expect(a.shape).toEqual([0, 5]);
    expectArraysEqual(await a.data(), []);
  });
});

describeWithFlags('tensor.bytes()', ALL_ENVS, () => {
  /** Helper method to get the bytes from a typed array. */
  function getBytes(a: TypedArray): Uint8Array {
    return new Uint8Array(a.buffer);
  }

  it('float32 tensor', async () => {
    const a = tf.tensor([1.1, 3.2, 7], [3], 'float32');
    expect(await a.bytes()).toEqual(getBytes(new Float32Array([1.1, 3.2, 7])));
  });

  it('int32 tensor', async () => {
    const a = tf.tensor([1.1, 3.2, 7], [3], 'int32');
    expect(await a.bytes()).toEqual(getBytes(new Int32Array([1, 3, 7])));
  });

  it('bool tensor', async () => {
    const a = tf.tensor([true, true, false], [3], 'bool');
    expect(await a.bytes()).toEqual(new Uint8Array([1, 1, 0]));
  });

  it('string tensor from native strings', async () => {
    const a = tf.tensor(['hello', 'world'], [2], 'string');
    expect(await a.bytes()).toEqual([
      encodeString('hello'), encodeString('world')
    ]);
  });

  it('string tensor from encoded bytes', async () => {
    const a = tf.tensor(
        [encodeString('hello'), encodeString('world')], [2], 'string');
    expect(await a.bytes()).toEqual([
      encodeString('hello'), encodeString('world')
    ]);
  });
});
