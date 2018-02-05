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

import * as dl from '../index';
import * as test_util from '../test_util';
import {MathTests} from '../test_util';

import {Scalar, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D} from './tensor';
import {DType, Rank} from './types';

const tests: MathTests = it => {
  it('Tensors of arbitrary size', () => {
    // [1, 2, 3]
    let t: Tensor = Tensor1D.new([1, 2, 3]);
    expect(t instanceof Tensor1D).toBe(true);
    expect(t.rank).toBe(1);
    expect(t.size).toBe(3);
    test_util.expectArraysClose(t, [1, 2, 3]);
    // Out of bounds indexing.
    expect(t.get(4)).toBeUndefined();

    // [[1, 2, 3]]
    t = Tensor2D.new([1, 3], [1, 2, 3]);
    expect(t instanceof Tensor2D).toBe(true);
    expect(t.rank).toBe(2);
    expect(t.size).toBe(3);
    test_util.expectArraysClose(t, [1, 2, 3]);
    // Out of bounds indexing.
    expect(t.get(4)).toBeUndefined();

    // [[1, 2, 3],
    //  [4, 5, 6]]
    t = Tensor2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    expect(t instanceof Tensor2D).toBe(true);
    expect(t.rank).toBe(2);
    expect(t.size).toBe(6);

    test_util.expectArraysClose(t, [1, 2, 3, 4, 5, 6]);

    // Out of bounds indexing.
    expect(t.get(5, 3)).toBeUndefined();

    // Shape mismatch with the values.
    expect(() => Tensor2D.new([1, 2], [1])).toThrowError();
  });

  it('Tensors of explicit size', () => {
    const t = Tensor1D.new([5, 3, 2]);
    expect(t.rank).toBe(1);
    expect(t.shape).toEqual([3]);
    test_util.expectNumbersClose(t.get(1), 3);

    expect(() => Tensor3D.new([1, 2, 3, 5], [1, 2])).toThrowError();

    const t4 = Tensor4D.new([1, 2, 1, 2], [1, 2, 3, 4]);
    test_util.expectNumbersClose(t4.get(0, 0, 0, 0), 1);
    test_util.expectNumbersClose(t4.get(0, 0, 0, 1), 2);
    test_util.expectNumbersClose(t4.get(0, 1, 0, 0), 3);
    test_util.expectNumbersClose(t4.get(0, 1, 0, 1), 4);

    // Tensor of ones.
    const x = dl.ones<Rank.R3>([3, 4, 2]);
    expect(x.rank).toBe(3);
    expect(x.size).toBe(24);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 4; j++) {
        for (let k = 0; k < 2; k++) {
          test_util.expectNumbersClose(x.get(i, j, k), 1);
        }
      }
    }

    // Tensor of zeros.
    const z = dl.zeros<Rank.R3>([3, 4, 2]);
    expect(z.rank).toBe(3);
    expect(z.size).toBe(24);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 4; j++) {
        for (let k = 0; k < 2; k++) {
          test_util.expectNumbersClose(z.get(i, j, k), 0);
        }
      }
    }

    // Reshaping tensors.
    const a = Tensor2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    test_util.expectNumbersClose(a.get(1, 2), 6);
  });

  it('Tensor dataSync CPU --> GPU', () => {
    const a = Tensor2D.new([3, 2], [1, 2, 3, 4, 5, 6]);
    test_util.expectArraysClose(
        a.dataSync(), new Float32Array([1, 2, 3, 4, 5, 6]));
  });

  it('Tensor.data() CPU --> GPU', async () => {
    const a = Tensor2D.new([3, 2], [1, 2, 3, 4, 5, 6]);
    test_util.expectArraysClose(
        await a.data(), new Float32Array([1, 2, 3, 4, 5, 6]));
  });

  it('Scalar basic methods', () => {
    const a = Scalar.new(5);
    test_util.expectNumbersClose(a.get(), 5);
    test_util.expectArraysClose(a, [5]);
    expect(a.rank).toBe(0);
    expect(a.size).toBe(1);
    expect(a.shape).toEqual([]);
  });

  it('indexToLoc Scalar', () => {
    const a = Scalar.new(0);
    expect(a.indexToLoc(0)).toEqual([]);

    const b = dl.zeros<Rank.R0>([]);
    expect(b.indexToLoc(0)).toEqual([]);
  });

  it('indexToLoc Tensor1D', () => {
    const a = dl.zeros([3]);
    expect(a.indexToLoc(0)).toEqual([0]);
    expect(a.indexToLoc(1)).toEqual([1]);
    expect(a.indexToLoc(2)).toEqual([2]);

    const b = dl.zeros<Rank.R1>([3]);
    expect(b.indexToLoc(0)).toEqual([0]);
    expect(b.indexToLoc(1)).toEqual([1]);
    expect(b.indexToLoc(2)).toEqual([2]);
  });

  it('indexToLoc Tensor2D', () => {
    const a = dl.zeros([3, 2]);
    expect(a.indexToLoc(0)).toEqual([0, 0]);
    expect(a.indexToLoc(1)).toEqual([0, 1]);
    expect(a.indexToLoc(2)).toEqual([1, 0]);
    expect(a.indexToLoc(3)).toEqual([1, 1]);
    expect(a.indexToLoc(4)).toEqual([2, 0]);
    expect(a.indexToLoc(5)).toEqual([2, 1]);

    const b = dl.zeros<Rank.R2>([3, 2]);
    expect(b.indexToLoc(0)).toEqual([0, 0]);
    expect(b.indexToLoc(1)).toEqual([0, 1]);
    expect(b.indexToLoc(2)).toEqual([1, 0]);
    expect(b.indexToLoc(3)).toEqual([1, 1]);
    expect(b.indexToLoc(4)).toEqual([2, 0]);
    expect(b.indexToLoc(5)).toEqual([2, 1]);
  });

  it('indexToLoc Tensor3D', () => {
    const a = dl.zeros([3, 2, 2]);
    expect(a.indexToLoc(0)).toEqual([0, 0, 0]);
    expect(a.indexToLoc(1)).toEqual([0, 0, 1]);
    expect(a.indexToLoc(2)).toEqual([0, 1, 0]);
    expect(a.indexToLoc(3)).toEqual([0, 1, 1]);
    expect(a.indexToLoc(4)).toEqual([1, 0, 0]);
    expect(a.indexToLoc(5)).toEqual([1, 0, 1]);
    expect(a.indexToLoc(11)).toEqual([2, 1, 1]);

    const b = dl.zeros<Rank.R3>([3, 2, 2]);
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
    const a = Tensor.make([2, 1, 1, 1, 2], {values});
    expect(a.indexToLoc(0)).toEqual([0, 0, 0, 0, 0]);
    expect(a.indexToLoc(1)).toEqual([0, 0, 0, 0, 1]);
    expect(a.indexToLoc(2)).toEqual([1, 0, 0, 0, 0]);
    expect(a.indexToLoc(3)).toEqual([1, 0, 0, 0, 1]);
  });

  it('locToIndex Scalar', () => {
    const a = Scalar.new(0);
    expect(a.locToIndex([])).toEqual(0);

    const b = dl.zeros<Rank.R0>([]);
    expect(b.locToIndex([])).toEqual(0);
  });

  it('locToIndex Tensor1D', () => {
    const a = dl.zeros<Rank.R1>([3]);
    expect(a.locToIndex([0])).toEqual(0);
    expect(a.locToIndex([1])).toEqual(1);
    expect(a.locToIndex([2])).toEqual(2);

    const b = dl.zeros<Rank.R1>([3]);
    expect(b.locToIndex([0])).toEqual(0);
    expect(b.locToIndex([1])).toEqual(1);
    expect(b.locToIndex([2])).toEqual(2);
  });

  it('locToIndex Tensor2D', () => {
    const a = dl.zeros<Rank.R2>([3, 2]);
    expect(a.locToIndex([0, 0])).toEqual(0);
    expect(a.locToIndex([0, 1])).toEqual(1);
    expect(a.locToIndex([1, 0])).toEqual(2);
    expect(a.locToIndex([1, 1])).toEqual(3);
    expect(a.locToIndex([2, 0])).toEqual(4);
    expect(a.locToIndex([2, 1])).toEqual(5);

    const b = dl.zeros<Rank.R2>([3, 2]);
    expect(b.locToIndex([0, 0])).toEqual(0);
    expect(b.locToIndex([0, 1])).toEqual(1);
    expect(b.locToIndex([1, 0])).toEqual(2);
    expect(b.locToIndex([1, 1])).toEqual(3);
    expect(b.locToIndex([2, 0])).toEqual(4);
    expect(b.locToIndex([2, 1])).toEqual(5);
  });

  it('locToIndex Tensor3D', () => {
    const a = dl.zeros<Rank.R3>([3, 2, 2]);
    expect(a.locToIndex([0, 0, 0])).toEqual(0);
    expect(a.locToIndex([0, 0, 1])).toEqual(1);
    expect(a.locToIndex([0, 1, 0])).toEqual(2);
    expect(a.locToIndex([0, 1, 1])).toEqual(3);
    expect(a.locToIndex([1, 0, 0])).toEqual(4);
    expect(a.locToIndex([1, 0, 1])).toEqual(5);
    expect(a.locToIndex([2, 1, 1])).toEqual(11);

    const b = dl.zeros<Rank.R3>([3, 2, 2]);
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
};
const testsNew: MathTests = it => {
  it('Tensor1D.new() from number[]', () => {
    const a = Tensor1D.new([1, 2, 3]);
    test_util.expectArraysClose(a, [1, 2, 3]);
  });

  it('Tensor1D.new() from number[][], shape mismatch', () => {
    // tslint:disable-next-line:no-any
    expect(() => Tensor1D.new([[1], [2], [3]] as any)).toThrowError();
  });

  it('Tensor2D.new() from number[][]', () => {
    const a = Tensor2D.new([2, 3], [[1, 2, 3], [4, 5, 6]]);
    test_util.expectArraysClose(a, [1, 2, 3, 4, 5, 6]);
  });

  it('Tensor2D.new() from number[][], but shape does not match', () => {
    // Actual shape is [2, 3].
    expect(() => Tensor2D.new([3, 2], [[1, 2, 3], [4, 5, 6]])).toThrowError();
  });

  it('Tensor3D.new() from number[][][]', () => {
    const a = Tensor3D.new([2, 3, 1], [[[1], [2], [3]], [[4], [5], [6]]]);
    test_util.expectArraysClose(a, [1, 2, 3, 4, 5, 6]);
  });

  it('Tensor3D.new() from number[][][], but shape does not match', () => {
    const values = [[[1], [2], [3]], [[4], [5], [6]]];
    // Actual shape is [2, 3, 1].
    expect(() => Tensor3D.new([3, 2, 1], values)).toThrowError();
  });

  it('Tensor4D.new() from number[][][][]', () => {
    const a = Tensor4D.new([2, 2, 1, 1], [[[[1]], [[2]]], [[[4]], [[5]]]]);
    test_util.expectArraysClose(a, [1, 2, 4, 5]);
  });

  it('Tensor4D.new() from number[][][][], but shape does not match', () => {
    const f = () => {
      // Actual shape is [2, 2, 1, 1].
      Tensor4D.new([2, 1, 2, 1], [[[[1]], [[2]]], [[[4]], [[5]]]]);
    };
    expect(f).toThrowError();
  });
};

const testsScalarNew: MathTests = it => {
  it('default dtype', () => {
    const a = Scalar.new(3);
    expect(a.dtype).toBe('float32');
    test_util.expectArraysClose(a, [3]);
  });

  it('float32 dtype', () => {
    const a = Scalar.new(3, 'float32');
    expect(a.dtype).toBe('float32');
    test_util.expectArraysClose(a, [3]);
  });

  it('int32 dtype', () => {
    const a = Scalar.new(3, 'int32');
    expect(a.dtype).toBe('int32');
    test_util.expectArraysEqual(a, [3]);
  });

  it('int32 dtype, 3.9 => 3, like numpy', () => {
    const a = Scalar.new(3.9, 'int32');
    expect(a.dtype).toBe('int32');
    test_util.expectArraysEqual(a, [3]);
  });

  it('int32 dtype, -3.9 => -3, like numpy', () => {
    const a = Scalar.new(-3.9, 'int32');
    expect(a.dtype).toBe('int32');
    test_util.expectArraysEqual(a, [-3]);
  });

  it('bool dtype, 3 => true, like numpy', () => {
    const a = Scalar.new(3, 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.get()).toBe(1);
  });

  it('bool dtype, -2 => true, like numpy', () => {
    const a = Scalar.new(-2, 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.get()).toBe(1);
  });

  it('bool dtype, 0 => false, like numpy', () => {
    const a = Scalar.new(0, 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.get()).toBe(0);
  });

  it('bool dtype from boolean', () => {
    const a = Scalar.new(false, 'bool');
    expect(a.get()).toBe(0);
    expect(a.dtype).toBe('bool');

    const b = Scalar.new(true, 'bool');
    expect(b.get()).toBe(1);
    expect(b.dtype).toBe('bool');
  });

  it('int32 dtype from boolean', () => {
    const a = Scalar.new(true, 'int32');
    expect(a.get()).toBe(1);
    expect(a.dtype).toBe('int32');
  });

  it('default dtype from boolean', () => {
    const a = Scalar.new(false);
    test_util.expectNumbersClose(a.get(), 0);
    expect(a.dtype).toBe('float32');
  });
};
const testsTensor1DNew: MathTests = it => {
  it('default dtype', () => {
    const a = Tensor1D.new([1, 2, 3]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    test_util.expectArraysClose(a, [1, 2, 3]);
  });

  it('float32 dtype', () => {
    const a = Tensor1D.new([1, 2, 3], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    test_util.expectArraysClose(a, [1, 2, 3]);
  });

  it('int32 dtype', () => {
    const a = Tensor1D.new([1, 2, 3], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3]);
    test_util.expectArraysEqual(a, [1, 2, 3]);
  });

  it('int32 dtype, non-ints get floored, like numpy', () => {
    const a = Tensor1D.new([1.1, 2.5, 3.9], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3]);
    test_util.expectArraysEqual(a, [1, 2, 3]);
  });

  it('int32 dtype, negative non-ints get ceiled, like numpy', () => {
    const a = Tensor1D.new([-1.1, -2.5, -3.9], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3]);
    test_util.expectArraysEqual(a, [-1, -2, -3]);
  });

  it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', () => {
    const a = Tensor1D.new([1, -2, 0, 3], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([4]);
    expect(a.get(0)).toBe(1);
    expect(a.get(1)).toBe(1);
    expect(a.get(2)).toBe(0);
    expect(a.get(3)).toBe(1);
  });

  it('default dtype from boolean[]', () => {
    const a = Tensor1D.new([false, false, true]);
    expect(a.dtype).toBe('float32');
    test_util.expectArraysClose(a, [0, 0, 1]);
  });

  it('int32 dtype from boolean[]', () => {
    const a = Tensor1D.new([false, false, true], 'int32');
    expect(a.dtype).toBe('int32');
    test_util.expectArraysEqual(a, [0, 0, 1]);
  });

  it('bool dtype from boolean[]', () => {
    const a = Tensor1D.new([false, false, true], 'bool');
    expect(a.dtype).toBe('bool');
    test_util.expectArraysEqual(a, [0, 0, 1]);
  });
};
const testsTensor2DNew: MathTests = it => {
  it('default dtype', () => {
    const a = Tensor2D.new([2, 2], [1, 2, 3, 4]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2]);
    test_util.expectArraysClose(a, [1, 2, 3, 4]);
  });

  it('float32 dtype', () => {
    const a = Tensor2D.new([2, 2], [1, 2, 3, 4]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2]);
    test_util.expectArraysClose(a, [1, 2, 3, 4]);
  });

  it('int32 dtype', () => {
    const a = Tensor2D.new([2, 2], [[1, 2], [3, 4]], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2]);
    test_util.expectArraysEqual(a, [1, 2, 3, 4]);
  });

  it('int32 dtype, non-ints get floored, like numpy', () => {
    const a = Tensor2D.new([2, 2], [1.1, 2.5, 3.9, 4.0], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2]);
    test_util.expectArraysEqual(a, [1, 2, 3, 4]);
  });

  it('int32 dtype, negative non-ints get ceiled, like numpy', () => {
    const a = Tensor2D.new([2, 2], [-1.1, -2.5, -3.9, -4.0], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2]);
    test_util.expectArraysEqual(a, [-1, -2, -3, -4]);
  });

  it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', () => {
    const a = Tensor2D.new([2, 2], [1, -2, 0, 3], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([2, 2]);
    expect(a.get(0, 0)).toBe(1);
    expect(a.get(0, 1)).toBe(1);
    expect(a.get(1, 0)).toBe(0);
    expect(a.get(1, 1)).toBe(1);
  });

  it('default dtype from boolean[]', () => {
    const a = Tensor2D.new([2, 2], [[false, false], [true, false]]);
    expect(a.dtype).toBe('float32');
    test_util.expectArraysClose(a, [0, 0, 1, 0]);
  });

  it('int32 dtype from boolean[]', () => {
    const a = Tensor2D.new([2, 2], [[false, false], [true, false]], 'int32');
    expect(a.dtype).toBe('int32');
    test_util.expectArraysEqual(a, [0, 0, 1, 0]);
  });

  it('bool dtype from boolean[]', () => {
    const a = Tensor2D.new([2, 2], [[false, false], [true, false]], 'bool');
    expect(a.dtype).toBe('bool');
    test_util.expectArraysEqual(a, [0, 0, 1, 0]);
  });
};
const testsTensor3DNew: MathTests = it => {
  it('default dtype', () => {
    const a = Tensor3D.new([2, 2, 1], [1, 2, 3, 4]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 1]);
    test_util.expectArraysClose(a, [1, 2, 3, 4]);
  });

  it('float32 dtype', () => {
    const a = Tensor3D.new([2, 2, 1], [1, 2, 3, 4]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 1]);
    test_util.expectArraysClose(a, [1, 2, 3, 4]);
  });

  it('int32 dtype', () => {
    const a = Tensor3D.new([2, 2, 1], [[[1], [2]], [[3], [4]]], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 1]);
    test_util.expectArraysEqual(a, [1, 2, 3, 4]);
  });

  it('int32 dtype, non-ints get floored, like numpy', () => {
    const a = Tensor3D.new([2, 2, 1], [1.1, 2.5, 3.9, 4.0], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 1]);
    test_util.expectArraysEqual(a, [1, 2, 3, 4]);
  });

  it('int32 dtype, negative non-ints get ceiled, like numpy', () => {
    const a = Tensor3D.new([2, 2, 1], [-1.1, -2.5, -3.9, -4.0], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 1]);
    test_util.expectArraysEqual(a, [-1, -2, -3, -4]);
  });

  it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', () => {
    const a = Tensor3D.new([2, 2, 1], [1, -2, 0, 3], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([2, 2, 1]);
    expect(a.get(0, 0, 0)).toBe(1);
    expect(a.get(0, 1, 0)).toBe(1);
    expect(a.get(1, 0, 0)).toBe(0);
    expect(a.get(1, 1, 0)).toBe(1);
  });

  it('default dtype from boolean[]', () => {
    const a = Tensor3D.new([2, 2, 1], [[[false], [false]], [[true], [false]]]);
    expect(a.dtype).toBe('float32');
    test_util.expectArraysClose(a, [0, 0, 1, 0]);
  });

  it('int32 dtype from boolean[]', () => {
    const a = Tensor3D.new(
        [2, 2, 1], [[[false], [false]], [[true], [false]]], 'int32');
    expect(a.dtype).toBe('int32');
    test_util.expectArraysEqual(a, [0, 0, 1, 0]);
  });

  it('bool dtype from boolean[]', () => {
    const a = Tensor3D.new(
        [2, 2, 1], [[[false], [false]], [[true], [false]]], 'bool');
    expect(a.dtype).toBe('bool');
    test_util.expectArraysEqual(a, [0, 0, 1, 0]);
  });
};
const testsTensor4DNew: MathTests = it => {
  it('default dtype', () => {
    const a = Tensor4D.new([2, 2, 1, 1], [1, 2, 3, 4]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysClose(a, [1, 2, 3, 4]);
  });

  it('float32 dtype', () => {
    const a = Tensor4D.new([2, 2, 1, 1], [1, 2, 3, 4]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysClose(a, [1, 2, 3, 4]);
  });

  it('int32 dtype', () => {
    const a =
        Tensor4D.new([2, 2, 1, 1], [[[[1]], [[2]]], [[[3]], [[4]]]], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysEqual(a, [1, 2, 3, 4]);
  });

  it('int32 dtype, non-ints get floored, like numpy', () => {
    const a = Tensor4D.new([2, 2, 1, 1], [1.1, 2.5, 3.9, 4.0], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysEqual(a, [1, 2, 3, 4]);
  });

  it('int32 dtype, negative non-ints get ceiled, like numpy', () => {
    const a = Tensor4D.new([2, 2, 1, 1], [-1.1, -2.5, -3.9, -4.0], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysEqual(a, [-1, -2, -3, -4]);
  });

  it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', () => {
    const a = Tensor4D.new([2, 2, 1, 1], [1, -2, 0, 3], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    expect(a.get(0, 0, 0, 0)).toBe(1);
    expect(a.get(0, 1, 0, 0)).toBe(1);
    expect(a.get(1, 0, 0, 0)).toBe(0);
    expect(a.get(1, 1, 0, 0)).toBe(1);
  });

  it('default dtype from boolean[]', () => {
    const a =
        Tensor4D.new([1, 2, 2, 1], [[[[false], [false]], [[true], [false]]]]);
    expect(a.dtype).toBe('float32');
    test_util.expectArraysClose(a, [0, 0, 1, 0]);
  });

  it('int32 dtype from boolean[]', () => {
    const a = Tensor4D.new(
        [1, 2, 2, 1], [[[[false], [false]], [[true], [false]]]], 'int32');
    expect(a.dtype).toBe('int32');
    test_util.expectArraysEqual(a, [0, 0, 1, 0]);
  });

  it('bool dtype from boolean[]', () => {
    const a = Tensor4D.new(
        [1, 2, 2, 1], [[[[false], [false]], [[true], [false]]]], 'bool');
    expect(a.dtype).toBe('bool');
    test_util.expectArraysEqual(a, [0, 0, 1, 0]);
  });
};
const testsReshape: MathTests = it => {
  it('Scalar default dtype', () => {
    const a = Scalar.new(4);
    const b = a.reshape([1, 1]);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([1, 1]);
  });

  it('Scalar bool dtype', () => {
    const a = Scalar.new(4, 'bool');
    const b = a.reshape([1, 1, 1]);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([1, 1, 1]);
  });

  it('Tensor1D default dtype', () => {
    const a = Tensor1D.new([1, 2, 3, 4]);
    const b = a.reshape([2, 2]);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
  });

  it('Tensor1D int32 dtype', () => {
    const a = Tensor1D.new([1, 2, 3, 4], 'int32');
    const b = a.reshape([2, 2]);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2]);
  });

  it('Tensor2D default dtype', () => {
    const a = Tensor2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    const b = a.reshape([6]);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([6]);
  });

  it('Tensor2D bool dtype', () => {
    const a = Tensor2D.new([2, 3], [1, 2, 3, 4, 5, 6], 'bool');
    const b = a.reshape([6]);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([6]);
  });

  it('Tensor3D default dtype', () => {
    const a = Tensor3D.new([2, 3, 1], [1, 2, 3, 4, 5, 6]);
    const b = a.reshape([6]);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([6]);
  });

  it('Tensor3D bool dtype', () => {
    const a = Tensor3D.new([2, 3, 1], [1, 2, 3, 4, 5, 6], 'bool');
    const b = a.reshape([6]);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([6]);
  });

  it('Tensor4D default dtype', () => {
    const a = Tensor4D.new([2, 3, 1, 1], [1, 2, 3, 4, 5, 6]);
    const b = a.reshape([2, 3]);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 3]);
  });

  it('Tensor4D int32 dtype', () => {
    const a = Tensor4D.new([2, 3, 1, 1], [1, 2, 3, 4, 5, 6], 'int32');
    const b = a.reshape([3, 2]);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([3, 2]);
  });

  it('reshape is functional', math => {
    const a = Scalar.new(2.4);
    const b = a.reshape([]);
    expect(a.id).not.toBe(b.id);
    b.dispose();
    test_util.expectArraysClose(a, [2.4]);
  });
};
const testsAsType: MathTests = it => {
  it('scalar bool -> int32', () => {
    const a = Scalar.new(true, 'bool').toInt();
    expect(a.dtype).toBe('int32');
    expect(a.get()).toBe(1);
  });

  it('Tensor1D float32 -> int32', () => {
    const a = Tensor1D.new([1.1, 3.9, -2.9, 0]).toInt();
    expect(a.dtype).toBe('int32');
    test_util.expectArraysEqual(a, [1, 3, -2, 0]);
  });

  it('Tensor2D float32 -> bool', () => {
    const a = Tensor2D.new([2, 2], [1.1, 3.9, -2.9, 0]).asType(DType.bool);
    expect(a.dtype).toBe('bool');
    expect(a.get(0, 0)).toBe(1);
    expect(a.get(0, 1)).toBe(1);
    expect(a.get(1, 0)).toBe(1);
    expect(a.get(1, 1)).toBe(0);
  });

  it('Tensor2D int32 -> bool', () => {
    const a = Tensor2D.new([2, 2], [1, 3, 0, -1], 'int32').toBool();
    expect(a.dtype).toBe('bool');
    expect(a.get(0, 0)).toBe(1);
    expect(a.get(0, 1)).toBe(1);
    expect(a.get(1, 0)).toBe(0);
    expect(a.get(1, 1)).toBe(1);
  });

  it('Tensor3D bool -> float32', () => {
    const a =
        Tensor3D.new([2, 2, 1], [true, false, false, true], 'bool').toFloat();
    expect(a.dtype).toBe('float32');
    test_util.expectArraysClose(a, [1, 0, 0, 1]);
  });

  it('bool CPU -> GPU -> CPU', () => {
    const a = Tensor1D.new([1, 2, 0, 0, 5], 'bool');
    test_util.expectArraysEqual(a, [1, 1, 0, 0, 1]);
  });

  it('int32 CPU -> GPU -> CPU', () => {
    const a = Tensor1D.new([1, 2, 0, 0, 5], 'int32');
    test_util.expectArraysEqual(a, [1, 2, 0, 0, 5]);
  });

  it('asType is functional', math => {
    const a = Scalar.new(2.4, 'float32');
    const b = a.toFloat();
    expect(a.id).not.toBe(b.id);
    b.dispose();
    test_util.expectArraysClose(a, [2.4]);
  });
};

const testSqueeze: MathTests = it => {
  it('squeeze no axis', () => {
    const a = Tensor2D.new([3, 1], [4, 2, 1], 'bool');
    const b = a.squeeze();
    expect(b.shape).toEqual([3]);
  });

  it('squeeze with axis', () => {
    const a = Tensor3D.new([3, 1, 1], [4, 2, 1], 'bool');
    const b = a.squeeze([1]);
    expect(b.shape).toEqual([3, 1]);
  });

  it('squeeze wrong axis', () => {
    const a = Tensor3D.new([3, 1, 1], [4, 2, 1], 'bool');
    expect(() => a.squeeze([0, 1])).toThrowError('axis 0 is not 1');
  });
};
const testsAsXD: MathTests = it => {
  it('scalar -> 2d', () => {
    const a = Scalar.new(4, 'int32');
    const b = a.as2D(1, 1);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([1, 1]);
  });

  it('1d -> 2d', () => {
    const a = Tensor1D.new([4, 2, 1], 'bool');
    const b = a.as2D(3, 1);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([3, 1]);
  });

  it('2d -> 4d', () => {
    const a = Tensor2D.new([2, 2], [4, 2, 1, 3]);
    const b = a.as4D(1, 1, 2, 2);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([1, 1, 2, 2]);
  });

  it('3d -> 2d', () => {
    const a = Tensor3D.new([2, 2, 1], [4, 2, 1, 3], 'float32');
    const b = a.as2D(2, 2);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
  });

  it('4d -> 1d', () => {
    const a = Tensor4D.new([2, 2, 1, 1], [4, 2, 1, 3], 'bool');
    const b = a.as1D();
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([4]);
  });
};

const allTests = [
  tests, testsNew, testsScalarNew, testsTensor1DNew, testsTensor2DNew,
  testsTensor3DNew, testsTensor4DNew, testsReshape, testsAsType, testsAsXD,
  testSqueeze
];

test_util.describeMathCPU('Tensor', allTests);
test_util.describeMathGPU('Tensor', allTests, [
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
]);
