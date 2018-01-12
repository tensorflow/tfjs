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

import * as test_util from '../test_util';
import {MathTests} from '../test_util';
import * as util from '../util';
// tslint:disable-next-line:max-line-length
import {Array1D, Array2D, Array3D, Array4D, DType, NDArray, Scalar} from './ndarray';

const tests: MathTests = it => {
  it('NDArrays of arbitrary size', () => {
    // [1, 2, 3]
    let t: NDArray = Array1D.new([1, 2, 3]);
    expect(t instanceof Array1D).toBe(true);
    expect(t.rank).toBe(1);
    expect(t.size).toBe(3);
    test_util.expectArraysClose(t, [1, 2, 3]);
    // Out of bounds indexing.
    expect(t.get(4)).toBeUndefined();

    // [[1, 2, 3]]
    t = Array2D.new([1, 3], [1, 2, 3]);
    expect(t instanceof Array2D).toBe(true);
    expect(t.rank).toBe(2);
    expect(t.size).toBe(3);
    test_util.expectArraysClose(t, [1, 2, 3]);
    // Out of bounds indexing.
    expect(t.get(4)).toBeUndefined();

    // [[1, 2, 3],
    //  [4, 5, 6]]
    t = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    expect(t instanceof Array2D).toBe(true);
    expect(t.rank).toBe(2);
    expect(t.size).toBe(6);

    test_util.expectArraysClose(t, [1, 2, 3, 4, 5, 6]);

    // Out of bounds indexing.
    expect(t.get(5, 3)).toBeUndefined();

    // Shape mismatch with the values.
    expect(() => Array2D.new([1, 2], [1])).toThrowError();
  });

  it('NDArrays of explicit size', () => {
    const t = Array1D.new([5, 3, 2]);
    expect(t.rank).toBe(1);
    expect(t.shape).toEqual([3]);
    test_util.expectNumbersClose(t.get(1), 3);

    expect(() => Array3D.new([1, 2, 3, 5], [
      1, 2
    ])).toThrowError('Shape should be of length 3');

    const t4 = Array4D.new([1, 2, 1, 2], [1, 2, 3, 4]);
    test_util.expectNumbersClose(t4.get(0, 0, 0, 0), 1);
    test_util.expectNumbersClose(t4.get(0, 0, 0, 1), 2);
    test_util.expectNumbersClose(t4.get(0, 1, 0, 0), 3);
    test_util.expectNumbersClose(t4.get(0, 1, 0, 1), 4);

    const t4Like = NDArray.like(t4);
    // Change t4.
    t4.set(10, 0, 0, 0, 1);
    test_util.expectNumbersClose(t4.get(0, 0, 0, 1), 10);
    // Make suree t4_like hasn't changed.
    test_util.expectNumbersClose(t4Like.get(0, 0, 0, 1), 2);

    // NDArray of ones.
    const x = Array3D.ones([3, 4, 2]);
    expect(x.rank).toBe(3);
    expect(x.size).toBe(24);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 4; j++) {
        for (let k = 0; k < 2; k++) {
          test_util.expectNumbersClose(x.get(i, j, k), 1);
        }
      }
    }

    // NDArray of zeros.
    const z = Array3D.zeros([3, 4, 2]);
    expect(z.rank).toBe(3);
    expect(z.size).toBe(24);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 4; j++) {
        for (let k = 0; k < 2; k++) {
          test_util.expectNumbersClose(z.get(i, j, k), 0);
        }
      }
    }

    // Reshaping ndarrays.
    const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    const b = a.reshape([3, 2, 1]);
    test_util.expectNumbersClose(a.get(1, 2), 6);

    // Modify the reshaped ndarray.
    b.set(10, 2, 1, 0);
    // Make sure the original ndarray is also modified.
    test_util.expectNumbersClose(a.get(1, 2), 10);
  });

  it('NDArray dataSync CPU --> GPU', () => {
    const a = Array2D.new([3, 2], [1, 2, 3, 4, 5, 6]);
    test_util.expectArraysClose(
        a.dataSync(), new Float32Array([1, 2, 3, 4, 5, 6]));
  });

  it('NDArray.data() CPU --> GPU', async () => {
    const a = Array2D.new([3, 2], [1, 2, 3, 4, 5, 6]);
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

    const b = NDArray.zeros<'float32', '0'>([]);
    expect(b.indexToLoc(0)).toEqual([]);
  });

  it('indexToLoc Array1D', () => {
    const a = Array1D.zeros([3]);
    expect(a.indexToLoc(0)).toEqual([0]);
    expect(a.indexToLoc(1)).toEqual([1]);
    expect(a.indexToLoc(2)).toEqual([2]);

    const b = NDArray.zeros<'float32', '1'>([3]);
    expect(b.indexToLoc(0)).toEqual([0]);
    expect(b.indexToLoc(1)).toEqual([1]);
    expect(b.indexToLoc(2)).toEqual([2]);
  });

  it('indexToLoc Array2D', () => {
    const a = Array2D.zeros([3, 2]);
    expect(a.indexToLoc(0)).toEqual([0, 0]);
    expect(a.indexToLoc(1)).toEqual([0, 1]);
    expect(a.indexToLoc(2)).toEqual([1, 0]);
    expect(a.indexToLoc(3)).toEqual([1, 1]);
    expect(a.indexToLoc(4)).toEqual([2, 0]);
    expect(a.indexToLoc(5)).toEqual([2, 1]);

    const b = NDArray.zeros<'float32', '2'>([3, 2]);
    expect(b.indexToLoc(0)).toEqual([0, 0]);
    expect(b.indexToLoc(1)).toEqual([0, 1]);
    expect(b.indexToLoc(2)).toEqual([1, 0]);
    expect(b.indexToLoc(3)).toEqual([1, 1]);
    expect(b.indexToLoc(4)).toEqual([2, 0]);
    expect(b.indexToLoc(5)).toEqual([2, 1]);
  });

  it('indexToLoc Array3D', () => {
    const a = Array3D.zeros([3, 2, 2]);
    expect(a.indexToLoc(0)).toEqual([0, 0, 0]);
    expect(a.indexToLoc(1)).toEqual([0, 0, 1]);
    expect(a.indexToLoc(2)).toEqual([0, 1, 0]);
    expect(a.indexToLoc(3)).toEqual([0, 1, 1]);
    expect(a.indexToLoc(4)).toEqual([1, 0, 0]);
    expect(a.indexToLoc(5)).toEqual([1, 0, 1]);
    expect(a.indexToLoc(11)).toEqual([2, 1, 1]);

    const b = NDArray.zeros<'float32', '3'>([3, 2, 2]);
    expect(b.indexToLoc(0)).toEqual([0, 0, 0]);
    expect(b.indexToLoc(1)).toEqual([0, 0, 1]);
    expect(b.indexToLoc(2)).toEqual([0, 1, 0]);
    expect(b.indexToLoc(3)).toEqual([0, 1, 1]);
    expect(b.indexToLoc(4)).toEqual([1, 0, 0]);
    expect(b.indexToLoc(5)).toEqual([1, 0, 1]);
    expect(b.indexToLoc(11)).toEqual([2, 1, 1]);
  });

  it('indexToLoc NDArray 5D', () => {
    const values = new Float32Array([1, 2, 3, 4]);
    const a = NDArray.make([2, 1, 1, 1, 2], {values});
    expect(a.indexToLoc(0)).toEqual([0, 0, 0, 0, 0]);
    expect(a.indexToLoc(1)).toEqual([0, 0, 0, 0, 1]);
    expect(a.indexToLoc(2)).toEqual([1, 0, 0, 0, 0]);
    expect(a.indexToLoc(3)).toEqual([1, 0, 0, 0, 1]);
  });

  it('locToIndex Scalar', () => {
    const a = Scalar.new(0);
    expect(a.locToIndex([])).toEqual(0);

    const b = NDArray.zeros<'float32', '0'>([]);
    expect(b.locToIndex([])).toEqual(0);
  });

  it('locToIndex Array1D', () => {
    const a = Array1D.zeros([3]);
    expect(a.locToIndex([0])).toEqual(0);
    expect(a.locToIndex([1])).toEqual(1);
    expect(a.locToIndex([2])).toEqual(2);

    const b = NDArray.zeros<'float32', '1'>([3]);
    expect(b.locToIndex([0])).toEqual(0);
    expect(b.locToIndex([1])).toEqual(1);
    expect(b.locToIndex([2])).toEqual(2);
  });

  it('locToIndex Array2D', () => {
    const a = Array2D.zeros([3, 2]);
    expect(a.locToIndex([0, 0])).toEqual(0);
    expect(a.locToIndex([0, 1])).toEqual(1);
    expect(a.locToIndex([1, 0])).toEqual(2);
    expect(a.locToIndex([1, 1])).toEqual(3);
    expect(a.locToIndex([2, 0])).toEqual(4);
    expect(a.locToIndex([2, 1])).toEqual(5);

    const b = NDArray.zeros<'float32', '2'>([3, 2]);
    expect(b.locToIndex([0, 0])).toEqual(0);
    expect(b.locToIndex([0, 1])).toEqual(1);
    expect(b.locToIndex([1, 0])).toEqual(2);
    expect(b.locToIndex([1, 1])).toEqual(3);
    expect(b.locToIndex([2, 0])).toEqual(4);
    expect(b.locToIndex([2, 1])).toEqual(5);
  });

  it('locToIndex Array3D', () => {
    const a = Array3D.zeros([3, 2, 2]);
    expect(a.locToIndex([0, 0, 0])).toEqual(0);
    expect(a.locToIndex([0, 0, 1])).toEqual(1);
    expect(a.locToIndex([0, 1, 0])).toEqual(2);
    expect(a.locToIndex([0, 1, 1])).toEqual(3);
    expect(a.locToIndex([1, 0, 0])).toEqual(4);
    expect(a.locToIndex([1, 0, 1])).toEqual(5);
    expect(a.locToIndex([2, 1, 1])).toEqual(11);

    const b = NDArray.zeros<'float32', '3'>([3, 2, 2]);
    expect(b.locToIndex([0, 0, 0])).toEqual(0);
    expect(b.locToIndex([0, 0, 1])).toEqual(1);
    expect(b.locToIndex([0, 1, 0])).toEqual(2);
    expect(b.locToIndex([0, 1, 1])).toEqual(3);
    expect(b.locToIndex([1, 0, 0])).toEqual(4);
    expect(b.locToIndex([1, 0, 1])).toEqual(5);
    expect(b.locToIndex([2, 1, 1])).toEqual(11);
  });

  it('NDArray<D, X> is assignable to Scalar/ArrayXD', math => {
    // This test asserts compilation, not doing any run-time assertion.
    const a: NDArray<'float32', '0'> = null;
    const b: Scalar<'float32'> = a;
    expect(b).toBeNull();

    const a1: NDArray<'float32', '1'> = null;
    const b1: Array1D<'float32'> = a1;
    expect(b1).toBeNull();

    const a2: NDArray<'float32', '2'> = null;
    const b2: Array2D<'float32'> = a2;
    expect(b2).toBeNull();

    const a3: NDArray<'float32', '3'> = null;
    const b3: Array3D<'float32'> = a3;
    expect(b3).toBeNull();

    const a4: NDArray<'float32', '4'> = null;
    const b4: Array4D<'float32'> = a4;
    expect(b4).toBeNull();
  });
};
const testsNew: MathTests = it => {
  it('Array1D.new() from number[]', () => {
    const a = Array1D.new([1, 2, 3]);
    test_util.expectArraysClose(a, [1, 2, 3]);
  });

  it('Array1D.new() from number[][], shape mismatch', () => {
    // tslint:disable-next-line:no-any
    expect(() => Array1D.new([[1], [2], [3]] as any)).toThrowError();
  });

  it('Array2D.new() from number[][]', () => {
    const a = Array2D.new([2, 3], [[1, 2, 3], [4, 5, 6]]);
    test_util.expectArraysClose(a, [1, 2, 3, 4, 5, 6]);
  });

  it('Array2D.new() from number[][], but shape does not match', () => {
    // Actual shape is [2, 3].
    expect(() => Array2D.new([3, 2], [[1, 2, 3], [4, 5, 6]])).toThrowError();
  });

  it('Array3D.new() from number[][][]', () => {
    const a = Array3D.new([2, 3, 1], [[[1], [2], [3]], [[4], [5], [6]]]);
    test_util.expectArraysClose(a, [1, 2, 3, 4, 5, 6]);
  });

  it('Array3D.new() from number[][][], but shape does not match', () => {
    const values = [[[1], [2], [3]], [[4], [5], [6]]];
    // Actual shape is [2, 3, 1].
    expect(() => Array3D.new([3, 2, 1], values)).toThrowError();
  });

  it('Array4D.new() from number[][][][]', () => {
    const a = Array4D.new([2, 2, 1, 1], [[[[1]], [[2]]], [[[4]], [[5]]]]);
    test_util.expectArraysClose(a, [1, 2, 4, 5]);
  });

  it('Array4D.new() from number[][][][], but shape does not match', () => {
    const f = () => {
      // Actual shape is [2, 2, 1, 1].
      Array4D.new([2, 1, 2, 1], [[[[1]], [[2]]], [[[4]], [[5]]]]);
    };
    expect(f).toThrowError();
  });
};
const testsZeros: MathTests = it => {
  it('1D default dtype', () => {
    const a = Array1D.zeros([3]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    test_util.expectArraysClose(a, [0, 0, 0]);
  });

  it('1D float32 dtype', () => {
    const a = Array1D.zeros([3], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    test_util.expectArraysClose(a, [0, 0, 0]);
  });

  it('1D int32 dtype', () => {
    const a = Array1D.zeros([3], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3]);
    test_util.expectArraysEqual(a, [0, 0, 0]);
  });

  it('1D bool dtype', () => {
    const a = Array1D.zeros([3], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([3]);
    test_util.expectArraysEqual(a, [0, 0, 0]);
  });

  it('2D default dtype', () => {
    const a = Array2D.zeros([3, 2]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2]);
    test_util.expectArraysClose(a, [0, 0, 0, 0, 0, 0]);
  });

  it('2D float32 dtype', () => {
    const a = Array2D.zeros([3, 2], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2]);
    test_util.expectArraysClose(a, [0, 0, 0, 0, 0, 0]);
  });

  it('2D int32 dtype', () => {
    const a = Array2D.zeros([3, 2], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3, 2]);
    test_util.expectArraysEqual(a, [0, 0, 0, 0, 0, 0]);
  });

  it('2D bool dtype', () => {
    const a = Array2D.zeros([3, 2], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([3, 2]);
    test_util.expectArraysEqual(a, [0, 0, 0, 0, 0, 0]);
  });

  it('3D default dtype', () => {
    const a = Array3D.zeros([2, 2, 2]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 2]);
    test_util.expectArraysClose(a, [0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it('3D float32 dtype', () => {
    const a = Array3D.zeros([2, 2, 2], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 2]);
    test_util.expectArraysClose(a, [0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it('3D int32 dtype', () => {
    const a = Array3D.zeros([2, 2, 2], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 2]);
    test_util.expectArraysEqual(a, [0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it('3D bool dtype', () => {
    const a = Array3D.zeros([2, 2, 2], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([2, 2, 2]);
    test_util.expectArraysEqual(a, [0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it('4D default dtype', () => {
    const a = Array4D.zeros([3, 2, 1, 1]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    test_util.expectArraysClose(a, [0, 0, 0, 0, 0, 0]);
  });

  it('4D float32 dtype', () => {
    const a = Array4D.zeros([3, 2, 1, 1], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    test_util.expectArraysClose(a, [0, 0, 0, 0, 0, 0]);
  });

  it('4D int32 dtype', () => {
    const a = Array4D.zeros([3, 2, 1, 1], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    test_util.expectArraysEqual(a, [0, 0, 0, 0, 0, 0]);
  });

  it('4D bool dtype', () => {
    const a = Array4D.zeros([3, 2, 1, 1], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    test_util.expectArraysEqual(a, [0, 0, 0, 0, 0, 0]);
  });
};
const testsOnes: MathTests = it => {
  it('1D default dtype', () => {
    const a = Array1D.ones([3]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    test_util.expectArraysClose(a, [1, 1, 1]);
  });

  it('1D float32 dtype', () => {
    const a = Array1D.ones([3], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    test_util.expectArraysClose(a, [1, 1, 1]);
  });

  it('1D int32 dtype', () => {
    const a = Array1D.ones([3], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3]);
    test_util.expectArraysEqual(a, [1, 1, 1]);
  });

  it('1D bool dtype', () => {
    const a = Array1D.ones([3], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([3]);
    test_util.expectArraysEqual(a, [1, 1, 1]);
  });

  it('2D default dtype', () => {
    const a = Array2D.ones([3, 2]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2]);
    test_util.expectArraysClose(a, [1, 1, 1, 1, 1, 1]);
  });

  it('2D float32 dtype', () => {
    const a = Array2D.ones([3, 2], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2]);
    test_util.expectArraysClose(a, [1, 1, 1, 1, 1, 1]);
  });

  it('2D int32 dtype', () => {
    const a = Array2D.ones([3, 2], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3, 2]);
    test_util.expectArraysEqual(a, [1, 1, 1, 1, 1, 1]);
  });

  it('2D bool dtype', () => {
    const a = Array2D.ones([3, 2], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([3, 2]);
    test_util.expectArraysEqual(a, [1, 1, 1, 1, 1, 1]);
  });

  it('3D default dtype', () => {
    const a = Array3D.ones([2, 2, 2]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 2]);
    test_util.expectArraysClose(a, [1, 1, 1, 1, 1, 1, 1, 1]);
  });

  it('3D float32 dtype', () => {
    const a = Array3D.ones([2, 2, 2], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 2]);
    test_util.expectArraysClose(a, [1, 1, 1, 1, 1, 1, 1, 1]);
  });

  it('3D int32 dtype', () => {
    const a = Array3D.ones([2, 2, 2], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 2]);
    test_util.expectArraysEqual(a, [1, 1, 1, 1, 1, 1, 1, 1]);
  });

  it('3D bool dtype', () => {
    const a = Array3D.ones([2, 2, 2], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([2, 2, 2]);
    test_util.expectArraysEqual(a, [1, 1, 1, 1, 1, 1, 1, 1]);
  });

  it('4D default dtype', () => {
    const a = Array4D.ones([3, 2, 1, 1]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    test_util.expectArraysClose(a, [1, 1, 1, 1, 1, 1]);
  });

  it('4D float32 dtype', () => {
    const a = Array4D.ones([3, 2, 1, 1], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    test_util.expectArraysClose(a, [1, 1, 1, 1, 1, 1]);
  });

  it('4D int32 dtype', () => {
    const a = Array4D.ones([3, 2, 1, 1], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    test_util.expectArraysEqual(a, [1, 1, 1, 1, 1, 1]);
  });

  it('4D bool dtype', () => {
    const a = Array4D.ones([3, 2, 1, 1], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    test_util.expectArraysEqual(a, [1, 1, 1, 1, 1, 1]);
  });
};
const testsZerosLike: MathTests = it => {
  it('1D default dtype', () => {
    const a = Array1D.new([1, 2, 3]);
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    test_util.expectArraysClose(b, [0, 0, 0]);
  });

  it('1D float32 dtype', () => {
    const a = Array1D.new([1, 2, 3], 'float32');
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    test_util.expectArraysClose(b, [0, 0, 0]);
  });

  it('1D int32 dtype', () => {
    const a = Array1D.new([1, 2, 3], 'int32');
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([3]);
    test_util.expectArraysEqual(b, [0, 0, 0]);
  });

  it('1D bool dtype', () => {
    const a = Array1D.new([1, 2, 3], 'bool');
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([3]);
    test_util.expectArraysEqual(b, [0, 0, 0]);
  });

  it('2D default dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4]);
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    test_util.expectArraysClose(b, [0, 0, 0, 0]);
  });

  it('2D float32 dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4], 'float32');
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    test_util.expectArraysClose(b, [0, 0, 0, 0]);
  });

  it('2D int32 dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4], 'int32');
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2]);
    test_util.expectArraysEqual(b, [0, 0, 0, 0]);
  });

  it('2D bool dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4], 'bool');
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2]);
    test_util.expectArraysEqual(b, [0, 0, 0, 0]);
  });

  it('3D default dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    test_util.expectArraysClose(b, [0, 0, 0, 0]);
  });

  it('3D float32 dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4], 'float32');
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    test_util.expectArraysClose(b, [0, 0, 0, 0]);
  });

  it('3D int32 dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4], 'int32');
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1]);
    test_util.expectArraysEqual(b, [0, 0, 0, 0]);
  });

  it('3D bool dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4], 'bool');
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1]);
    test_util.expectArraysEqual(b, [0, 0, 0, 0]);
  });

  it('4D default dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4]);
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysClose(b, [0, 0, 0, 0]);
  });

  it('4D float32 dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'float32');
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysClose(b, [0, 0, 0, 0]);
  });

  it('4D int32 dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'int32');
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysEqual(b, [0, 0, 0, 0]);
  });

  it('4D bool dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'bool');
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysEqual(b, [0, 0, 0, 0]);
  });
};
const testsOnesLike: MathTests = it => {
  it('1D default dtype', () => {
    const a = Array1D.new([1, 2, 3]);
    const b = NDArray.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    test_util.expectArraysClose(b, [1, 1, 1]);
  });

  it('1D float32 dtype', () => {
    const a = Array1D.new([1, 2, 3], 'float32');
    const b = NDArray.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    test_util.expectArraysClose(b, [1, 1, 1]);
  });

  it('1D int32 dtype', () => {
    const a = Array1D.new([1, 2, 3], 'int32');
    const b = NDArray.onesLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([3]);
    test_util.expectArraysEqual(b, [1, 1, 1]);
  });

  it('1D bool dtype', () => {
    const a = Array1D.new([1, 2, 3], 'bool');
    const b = NDArray.onesLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([3]);
    test_util.expectArraysEqual(b, [1, 1, 1]);
  });

  it('2D default dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4]);
    const b = NDArray.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    test_util.expectArraysClose(b, [1, 1, 1, 1]);
  });

  it('2D float32 dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4], 'float32');
    const b = NDArray.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    test_util.expectArraysClose(b, [1, 1, 1, 1]);
  });

  it('2D int32 dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4], 'int32');
    const b = NDArray.onesLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2]);
    test_util.expectArraysEqual(b, [1, 1, 1, 1]);
  });

  it('2D bool dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4], 'bool');
    const b = NDArray.onesLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2]);
    test_util.expectArraysEqual(b, [1, 1, 1, 1]);
  });

  it('3D default dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
    const b = NDArray.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    test_util.expectArraysClose(b, [1, 1, 1, 1]);
  });

  it('3D float32 dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4], 'float32');
    const b = NDArray.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    test_util.expectArraysClose(b, [1, 1, 1, 1]);
  });

  it('3D int32 dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4], 'int32');
    const b = NDArray.onesLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1]);
    test_util.expectArraysEqual(b, [1, 1, 1, 1]);
  });

  it('3D bool dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4], 'bool');
    const b = NDArray.onesLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1]);
    test_util.expectArraysEqual(b, [1, 1, 1, 1]);
  });

  it('4D default dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4]);
    const b = NDArray.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysClose(b, [1, 1, 1, 1]);
  });

  it('4D float32 dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'float32');
    const b = NDArray.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysClose(b, [1, 1, 1, 1]);
  });

  it('4D int32 dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'int32');
    const b = NDArray.onesLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysEqual(b, [1, 1, 1, 1]);
  });

  it('4D bool dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'bool');
    const b = NDArray.onesLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysEqual(b, [1, 1, 1, 1]);
  });
};

const testsFill: MathTests = it => {
  it('1D fill', () => {
    const a = Array1D.zeros([3]);
    a.fill(2);

    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    test_util.expectArraysClose(a, [2, 2, 2]);
  });

  it('2D fill', () => {
    const a = Array2D.zeros([3, 2]);
    a.fill(2);

    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2]);
    test_util.expectArraysClose(a, [2, 2, 2, 2, 2, 2]);
  });

  it('3D fill', () => {
    const a = Array2D.zeros([3, 2, 1]);
    a.fill(2);

    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2, 1]);
    test_util.expectArraysClose(a, [2, 2, 2, 2, 2, 2]);
  });

  it('4D fill', () => {
    const a = Array2D.zeros([3, 2, 1, 2]);
    a.fill(2);

    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2, 1, 2]);
    test_util.expectArraysClose(a, [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]);
  });
};

const testsLike: MathTests = it => {
  it('1D default dtype', () => {
    const a = Array1D.new([1, 2, 3]);
    const b = NDArray.like(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    test_util.expectArraysClose(b, [1, 2, 3]);
  });

  it('1D float32 dtype', () => {
    const a = Array1D.new([1, 2, 3], 'float32');
    const b = NDArray.like(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    test_util.expectArraysClose(b, [1, 2, 3]);
  });

  it('1D int32 dtype', () => {
    const a = Array1D.new([1, 2, 3], 'int32');
    const b = NDArray.like(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([3]);
    test_util.expectArraysEqual(b, [1, 2, 3]);
  });

  it('1D bool dtype', () => {
    const a = Array1D.new([1, 2, 3], 'bool');
    const b = NDArray.like(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([3]);
    test_util.expectArraysEqual(b, [1, 1, 1]);
  });

  it('2D default dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4]);
    const b = NDArray.like(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    test_util.expectArraysClose(b, [1, 2, 3, 4]);
  });

  it('2D float32 dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4], 'float32');
    const b = NDArray.like(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    test_util.expectArraysClose(b, [1, 2, 3, 4]);
  });

  it('2D int32 dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4], 'int32');
    const b = NDArray.like(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2]);
    test_util.expectArraysEqual(b, [1, 2, 3, 4]);
  });

  it('2D bool dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4], 'bool');
    const b = NDArray.like(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2]);
    test_util.expectArraysEqual(b, [1, 1, 1, 1]);
  });

  it('3D default dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
    const b = NDArray.like(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    test_util.expectArraysClose(b, [1, 2, 3, 4]);
  });

  it('3D float32 dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4], 'float32');
    const b = NDArray.like(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    test_util.expectArraysClose(b, [1, 2, 3, 4]);
  });

  it('3D int32 dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4], 'int32');
    const b = NDArray.like(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1]);
    test_util.expectArraysEqual(b, [1, 2, 3, 4]);
  });

  it('3D bool dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4], 'bool');
    const b = NDArray.like(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1]);
    test_util.expectArraysEqual(b, [1, 1, 1, 1]);
  });

  it('4D default dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4]);
    const b = NDArray.like(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysClose(b, [1, 2, 3, 4]);
  });

  it('4D float32 dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'float32');
    const b = NDArray.like(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysClose(b, [1, 2, 3, 4]);
  });

  it('4D int32 dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'int32');
    const b = NDArray.like(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysEqual(b, [1, 2, 3, 4]);
  });

  it('4D bool dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'bool');
    const b = NDArray.like(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysEqual(b, [1, 1, 1, 1]);
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
const testsArray1DNew: MathTests = it => {
  it('default dtype', () => {
    const a = Array1D.new([1, 2, 3]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    test_util.expectArraysClose(a, [1, 2, 3]);
  });

  it('float32 dtype', () => {
    const a = Array1D.new([1, 2, 3], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    test_util.expectArraysClose(a, [1, 2, 3]);
  });

  it('int32 dtype', () => {
    const a = Array1D.new([1, 2, 3], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3]);
    test_util.expectArraysEqual(a, [1, 2, 3]);
  });

  it('int32 dtype, non-ints get floored, like numpy', () => {
    const a = Array1D.new([1.1, 2.5, 3.9], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3]);
    test_util.expectArraysEqual(a, [1, 2, 3]);
  });

  it('int32 dtype, negative non-ints get ceiled, like numpy', () => {
    const a = Array1D.new([-1.1, -2.5, -3.9], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3]);
    test_util.expectArraysEqual(a, [-1, -2, -3]);
  });

  it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', () => {
    const a = Array1D.new([1, -2, 0, 3], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([4]);
    expect(a.get(0)).toBe(1);
    expect(a.get(1)).toBe(1);
    expect(a.get(2)).toBe(0);
    expect(a.get(3)).toBe(1);
  });

  it('default dtype from boolean[]', () => {
    const a = Array1D.new([false, false, true]);
    expect(a.dtype).toBe('float32');
    test_util.expectArraysClose(a, [0, 0, 1]);
  });

  it('int32 dtype from boolean[]', () => {
    const a = Array1D.new([false, false, true], 'int32');
    expect(a.dtype).toBe('int32');
    test_util.expectArraysEqual(a, [0, 0, 1]);
  });

  it('bool dtype from boolean[]', () => {
    const a = Array1D.new([false, false, true], 'bool');
    expect(a.dtype).toBe('bool');
    test_util.expectArraysEqual(a, [0, 0, 1]);
  });
};
const testsArray2DNew: MathTests = it => {
  it('default dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2]);
    test_util.expectArraysClose(a, [1, 2, 3, 4]);
  });

  it('float32 dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2]);
    test_util.expectArraysClose(a, [1, 2, 3, 4]);
  });

  it('int32 dtype', () => {
    const a = Array2D.new([2, 2], [[1, 2], [3, 4]], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2]);
    test_util.expectArraysEqual(a, [1, 2, 3, 4]);
  });

  it('int32 dtype, non-ints get floored, like numpy', () => {
    const a = Array2D.new([2, 2], [1.1, 2.5, 3.9, 4.0], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2]);
    test_util.expectArraysEqual(a, [1, 2, 3, 4]);
  });

  it('int32 dtype, negative non-ints get ceiled, like numpy', () => {
    const a = Array2D.new([2, 2], [-1.1, -2.5, -3.9, -4.0], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2]);
    test_util.expectArraysEqual(a, [-1, -2, -3, -4]);
  });

  it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', () => {
    const a = Array2D.new([2, 2], [1, -2, 0, 3], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([2, 2]);
    expect(a.get(0, 0)).toBe(1);
    expect(a.get(0, 1)).toBe(1);
    expect(a.get(1, 0)).toBe(0);
    expect(a.get(1, 1)).toBe(1);
  });

  it('default dtype from boolean[]', () => {
    const a = Array2D.new([2, 2], [[false, false], [true, false]]);
    expect(a.dtype).toBe('float32');
    test_util.expectArraysClose(a, [0, 0, 1, 0]);
  });

  it('int32 dtype from boolean[]', () => {
    const a = Array2D.new([2, 2], [[false, false], [true, false]], 'int32');
    expect(a.dtype).toBe('int32');
    test_util.expectArraysEqual(a, [0, 0, 1, 0]);
  });

  it('bool dtype from boolean[]', () => {
    const a = Array2D.new([2, 2], [[false, false], [true, false]], 'bool');
    expect(a.dtype).toBe('bool');
    test_util.expectArraysEqual(a, [0, 0, 1, 0]);
  });
};
const testsArray3DNew: MathTests = it => {
  it('default dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 1]);
    test_util.expectArraysClose(a, [1, 2, 3, 4]);
  });

  it('float32 dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 1]);
    test_util.expectArraysClose(a, [1, 2, 3, 4]);
  });

  it('int32 dtype', () => {
    const a = Array3D.new([2, 2, 1], [[[1], [2]], [[3], [4]]], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 1]);
    test_util.expectArraysEqual(a, [1, 2, 3, 4]);
  });

  it('int32 dtype, non-ints get floored, like numpy', () => {
    const a = Array3D.new([2, 2, 1], [1.1, 2.5, 3.9, 4.0], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 1]);
    test_util.expectArraysEqual(a, [1, 2, 3, 4]);
  });

  it('int32 dtype, negative non-ints get ceiled, like numpy', () => {
    const a = Array3D.new([2, 2, 1], [-1.1, -2.5, -3.9, -4.0], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 1]);
    test_util.expectArraysEqual(a, [-1, -2, -3, -4]);
  });

  it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', () => {
    const a = Array3D.new([2, 2, 1], [1, -2, 0, 3], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([2, 2, 1]);
    expect(a.get(0, 0, 0)).toBe(1);
    expect(a.get(0, 1, 0)).toBe(1);
    expect(a.get(1, 0, 0)).toBe(0);
    expect(a.get(1, 1, 0)).toBe(1);
  });

  it('default dtype from boolean[]', () => {
    const a = Array3D.new([2, 2, 1], [[[false], [false]], [[true], [false]]]);
    expect(a.dtype).toBe('float32');
    test_util.expectArraysClose(a, [0, 0, 1, 0]);
  });

  it('int32 dtype from boolean[]', () => {
    const a = Array3D.new(
        [2, 2, 1], [[[false], [false]], [[true], [false]]], 'int32');
    expect(a.dtype).toBe('int32');
    test_util.expectArraysEqual(a, [0, 0, 1, 0]);
  });

  it('bool dtype from boolean[]', () => {
    const a =
        Array3D.new([2, 2, 1], [[[false], [false]], [[true], [false]]], 'bool');
    expect(a.dtype).toBe('bool');
    test_util.expectArraysEqual(a, [0, 0, 1, 0]);
  });
};
const testsArray4DNew: MathTests = it => {
  it('default dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysClose(a, [1, 2, 3, 4]);
  });

  it('float32 dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysClose(a, [1, 2, 3, 4]);
  });

  it('int32 dtype', () => {
    const a =
        Array4D.new([2, 2, 1, 1], [[[[1]], [[2]]], [[[3]], [[4]]]], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysEqual(a, [1, 2, 3, 4]);
  });

  it('int32 dtype, non-ints get floored, like numpy', () => {
    const a = Array4D.new([2, 2, 1, 1], [1.1, 2.5, 3.9, 4.0], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysEqual(a, [1, 2, 3, 4]);
  });

  it('int32 dtype, negative non-ints get ceiled, like numpy', () => {
    const a = Array4D.new([2, 2, 1, 1], [-1.1, -2.5, -3.9, -4.0], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysEqual(a, [-1, -2, -3, -4]);
  });

  it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, -2, 0, 3], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    expect(a.get(0, 0, 0, 0)).toBe(1);
    expect(a.get(0, 1, 0, 0)).toBe(1);
    expect(a.get(1, 0, 0, 0)).toBe(0);
    expect(a.get(1, 1, 0, 0)).toBe(1);
  });

  it('default dtype from boolean[]', () => {
    const a =
        Array4D.new([1, 2, 2, 1], [[[[false], [false]], [[true], [false]]]]);
    expect(a.dtype).toBe('float32');
    test_util.expectArraysClose(a, [0, 0, 1, 0]);
  });

  it('int32 dtype from boolean[]', () => {
    const a = Array4D.new(
        [1, 2, 2, 1], [[[[false], [false]], [[true], [false]]]], 'int32');
    expect(a.dtype).toBe('int32');
    test_util.expectArraysEqual(a, [0, 0, 1, 0]);
  });

  it('bool dtype from boolean[]', () => {
    const a = Array4D.new(
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

  it('Array1D default dtype', () => {
    const a = Array1D.new([1, 2, 3, 4]);
    const b = a.reshape([2, 2]);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
  });

  it('Array1D int32 dtype', () => {
    const a = Array1D.new([1, 2, 3, 4], 'int32');
    const b = a.reshape([2, 2]);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2]);
  });

  it('Array2D default dtype', () => {
    const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    const b = a.reshape([6]);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([6]);
  });

  it('Array2D bool dtype', () => {
    const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6], 'bool');
    const b = a.reshape([6]);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([6]);
  });

  it('Array3D default dtype', () => {
    const a = Array3D.new([2, 3, 1], [1, 2, 3, 4, 5, 6]);
    const b = a.reshape([6]);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([6]);
  });

  it('Array3D bool dtype', () => {
    const a = Array3D.new([2, 3, 1], [1, 2, 3, 4, 5, 6], 'bool');
    const b = a.reshape([6]);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([6]);
  });

  it('Array4D default dtype', () => {
    const a = Array4D.new([2, 3, 1, 1], [1, 2, 3, 4, 5, 6]);
    const b = a.reshape([2, 3]);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 3]);
  });

  it('Array4D int32 dtype', () => {
    const a = Array4D.new([2, 3, 1, 1], [1, 2, 3, 4, 5, 6], 'int32');
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
    const a = Scalar.new(true, 'bool').asType('int32');
    expect(a.dtype).toBe('int32');
    expect(a.get()).toBe(1);
  });

  it('array1d float32 -> int32', () => {
    const a = Array1D.new([1.1, 3.9, -2.9, 0]).asType('int32');
    expect(a.dtype).toBe('int32');
    test_util.expectArraysEqual(a, [1, 3, -2, 0]);
  });

  it('array2d float32 -> bool', () => {
    const a = Array2D.new([2, 2], [1.1, 3.9, -2.9, 0]).asType(DType.bool);
    expect(a.dtype).toBe('bool');
    expect(a.get(0, 0)).toBe(1);
    expect(a.get(0, 1)).toBe(1);
    expect(a.get(1, 0)).toBe(1);
    expect(a.get(1, 1)).toBe(0);
  });

  it('array2d int32 -> bool', () => {
    const a = Array2D.new([2, 2], [1, 3, 0, -1], 'int32').asType('bool');
    expect(a.dtype).toBe('bool');
    expect(a.get(0, 0)).toBe(1);
    expect(a.get(0, 1)).toBe(1);
    expect(a.get(1, 0)).toBe(0);
    expect(a.get(1, 1)).toBe(1);
  });

  it('array3d bool -> float32', () => {
    const a = Array3D.new([2, 2, 1], [true, false, false, true], 'bool')
                  .asType('float32');
    expect(a.dtype).toBe('float32');
    test_util.expectArraysClose(a, [1, 0, 0, 1]);
  });

  it('bool CPU -> GPU -> CPU', () => {
    const a = Array1D.new([1, 2, 0, 0, 5], 'bool');
    test_util.expectArraysEqual(a, [1, 1, 0, 0, 1]);
  });

  it('int32 CPU -> GPU -> CPU', () => {
    const a = Array1D.new([1, 2, 0, 0, 5], 'int32');
    test_util.expectArraysEqual(a, [1, 2, 0, 0, 5]);
  });

  it('asType is functional', math => {
    const a = Scalar.new(2.4, 'float32');
    const b = a.asType('float32');
    expect(a.id).not.toBe(b.id);
    b.dispose();
    test_util.expectArraysClose(a, [2.4]);
  });
};

const testSqueeze: MathTests = it => {
  it('squeeze no axis', () => {
    const a = Array2D.new([3, 1], [4, 2, 1], 'bool');
    const b = a.squeeze();
    expect(b.shape).toEqual([3]);
  });

  it('squeeze with axis', () => {
    const a = Array3D.new([3, 1, 1], [4, 2, 1], 'bool');
    const b = a.squeeze([1]);
    expect(b.shape).toEqual([3, 1]);
  });

  it('squeeze wrong axis', () => {
    const a = Array3D.new([3, 1, 1], [4, 2, 1], 'bool');
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
    const a = Array1D.new([4, 2, 1], 'bool');
    const b = a.as2D(3, 1);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([3, 1]);
  });

  it('2d -> 4d', () => {
    const a = Array2D.new([2, 2], [4, 2, 1, 3]);
    const b = a.as4D(1, 1, 2, 2);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([1, 1, 2, 2]);
  });

  it('3d -> 2d', () => {
    const a = Array3D.new([2, 2, 1], [4, 2, 1, 3], 'float32');
    const b = a.as2D(2, 2);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
  });

  it('4d -> 1d', () => {
    const a = Array4D.new([2, 2, 1, 1], [4, 2, 1, 3], 'bool');
    const b = a.as1D();
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([4]);
  });
};
const testsRand: MathTests = it => {
  it('should return a random 1D float32 array', () => {
    const shape: [number] = [10];

    // Enusre defaults to float32 w/o type:
    let result = NDArray.rand(shape, () => util.randUniform(0, 2));
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 2);

    result = NDArray.rand(shape, () => util.randUniform(0, 1.5));
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 1D int32 array', () => {
    const shape: [number] = [10];
    const result = NDArray.rand(shape, () => util.randUniform(0, 2), 'int32');
    expect(result.dtype).toBe('int32');
    test_util.expectValuesInRange(result, 0, 2);
  });

  it('should return a random 1D bool array', () => {
    const shape: [number] = [10];
    const result = NDArray.rand(shape, () => util.randUniform(0, 1), 'bool');
    expect(result.dtype).toBe('bool');
    test_util.expectValuesInRange(result, 0, 1);
  });

  it('should return a random 2D float32 array', () => {
    const shape: [number] = [3, 4];

    // Enusre defaults to float32 w/o type:
    let result = NDArray.rand(shape, () => util.randUniform(0, 2.5));
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 2.5);

    result = NDArray.rand(shape, () => util.randUniform(0, 1.5), 'float32');
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 2D int32 array', () => {
    const shape: [number] = [3, 4];
    const result = NDArray.rand(shape, () => util.randUniform(0, 2), 'int32');
    expect(result.dtype).toBe('int32');
    test_util.expectValuesInRange(result, 0, 2);
  });

  it('should return a random 2D bool array', () => {
    const shape: [number] = [3, 4];
    const result = NDArray.rand(shape, () => util.randUniform(0, 1), 'bool');
    expect(result.dtype).toBe('bool');
    test_util.expectValuesInRange(result, 0, 1);
  });

  it('should return a random 3D float32 array', () => {
    const shape: [number] = [3, 4, 5];

    // Enusre defaults to float32 w/o type:
    let result = NDArray.rand(shape, () => util.randUniform(0, 2.5));
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 2.5);

    result = NDArray.rand(shape, () => util.randUniform(0, 1.5), 'float32');
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 3D int32 array', () => {
    const shape: [number] = [3, 4, 5];
    const result = NDArray.rand(shape, () => util.randUniform(0, 2), 'int32');
    expect(result.dtype).toBe('int32');
    test_util.expectValuesInRange(result, 0, 2);
  });

  it('should return a random 3D bool array', () => {
    const shape: [number] = [3, 4, 5];
    const result = NDArray.rand(shape, () => util.randUniform(0, 1), 'bool');
    expect(result.dtype).toBe('bool');
    test_util.expectValuesInRange(result, 0, 1);
  });

  it('should return a random 4D float32 array', () => {
    const shape: [number] = [3, 4, 5, 6];

    // Enusre defaults to float32 w/o type:
    let result = NDArray.rand(shape, () => util.randUniform(0, 2.5));
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 2.5);

    result = NDArray.rand(shape, () => util.randUniform(0, 1.5));
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 4D int32 array', () => {
    const shape: [number] = [3, 4, 5, 6];
    const result = NDArray.rand(shape, () => util.randUniform(0, 2), 'int32');
    expect(result.dtype).toBe('int32');
    test_util.expectValuesInRange(result, 0, 2);
  });

  it('should return a random 4D bool array', () => {
    const shape: [number] = [3, 4, 5, 6];
    const result = NDArray.rand(shape, () => util.randUniform(0, 1), 'bool');
    expect(result.dtype).toBe('bool');
    test_util.expectValuesInRange(result, 0, 1);
  });
};
const testsRandNormal: MathTests = it => {
  const SEED = 2002;
  const EPSILON = 0.05;

  it('should return a float32 1D of random normal values', () => {
    const SAMPLES = 10000;

    // Ensure defaults to float32.
    let result = NDArray.randNormal([SAMPLES], 0, 0.5, null, SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES]);
    test_util.jarqueBeraNormalityTest(result);
    test_util.expectArrayInMeanStdRange(result, 0, 0.5, EPSILON);

    result = NDArray.randNormal([SAMPLES], 0, 1.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES]);
    test_util.jarqueBeraNormalityTest(result);
    test_util.expectArrayInMeanStdRange(result, 0, 1.5, EPSILON);
  });

  it('should return a int32 1D of random normal values', () => {
    const SAMPLES = 10000;
    const result = NDArray.randNormal([SAMPLES], 0, 2, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual([SAMPLES]);
    test_util.jarqueBeraNormalityTest(result);
    test_util.expectArrayInMeanStdRange(result, 0, 2, EPSILON);
  });

  it('should return a float32 2D of random normal values', () => {
    const SAMPLES = 250;

    // Ensure defaults to float32.
    let result = Array2D.randNormal([SAMPLES, SAMPLES], 0, 2.5, null, SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES]);
    test_util.jarqueBeraNormalityTest(result);
    test_util.expectArrayInMeanStdRange(result, 0, 2.5, EPSILON);

    result = Array2D.randNormal([SAMPLES, SAMPLES], 0, 3.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES]);
    test_util.jarqueBeraNormalityTest(result);
    test_util.expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);
  });

  it('should return a int32 2D of random normal values', () => {
    const SAMPLES = 100;
    const result = Array2D.randNormal([SAMPLES, SAMPLES], 0, 2, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES]);
    test_util.jarqueBeraNormalityTest(result);
    test_util.expectArrayInMeanStdRange(result, 0, 2, EPSILON);
  });

  it('should return a float32 3D of random normal values', () => {
    const SAMPLES = 50;

    // Ensure defaults to float32.
    let result =
        Array3D.randNormal([SAMPLES, SAMPLES, SAMPLES], 0, 0.5, null, SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES, SAMPLES]);
    test_util.jarqueBeraNormalityTest(result);
    test_util.expectArrayInMeanStdRange(result, 0, 0.5, EPSILON);

    result = Array3D.randNormal(
        [SAMPLES, SAMPLES, SAMPLES], 0, 1.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES, SAMPLES]);
    test_util.jarqueBeraNormalityTest(result);
    test_util.expectArrayInMeanStdRange(result, 0, 1.5, EPSILON);
  });

  it('should return a int32 3D of random normal values', () => {
    const SAMPLES = 50;
    const result =
        Array3D.randNormal([SAMPLES, SAMPLES, SAMPLES], 0, 2, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES, SAMPLES]);
    test_util.jarqueBeraNormalityTest(result);
    test_util.expectArrayInMeanStdRange(result, 0, 2, EPSILON);
  });

  it('should return a float32 4D of random normal values', () => {
    const SAMPLES = 25;

    // Ensure defaults to float32.
    let result = Array4D.randNormal(
        [SAMPLES, SAMPLES, SAMPLES, SAMPLES], 0, 0.5, null, SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES, SAMPLES, SAMPLES]);
    test_util.jarqueBeraNormalityTest(result);
    test_util.expectArrayInMeanStdRange(result, 0, 0.5, EPSILON);

    result = Array4D.randNormal(
        [SAMPLES, SAMPLES, SAMPLES, SAMPLES], 0, 1.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES, SAMPLES, SAMPLES]);
    test_util.jarqueBeraNormalityTest(result);
    test_util.expectArrayInMeanStdRange(result, 0, 1.5, EPSILON);
  });

  it('should return a int32 4D of random normal values', () => {
    const SAMPLES = 25;

    const result = Array4D.randNormal(
        [SAMPLES, SAMPLES, SAMPLES, SAMPLES], 0, 2, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES, SAMPLES, SAMPLES]);
    test_util.jarqueBeraNormalityTest(result);
    test_util.expectArrayInMeanStdRange(result, 0, 2, EPSILON);
  });
};
const testsRandTruncNormal: MathTests = it => {
  // Expect slightly higher variances for truncated values.
  const EPSILON = 0.60;
  const SEED = 2002;

  function assertTruncatedValues(array: NDArray, mean: number, stdv: number) {
    const bounds = mean + stdv * 2;
    const values = array.dataSync();
    for (let i = 0; i < values.length; i++) {
      expect(Math.abs(values[i])).toBeLessThanOrEqual(bounds);
    }
  }

  it('should return a random 1D float32 array', () => {
    const shape: [number] = [1000];

    // Ensure defaults to float32 w/o type:
    let result = NDArray.randTruncatedNormal(shape, 0, 3.5, null, SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 3.5);
    test_util.expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);

    result = NDArray.randTruncatedNormal(shape, 0, 4.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 4.5);
    test_util.expectArrayInMeanStdRange(result, 0, 4.5, EPSILON);
  });

  it('should return a randon 1D int32 array', () => {
    const shape: [number] = [1000];
    const result = NDArray.randTruncatedNormal(shape, 0, 5, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    assertTruncatedValues(result, 0, 5);
    test_util.expectArrayInMeanStdRange(result, 0, 5, EPSILON);
  });

  it('should return a 2D float32 array', () => {
    const shape: [number, number] = [50, 50];

    // Ensure defaults to float32 w/o type:
    let result = Array2D.randTruncatedNormal(shape, 0, 3.5, null, SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 3.5);
    test_util.expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);

    result = Array2D.randTruncatedNormal(shape, 0, 4.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 4.5);
    test_util.expectArrayInMeanStdRange(result, 0, 4.5, EPSILON);
  });

  it('should return a 2D int32 array', () => {
    const shape: [number, number] = [50, 50];
    const result = Array2D.randTruncatedNormal(shape, 0, 5, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    assertTruncatedValues(result, 0, 5);
    test_util.expectArrayInMeanStdRange(result, 0, 5, EPSILON);
  });

  it('should return a 3D float32 array', () => {
    const shape: [number, number, number] = [10, 10, 10];

    // Ensure defaults to float32 w/o type:
    let result = Array3D.randTruncatedNormal(shape, 0, 3.5, null, SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 3.5);
    test_util.expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);

    result = Array3D.randTruncatedNormal(shape, 0, 4.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 4.5);
    test_util.expectArrayInMeanStdRange(result, 0, 4.5, EPSILON);
  });

  it('should return a 3D int32 array', () => {
    const shape: [number, number, number] = [10, 10, 10];
    const result = Array3D.randTruncatedNormal(shape, 0, 5, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    assertTruncatedValues(result, 0, 5);
    test_util.expectArrayInMeanStdRange(result, 0, 5, EPSILON);
  });

  it('should return a 4D float32 array', () => {
    const shape: [number, number, number, number] = [5, 5, 5, 5];

    // Ensure defaults to float32 w/o type:
    let result = Array4D.randTruncatedNormal(shape, 0, 3.5, null, SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 3.5);
    test_util.expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);

    result = Array4D.randTruncatedNormal(shape, 0, 4.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 4.5);
    test_util.expectArrayInMeanStdRange(result, 0, 4.5, EPSILON);
  });

  it('should return a 4D int32 array', () => {
    const shape: [number, number, number, number] = [5, 5, 5, 5];
    const result = Array4D.randTruncatedNormal(shape, 0, 5, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    assertTruncatedValues(result, 0, 5);
    test_util.expectArrayInMeanStdRange(result, 0, 5, EPSILON);
  });
};
const testsRandUniform: MathTests = it => {
  it('should return a random 1D float32 array', () => {
    const shape: [number] = [10];

    // Enusre defaults to float32 w/o type:
    let result = NDArray.randUniform(shape, 0, 2.5);
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 2.5);

    result = NDArray.randUniform(shape, 0, 1.5, 'float32');
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 1D int32 array', () => {
    const shape: [number] = [10];
    const result = NDArray.randUniform(shape, 0, 2, 'int32');
    expect(result.dtype).toBe('int32');
    test_util.expectValuesInRange(result, 0, 2);
  });

  it('should return a random 1D bool array', () => {
    const shape: [number] = [10];
    const result = NDArray.randUniform(shape, 0, 1, 'bool');
    expect(result.dtype).toBe('bool');
    test_util.expectValuesInRange(result, 0, 1);
  });

  it('should return a random 2D float32 array', () => {
    const shape: [number, number] = [3, 4];

    // Enusre defaults to float32 w/o type:
    let result = Array2D.randUniform(shape, 0, 2.5);
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 2.5);

    result = Array2D.randUniform(shape, 0, 1.5, 'float32');
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 2D int32 array', () => {
    const shape: [number, number] = [3, 4];
    const result = Array2D.randUniform(shape, 0, 2, 'int32');
    expect(result.dtype).toBe('int32');
    test_util.expectValuesInRange(result, 0, 2);
  });

  it('should return a random 2D bool array', () => {
    const shape: [number, number] = [3, 4];
    const result = Array2D.randUniform(shape, 0, 1, 'bool');
    expect(result.dtype).toBe('bool');
    test_util.expectValuesInRange(result, 0, 1);
  });

  it('should return a random 3D float32 array', () => {
    const shape: [number, number, number] = [3, 4, 5];

    // Enusre defaults to float32 w/o type:
    let result = Array3D.randUniform(shape, 0, 2.5);
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 2.5);

    result = Array3D.randUniform(shape, 0, 1.5, 'float32');
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 3D int32 array', () => {
    const shape: [number, number, number] = [3, 4, 5];
    const result = Array3D.randUniform(shape, 0, 2, 'int32');
    expect(result.dtype).toBe('int32');
    test_util.expectValuesInRange(result, 0, 2);
  });

  it('should return a random 3D bool array', () => {
    const shape: [number, number, number] = [3, 4, 5];
    const result = Array3D.randUniform(shape, 0, 1, 'bool');
    expect(result.dtype).toBe('bool');
    test_util.expectValuesInRange(result, 0, 1);
  });

  it('should return a random 4D float32 array', () => {
    const shape: [number, number, number, number] = [3, 4, 5, 6];

    // Enusre defaults to float32 w/o type:
    let result = Array4D.randUniform(shape, 0, 2.5);
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 2.5);

    result = Array4D.randUniform(shape, 0, 1.5, 'float32');
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 4D int32 array', () => {
    const shape: [number, number, number, number] = [3, 4, 5, 6];
    const result = Array4D.randUniform(shape, 0, 2, 'int32');
    expect(result.dtype).toBe('int32');
    test_util.expectValuesInRange(result, 0, 2);
  });

  it('should return a random 4D bool array', () => {
    const shape: [number, number, number, number] = [3, 4, 5, 6];
    const result = Array4D.randUniform(shape, 0, 1, 'bool');
    expect(result.dtype).toBe('bool');
    test_util.expectValuesInRange(result, 0, 1);
  });
};
const testsFromPixels: MathTests = it => {
  beforeEach(() => {});

  afterEach(() => {});

  it('ImageData 1x1x3', () => {
    const pixels = new ImageData(1, 1);
    pixels.data[0] = 0;
    pixels.data[1] = 80;
    pixels.data[2] = 160;
    pixels.data[3] = 240;

    const array = Array3D.fromPixels(pixels, 3);

    test_util.expectArraysEqual(array, [0, 80, 160]);
  });

  it('ImageData 1x1x4', () => {
    const pixels = new ImageData(1, 1);
    pixels.data[0] = 0;
    pixels.data[1] = 80;
    pixels.data[2] = 160;
    pixels.data[3] = 240;

    const array = Array3D.fromPixels(pixels, 4);

    test_util.expectArraysEqual(array, [0, 80, 160, 240]);
  });

  it('ImageData 2x2x3', () => {
    const pixels = new ImageData(2, 2);

    for (let i = 0; i < 8; i++) {
      pixels.data[i] = i * 2;
    }
    for (let i = 8; i < 16; i++) {
      pixels.data[i] = i * 2;
    }

    const array = Array3D.fromPixels(pixels, 3);

    test_util.expectArraysEqual(
        array, [0, 2, 4, 8, 10, 12, 16, 18, 20, 24, 26, 28]);
  });

  it('ImageData 2x2x4', () => {
    const pixels = new ImageData(2, 2);
    for (let i = 0; i < 8; i++) {
      pixels.data[i] = i * 2;
    }
    for (let i = 8; i < 16; i++) {
      pixels.data[i] = i * 2;
    }

    const array = Array3D.fromPixels(pixels, 4);

    test_util.expectArraysClose(
        array,
        new Int32Array(
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]));
  });
};

const allTests = [
  tests,
  testsNew,
  testsZeros,
  testsOnes,
  testsZerosLike,
  testsOnesLike,
  testsFill,
  testsLike,
  testsScalarNew,
  testsArray1DNew,
  testsArray2DNew,
  testsArray3DNew,
  testsArray4DNew,
  testsReshape,
  testsAsType,
  testsAsXD,
  testsRand,
  testsRandNormal,
  testsRandTruncNormal,
  testsRandUniform,
  testsFromPixels,
  testSqueeze
];

test_util.describeMathCPU('NDArray', allTests);
test_util.describeMathGPU('NDArray', allTests, [
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
]);
