/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as test_util from '../test_util';
import * as util from '../util';

import {MatrixOrientation} from './math';
import {NDArrayMathCPU} from './math_cpu';
import {Array1D, Array2D, Array3D, NDArray, Scalar} from './ndarray';

describe('NDArrayMathCPU clone', () => {
  it('returns a ndarray with the same shape and data', () => {
    const math = new NDArrayMathCPU();
    const a = Array2D.new([3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const aPrime = math.clone(a);
    expect(aPrime.shape).toEqual(a.shape);
    expect(aPrime.getValues()).toEqual(a.getValues());
  });
});

describe('NDArrayMathCPU slice2D', () => {
  let math: NDArrayMathCPU;
  beforeEach(() => {
    math = new NDArrayMathCPU();
  });

  it('slicing a 1x1 from a 1x1 returns a 1x1', () => {
    const a = Array2D.new([1, 1], [0]);
    const b = math.slice2D(a, [0, 0], [1, 1]);
    expect(b.shape).toEqual([1, 1]);
  });

  it('returns a ndarray of slice size', () => {
    const a = Array2D.zeros([100, 100]);
    const b = math.slice2D(a, [0, 0], [12, 34]);
    expect(b.shape).toEqual([12, 34]);
  });

  it('returns the upper-left submatrix when begin is [0, 0]', () => {
    const a = NDArray.randUniform<Array2D>([10, 10], -1, 1);
    const b = math.slice2D(a, [0, 0], [2, 2]);
    const aValues = a.getValues();
    const expected =
        new Float32Array([aValues[0], aValues[1], aValues[10], aValues[11]]);
    test_util.expectArraysClose(b.getValues(), expected, 0);
  });

  it('returns the rectangle specified', () => {
    const a = Array2D.new([4, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const b = math.slice2D(a, [1, 1], [3, 2]);
    const expected = new Float32Array([5, 6, 8, 9, 11, 12]);
    expect(b.getValues()).toEqual(expected);
  });

  it('throws when requesting out of bounds slice', () => {
    const a = Array2D.new([4, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    expect(() => math.slice2D(a, [1, 1], [10, 10])).toThrowError();
  });
});

describe('NDArrayMathCPU copy2D', () => {
  let math: NDArrayMathCPU;
  beforeEach(() => {
    math = new NDArrayMathCPU();
  });

  it('throws an error if source and dest shapes have different areas', () => {
    const source = Array2D.zeros([100, 100]);
    const dest = Array2D.zeros([100, 100]);
    const sourceSize: [number, number] = [20, 20];
    const destSize: [number, number] = [5, 5];
    expect(
        () => math.copy2D(source, [0, 0], sourceSize, dest, [0, 0], destSize))
        .toThrowError();
  });

  it('copies a src shape into a dst shape', () => {
    const source = Array2D.new([3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const dest = Array2D.zeros([6, 2]);
    math.copy2D(source, [1, 1], [2, 3], dest, [2, 0], [3, 2]);
    expect(dest.getValues()).toEqual(new Float32Array([
      0, 0, 0, 0, 6, 7, 8, 10, 11, 12, 0, 0
    ]));
  });

  it('throws when requesting out of bounds source copy', () => {
    const source = Array2D.new([3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const dest = Array2D.zeros([6, 2]);

    expect(() => math.copy2D(source, [1, 1], [10, 10], dest, [2, 0], [
      3, 2
    ])).toThrowError();
  });

  it('throws when requesting out of bounds dest copy', () => {
    const source = Array2D.new([3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const dest = Array2D.zeros([6, 2]);

    expect(() => math.copy2D(source, [1, 1], [2, 3], dest, [2, 0], [
      3, 10
    ])).toThrowError();
  });
});

describe('NDArrayMathCPU concat3D', () => {
  let math: NDArrayMathCPU;
  beforeEach(() => {
    math = new NDArrayMathCPU();
  });

  it('shapes correct concat axis=0', () => {
    const ndarray1 = Array3D.new([1, 1, 3], [1, 2, 3]);
    const ndarray2 = Array3D.new([1, 1, 3], [4, 5, 6]);
    const values = math.concat3D(ndarray1, ndarray2, 0);
    expect(values.shape).toEqual([2, 1, 3]);
    expect(values.getValues()).toEqual(new Float32Array([1, 2, 3, 4, 5, 6]));
  });

  it('concat axis=0', () => {
    const ndarray1 = Array3D.new([1, 2, 3], [1, 11, 111, 2, 22, 222]);
    const ndarray2 = Array3D.new(
        [2, 2, 3], [5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888]);
    const values = math.concat3D(ndarray1, ndarray2, 0);
    expect(values.shape).toEqual([3, 2, 3]);
    expect(values.getValues()).toEqual(new Float32Array([
      1, 11, 111, 2, 22, 222, 5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888
    ]));
  });

  it('shapes correct concat axis=1', () => {
    const ndarray1 = Array3D.new([1, 1, 3], [1, 2, 3]);
    const ndarray2 = Array3D.new([1, 1, 3], [4, 5, 6]);
    const values = math.concat3D(ndarray1, ndarray2, 1);
    expect(values.shape).toEqual([1, 2, 3]);
    expect(values.getValues()).toEqual(new Float32Array([1, 2, 3, 4, 5, 6]));
  });

  it('concat axis=1', () => {
    const ndarray1 = Array3D.new([2, 1, 3], [1, 11, 111, 3, 33, 333]);
    const ndarray2 = Array3D.new(
        [2, 2, 3], [5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888]);
    const values = math.concat3D(ndarray1, ndarray2, 1);
    expect(values.shape).toEqual([2, 3, 3]);
    expect(values.getValues()).toEqual(new Float32Array([
      1, 11, 111, 5, 55, 555, 6, 66, 666, 3, 33, 333, 7, 77, 777, 8, 88, 888
    ]));
  });

  it('shapes correct concat axis=2', () => {
    const ndarray1 = Array3D.new([1, 1, 3], [1, 2, 3]);
    const ndarray2 = Array3D.new([1, 1, 3], [4, 5, 6]);
    const values = math.concat3D(ndarray1, ndarray2, 2);
    expect(values.shape).toEqual([1, 1, 6]);
    expect(values.getValues()).toEqual(new Float32Array([1, 2, 3, 4, 5, 6]));
  });

  it('concat axis=2', () => {
    const ndarray1 = Array3D.new([2, 2, 2], [1, 11, 2, 22, 3, 33, 4, 44]);
    const ndarray2 = Array3D.new(
        [2, 2, 3], [5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888]);
    const values = math.concat3D(ndarray1, ndarray2, 2);
    expect(values.shape).toEqual([2, 2, 5]);
    expect(values.getValues()).toEqual(new Float32Array([
      1, 11, 5, 55, 555, 2, 22, 6, 66, 666,
      3, 33, 7, 77, 777, 4, 44, 8, 88, 888
    ]));
  });

  it('concat throws when invalid non-axis shapes, axis=0', () => {
    const axis = 0;
    const x1 = Array3D.new([1, 1, 3], [1, 11, 111]);
    const x2 = Array3D.new(
        [2, 2, 3], [5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888]);
    expect(() => math.concat3D(x1, x2, axis)).toThrowError();
  });

  it('concat throws when invalid non-axis shapes, axis=1', () => {
    const axis = 1;
    const x1 = Array3D.new([1, 1, 3], [1, 11, 111]);
    const x2 = Array3D.new(
        [2, 2, 3], [5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888]);
    expect(() => math.concat3D(x1, x2, axis)).toThrowError();
  });

  it('concat throws when invalid non-axis shapes, axis=2', () => {
    const axis = 2;
    const x1 = Array3D.new([1, 2, 2], [1, 11, 2, 22]);
    const x2 = Array3D.new(
        [2, 2, 3], [5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888]);
    expect(() => math.concat3D(x1, x2, axis)).toThrowError();
  });
});

describe('NDArrayMathCPU matMul', () => {
  let math: NDArrayMathCPU;
  beforeEach(() => {
    math = new NDArrayMathCPU();
  });

  it('A x B', () => {
    const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    const b = Array2D.new([3, 2], [0, 1, -3, 2, 2, 1]);
    const c = math.matMul(a, b);
    expect(c.shape).toEqual([2, 2]);
    expect(c.getValues()).toEqual(new Float32Array([0, 8, -3, 20]));
  });

  it('A x B^t', () => {
    const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    const b = Array2D.new([2, 3], [1, 0, 2, 4, 3, 0]);
    const c = math.matMul(
        a, b, MatrixOrientation.REGULAR, MatrixOrientation.TRANSPOSED);
    const expected = new Float32Array([7, 10, 16, 31]);
    expect(c.getValues()).toEqual(expected);
  });

  it('A^t x B', () => {
    const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    const b = Array2D.new([2, 3], [1, 0, 2, 4, 3, 0]);
    const c = math.matMul(
        a, b, MatrixOrientation.TRANSPOSED, MatrixOrientation.REGULAR);
    const expected = new Float32Array([17, 12, 2, 22, 15, 4, 27, 18, 6]);
    expect(c.getValues()).toEqual(expected);
  });

  it('A^t x B^t', () => {
    const a = Array2D.new([3, 2], [1, 2, 3, 4, 5, 6]);
    const b = Array2D.new([2, 3], [1, 0, 2, 4, 3, 0]);
    const c = math.matMul(
        a, b, MatrixOrientation.TRANSPOSED, MatrixOrientation.TRANSPOSED);
    const expected = new Float32Array([11, 13, 14, 20]);
    expect(c.getValues()).toEqual(expected);
  });

  it('A x B^t shapes do not match', () => {
    const a = NDArray.zeros<Array2D>([2, 3]);
    const b = NDArray.zeros<Array2D>([3, 2]);
    const f = () => {
      math.matMul(
          a, b, MatrixOrientation.REGULAR, MatrixOrientation.TRANSPOSED);
    };
    expect(f).toThrowError();
  });

  it('A^t x B shapes do not match', () => {
    const a = NDArray.zeros<Array2D>([2, 3]);
    const b = NDArray.zeros<Array2D>([3, 2]);
    const f = () => {
      math.matMul(
          a, b, MatrixOrientation.TRANSPOSED, MatrixOrientation.REGULAR);
    };
    expect(f).toThrowError();
  });

  it('A^t x B^t shapes do not match', () => {
    const a = NDArray.zeros<Array2D>([3, 2]);
    const b = NDArray.zeros<Array2D>([3, 2]);
    const f = () => {
      math.matMul(
          a, b, MatrixOrientation.TRANSPOSED, MatrixOrientation.TRANSPOSED);
    };
    expect(f).toThrowError();
  });

  it('matmul throws when inner dimensions dont match', () => {
    const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    const b = Array2D.new([4, 2], [0, 1, -3, 2, 2, 1, 2, 2]);
    expect(() => math.matMul(a, b)).toThrowError();
  });

  it('matmul throws when passed non matrices', () => {
    // tslint:disable-next-line:no-any
    const a: any =
        Array3D.new([2, 3, 2], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const b = Array2D.new([4, 2], [0, 1, -3, 2, 2, 1, 2, 2]);
    expect(() => math.matMul(a, b)).toThrowError();
    expect(() => math.matMul(b, a)).toThrowError();
  });

  it('Vector times matrix', () => {
    const v = Array1D.new([2, 3]);
    const matrix = Array2D.new([2, 2], [1, 2, 3, 4]);
    const result = math.vectorTimesMatrix(v, matrix);

    const expected = new Float32Array([11, 16]);
    expect(result.getValues()).toEqual(expected);
  });

  it('Vector times matrix throws when not passed a vector', () => {
    // tslint:disable-next-line:no-any
    const v: any = Array2D.new([2, 2], [1, 2, 3, 4]);
    const matrix = Array2D.new([2, 2], [1, 2, 3, 4]);
    expect(() => math.vectorTimesMatrix(v, matrix)).toThrowError();
  });

  it('Vector times matrix throws when not passed a matrix', () => {
    const v = Array1D.new([2, 3]);
    // tslint:disable-next-line:no-any
    const matrix: any = Array3D.new([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
    expect(() => math.vectorTimesMatrix(v, matrix)).toThrowError();
  });

  it('Matrix times vector', () => {
    const matrix = Array2D.new([2, 2], [1, 2, 3, 4]);
    const v = Array1D.new([2, 3]);
    const result = math.matrixTimesVector(matrix, v);

    const expected = new Float32Array([8, 18]);
    expect(result.getValues()).toEqual(expected);
  });

  it('matrix times vector throws when not passed a vector', () => {
    // tslint:disable-next-line:no-any
    const v: any = Array2D.new([2, 2], [1, 2, 3, 4]);
    const matrix = Array2D.new([2, 2], [1, 2, 3, 4]);
    expect(() => math.matrixTimesVector(matrix, v)).toThrowError();
  });

  it('matrix times vector throws when not passed a matrix', () => {
    const v = Array1D.new([2, 3]);
    // tslint:disable-next-line:no-any
    const matrix: any = Array3D.new([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
    expect(() => math.matrixTimesVector(matrix, v)).toThrowError();
  });

  it('Dot product', () => {
    const v1 = Array1D.new([2, 3]);
    const v2 = Array1D.new([2, 1]);
    const result = math.dotProduct(v1, v2);

    expect(result.get()).toEqual(7);
  });

  it('Dot product throws when vectors are different size', () => {
    const v1 = Array1D.new([2, 3, 3]);
    const v2 = Array1D.new([2, 1]);
    expect(() => math.dotProduct(v1, v2)).toThrowError();
    expect(() => math.dotProduct(v2, v1)).toThrowError();
  });

  it('Dot product throws when passed non vectors', () => {
    // tslint:disable-next-line:no-any
    const v1: any = Array2D.new([2, 2], [1, 2, 3, 3]);
    const v2 = Array1D.new([2, 1]);
    expect(() => math.dotProduct(v1, v2)).toThrowError();
    expect(() => math.dotProduct(v2, v1)).toThrowError();
  });

  it('Outer product', () => {
    const v1 = Array1D.new([2, 3]);
    const v2 = Array1D.new([2, 1]);
    const result = math.outerProduct(v1, v2);

    const expected = new Float32Array([4, 2, 6, 3]);
    expect(result.shape).toEqual([2, 2]);
    expect(result.getValues()).toEqual(expected);
  });

  it('Dot product propagates NaNs', () => {
    const v1 = Array1D.new([2, NaN]);
    const v2 = Array1D.new([2, 1]);
    const result = math.dotProduct(v1, v2);
    expect(result.get()).toEqual(NaN);
  });

  it('Matrix * vector propagates NaNs', () => {
    const matrix = Array2D.new([2, 2], [1, 2, 3, 4]);
    const v = Array1D.new([2, NaN]);
    const result = math.matrixTimesVector(matrix, v);

    const expected = new Float32Array([NaN, NaN]);
    expect(result.getValues()).toEqual(expected);
  });
});

describe('NDArrayMathCPU element-wise mul/div', () => {
  let math: NDArrayMathCPU;
  beforeEach(() => {
    math = new NDArrayMathCPU();
  });

  it('multiplication with broadcasting.', () => {
    // Same shapes, no broadcasting.
    let a = Array2D.new([2, 2], [1, 2, 3, 4]);
    let b = Array2D.new([2, 2], [5, 4, 3, 2]);
    let expected = Array2D.new([2, 2], [5, 8, 9, 8]);
    expect(expected.equals(math.elementWiseMulBroadcast(a, b))).toBe(true);

    // Broadcast a over b.
    a = Array2D.new([2, 2], [1, 2, 3, 4]);
    b = Array2D.new([4, 4], [2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8]);
    expected = Array2D.new(
        [4, 4], [2, 6, 4, 10, 9, 16, 15, 24, 4, 10, 6, 14, 15, 24, 21, 32]);
    expect(expected.equals(math.elementWiseMulBroadcast(a, b))).toBe(true);
  });

  it('multiplication, no broadcasting', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4]);
    const b = Array2D.new([2, 2], [5, 4, 3, 2]);
    const expected = Array2D.new([2, 2], [5, 8, 9, 8]);
    expect(expected.equals(math.elementWiseMul(a, b))).toBe(true);
  });

  it('multiplication propagates NaNs', () => {
    const a = Array2D.new([2, 2], [1, 3, 4, 0]);
    const b = Array2D.new([2, 2], [NaN, 3, NaN, 3]);
    const result = math.elementWiseMul(a, b).getValues();
    expect(result).toEqual(new Float32Array([NaN, 9, NaN, 0]));
  });

  it('mul throws when passed ndarrays of different shapes', () => {
    const a = Array2D.new([2, 3], [1, 2, -3, -4, 5, 6]);
    const b = Array2D.new([2, 2], [5, 3, 4, -7]);
    expect(() => math.elementWiseMul(a, b)).toThrowError();
    expect(() => math.elementWiseMul(b, a)).toThrowError();
  });

  it('divide', () => {
    const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    const c = Array2D.new([2, 3], [1, 2, 3, 4, 2, 5]);
    const r = math.divide(a, c);

    expect(r.get(0, 0)).toBeCloseTo(1);
    expect(r.get(0, 1)).toBeCloseTo(1);
    expect(r.get(0, 2)).toBeCloseTo(1);
    expect(r.get(1, 0)).toBeCloseTo(1);
    expect(r.get(1, 1)).toBeCloseTo(2.5);
    expect(r.get(1, 2)).toBeCloseTo(6 / 5);
  });

  it('divide propagates NaNs', () => {
    const a = Array2D.new([2, 1], [1, 2]);
    const c = Array2D.new([2, 1], [3, NaN]);
    const r = math.divide(a, c).getValues();
    expect(r[0]).toBeCloseTo(1 / 3);
    expect(r[1]).toEqual(NaN);
  });

  it('divide throws when passed ndarrays of different shapes', () => {
    const a = Array2D.new([2, 3], [1, 2, -3, -4, 5, 6]);
    const b = Array2D.new([2, 2], [5, 3, 4, -7]);
    expect(() => math.divide(a, b)).toThrowError();
    expect(() => math.divide(b, a)).toThrowError();
  });

  it('scalar divided by array', () => {
    const c = Scalar.new(2);
    const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    const r = math.scalarDividedByArray(c, a);

    expect(r.get(0, 0)).toBeCloseTo(2 / 1);
    expect(r.get(0, 1)).toBeCloseTo(2 / 2);
    expect(r.get(0, 2)).toBeCloseTo(2 / 3);
    expect(r.get(1, 0)).toBeCloseTo(2 / 4);
    expect(r.get(1, 1)).toBeCloseTo(2 / 5);
    expect(r.get(1, 2)).toBeCloseTo(2 / 6);
  });

  it('scalar divided by array propagates NaNs', () => {
    const c = Scalar.new(NaN);
    const a = Array2D.new([1, 3], [1, 2, 3]);
    const r = math.scalarDividedByArray(c, a).getValues();
    expect(r).toEqual(new Float32Array([NaN, NaN, NaN]));
  });

  it('scalar divided by array throws when passed non scalar', () => {
    // tslint:disable-next-line:no-any
    const c: any = Array1D.new([1, 2, 3]);
    const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);

    expect(() => math.scalarDividedByArray(c, a)).toThrowError();
  });

  it('array divided by scalar', () => {
    const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    const c = Scalar.new(2);
    const r = math.arrayDividedByScalar(a, c);

    expect(r.get(0, 0)).toBeCloseTo(1 / 2);
    expect(r.get(0, 1)).toBeCloseTo(2 / 2);
    expect(r.get(0, 2)).toBeCloseTo(3 / 2);
    expect(r.get(1, 0)).toBeCloseTo(4 / 2);
    expect(r.get(1, 1)).toBeCloseTo(5 / 2);
    expect(r.get(1, 2)).toBeCloseTo(6 / 2);
  });

  it('array divided by scalar propagates NaNs', () => {
    const a = Array2D.new([1, 3], [1, 2, NaN]);
    const c = Scalar.new(2);
    const r = math.arrayDividedByScalar(a, c).getValues();
    expect(r[0]).toBeCloseTo(1 / 2);
    expect(r[1]).toBeCloseTo(2 / 2);
    expect(r[2]).toEqual(NaN);
  });

  it('array divided by scalar throws when passed non scalar', () => {
    // tslint:disable-next-line:no-any
    const c: any = Array1D.new([1, 2, 3]);
    const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);

    expect(() => math.arrayDividedByScalar(a, c)).toThrowError();
  });
});

describe('NDArrayMathCPU add/sub', () => {
  let math: NDArrayMathCPU;
  beforeEach(() => {
    math = new NDArrayMathCPU();
  });

  it('add', () => {
    const a = Array1D.new([2, 5, 1]);
    const b = Array1D.new([4, 2, -1]);
    const expected = Array1D.new([6, 7, 0]);
    expect(expected.getValues()).toEqual(math.add(a, b).getValues());
  });

  it('add propagates NaNs', () => {
    const a = Array1D.new([2, 5, NaN]);
    const b = Array1D.new([4, 2, -1]);
    const res = math.add(a, b).getValues();
    expect(res).toEqual(new Float32Array([6, 7, NaN]));
  });

  it('add throws when passed ndarrays with different shape', () => {
    const a = Array1D.new([2, 5, 1, 5]);
    const b = Array1D.new([4, 2, -1]);
    expect(() => math.add(a, b)).toThrowError();
    expect(() => math.add(b, a)).toThrowError();
  });

  it('sub', () => {
    const a = Array1D.new([2, 5, 1]);
    const b = Array1D.new([4, 2, -1]);
    const expected = Array1D.new([-2, 3, 2]);
    expect(expected.getValues()).toEqual(math.sub(a, b).getValues());
  });

  it('sub propagates NaNs', () => {
    const a = Array1D.new([2, 5, 1]);
    const b = Array1D.new([4, NaN, -1]);
    const res = math.sub(a, b).getValues();
    expect(res).toEqual(new Float32Array([-2, NaN, 2]));
  });

  it('sub throws when passed ndarrays with different shape', () => {
    const a = Array1D.new([2, 5, 1, 5]);
    const b = Array1D.new([4, 2, -1]);
    expect(() => math.sub(a, b)).toThrowError();
    expect(() => math.sub(b, a)).toThrowError();
  });
});

describe('NDArrayMathCPU scalarTimesNDArray', () => {
  let math: NDArrayMathCPU;
  beforeEach(() => {
    math = new NDArrayMathCPU();
  });

  it('scalar times ndarray', () => {
    const a = Array2D.new([3, 2], [2, -5, 1, 1, 4, 0]);
    const c = Scalar.new(2);
    const expected = Array2D.new([3, 2], [4, -10, 2, 2, 8, 0]);
    expect(expected.getValues())
        .toEqual(math.scalarTimesArray(c, a).getValues());
  });

  it('scalar times ndarray throws when passed non-scalar', () => {
    const a = Array2D.new([3, 2], [2, -5, 1, 1, 4, 0]);
    // tslint:disable-next-line:no-any
    const c: any = Array1D.new([1, 2, 3, 4]);
    expect(() => math.scalarTimesArray(c, a)).toThrowError();
  });
});

describe('NDArrayMathCPU scaledNDArrayAdd', () => {
  let math: NDArrayMathCPU;
  beforeEach(() => {
    math = new NDArrayMathCPU();
  });

  it('Scaled ndarray add', () => {
    const a = Array2D.new([2, 3], [2, 4, 6, 8, 10, 12]);
    const b = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    const c1 = Scalar.new(3);
    const c2 = Scalar.new(2);

    const expected = Array2D.new([2, 3], [8, 16, 24, 32, 40, 48]);
    expect(math.scaledArrayAdd<Array2D>(c1, a, c2, b).equals(expected))
        .toBe(true);

    // Different sizes throws an error.
    const wrongSizeMat = Array2D.new([2, 2], [1, 2, 3, 4]);
    expect(() => math.scaledArrayAdd<Array2D>(c1, wrongSizeMat, c2, b))
        .toThrowError();
  });

  it('throws when passed non-scalars', () => {
    const a = Array2D.new([2, 3], [2, 4, 6, 8, 10, 12]);
    const b = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    const c1 = Array1D.randNormal([10]);
    const c2 = Scalar.new(2);

    expect(() => math.scaledArrayAdd<Array2D>(c1 as Scalar, a, c2, b))
        .toThrowError();
    expect(() => math.scaledArrayAdd<Array2D>(c2, a, c1 as Scalar, b))
        .toThrowError();
  });

  it('throws when NDArrays are different shape', () => {
    const a = Array2D.new([2, 3], [2, 4, 6, 8, 10, 12]);
    const b = Array2D.new([2, 4], [1, 2, 3, 4, 5, 6, 7, 8]);
    const c1 = Scalar.new(3);
    const c2 = Scalar.new(2);

    expect(() => math.scaledArrayAdd<Array2D>(c1, a, c2, b)).toThrowError();
  });
});

describe('NDArrayMathCPU argmin/max, argmaxequals, min/max', () => {
  let math: NDArrayMathCPU;
  beforeEach(() => {
    math = new NDArrayMathCPU();
  });

  it('Arg max', () => {
    expect(math.argMax(Array1D.new([5, 0, 3, 7, 3])).get()).toBe(3);
    expect(math.argMax(Array1D.new([-100.3, .3, 11.1, 9.9, 7.33])).get())
        .toBe(2);
    expect(math.argMax(Array1D.new([-100.3, -20.0, -10.0, -5])).get()).toBe(3);
  });

  it('Arg max propagates NaNs', () => {
    expect(math.argMax(Array1D.new([5, 0, 3, NaN, 3])).get()).toEqual(NaN);
  });

  it('Argmaxequals equals', () => {
    const a = Array1D.new([5, 0, 3, 7]);
    const b = Array1D.new([-100.3, -20.0, -10.0, -5]);
    const result = math.argMaxEquals(a, b);
    expect(result.get()).toBe(1);
  });

  it('Argmaxequals not equals', () => {
    const a = Array1D.new([5, 0, 3, 1]);
    const b = Array1D.new([-100.3, -20.0, -10.0, -5]);
    const result = math.argMaxEquals(a, b);
    expect(result.get()).toBe(0);
  });

  it('Argmaxequals propagates NaNs', () => {
    const a = Array1D.new([5, 3, 1, 3]);
    const b = Array1D.new([NaN, -20.0, -10.0, -5]);
    const result = math.argMaxEquals(a, b);
    expect(result.get()).toEqual(NaN);
  });

  it('throws when given arrays of different shape', () => {
    const a = Array1D.new([5, 0, 3, 7, 3, 10]);
    const b = Array1D.new([-100.3, -20.0, -10.0, -5, -100]);
    expect(() => math.argMaxEquals(a, b)).toThrowError();
  });

  it('topk', () => {
    const topk = math.topK(Array1D.new([1, -1, 100, -5, -10.6, 3.3, 5]), 3);
    test_util.expectArraysClose(
        topk.values.getValues(), new Float32Array([100, 5, 3.3]), 1e-6);
    test_util.expectArraysClose(
        topk.indices.getValues(), new Float32Array([2, 6, 5]), 1e-6);
  });

  it('Arg min', () => {
    expect(math.argMin(Array1D.new([5, 0, 3, 7, 3])).get()).toBe(1);
    expect(math.argMin(Array1D.new([-100.3, .3, 11.1, 9.9, 7.33])).get())
        .toBe(0);
  });

  it('Arg min propagates NaNs', () => {
    expect(math.argMin(Array1D.new([5, 0, NaN, 7, 3])).get()).toEqual(NaN);
  });

  it('min', () => {
    expect(math.min(Array1D.new([3, -1, 0, 100, -7, 2])).get()).toBe(-7);
  });

  it('min propagates NaNs', () => {
    expect(math.min(Array1D.new([3, NaN, 2])).get()).toEqual(NaN);
  });

  it('max', () => {
    expect(math.max(Array1D.new([3, -1, 0, 100, -7, 2])).get()).toBe(100);
  });

  it('max propagates NaNs', () => {
    expect(math.max(Array1D.new([3, NaN, 2])).get()).toEqual(NaN);
  });
});

describe('NDArrayMathCPU log/exp', () => {
  let math: NDArrayMathCPU;
  beforeEach(() => {
    math = new NDArrayMathCPU();
  });

  it('exp', () => {
    const r = math.exp(Array1D.new([1, 2, 0]));

    expect(r.get(0)).toBeCloseTo(Math.exp(1));
    expect(r.get(1)).toBeCloseTo(Math.exp(2));
    expect(r.get(2)).toBeCloseTo(1);
  });

  it('exp propagates NaNs', () => {
    const a = Array1D.new([1, NaN, 0]);
    const r = math.exp(a).getValues();
    expect(r).toEqual(new Float32Array([Math.exp(1), NaN, 1]));
  });

  it('log', () => {
    const r = math.log(Array1D.new([1, 2]));

    expect(r.get(0)).toBeCloseTo(Math.log(1));
    expect(r.get(1)).toBeCloseTo(Math.log(2));
  });

  it('log propagates NaNs', () => {
    const r = math.log(Array1D.new([1, NaN])).getValues();
    expect(r).toEqual(new Float32Array([Math.log(1), NaN]));
  });

  it('logSumExp', () => {
    const a = Array1D.new([1, 2, -3]);
    const result = math.logSumExp(a);
    expect(result.get())
        .toBeCloseTo(Math.log(Math.exp(1) + Math.exp(2) + Math.exp(-3)));
  });

  it('logSumExp propagates NaNs', () => {
    const a = Array1D.new([1, 2, NaN]);
    const result = math.logSumExp(a);
    expect(result.get()).toEqual(NaN);
  });
});

describe('softmax', () => {
  let math: NDArrayMathCPU;

  beforeEach(() => {
    math = new NDArrayMathCPU();
  });

  it('regular test', () => {
    const y = math.softmax(Array1D.new([2, 1, 3]));
    expect(y.get(0)).toBeCloseTo(0.24472847, 6);
    expect(y.get(1)).toBeCloseTo(0.09003057, 6);
    expect(y.get(2)).toBeCloseTo(0.66524095, 6);
    expect(y.get(0) + y.get(1) + y.get(2)).toBeCloseTo(1, 6);
  });

  it('Overflow', () => {
    const y = math.softmax(Array1D.new([10000, 10000]));
    expect(y.get(0)).toBeCloseTo(0.5, 3);
    expect(y.get(1)).toBeCloseTo(0.5, 3);
  });

  it('Underflow', () => {
    const y = math.softmax(Array1D.new([-10000, -10000]));
    expect(y.get(0)).toBeCloseTo(0.5, 3);
    expect(y.get(1)).toBeCloseTo(0.5, 3);
  });

  it('Huge difference between probabilities', () => {
    const y = math.softmax(Array1D.new([-10000, +10000]));
    expect(y.get(0)).toBeCloseTo(0.0, 6);
    expect(y.get(1)).toBeCloseTo(1, 6);
  });

  it('Propagates NaNs', () => {
    const y = math.softmax(Array1D.new([2, 1, NaN]));
    expect(y.getValues()).toEqual(new Float32Array([NaN, NaN, NaN]));
  });
});

describe('NDArrayMathCPU sum', () => {
  let math: NDArrayMathCPU;
  beforeEach(() => {
    math = new NDArrayMathCPU();
  });

  it('sums values in ndarray', () => {
    const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
    expect(math.sum(a).get()).toBe(7);
  });

  it('propagates NaNs', () => {
    const a = Array2D.new([3, 2], [1, 2, 3, NaN, 0, 1]);
    expect(math.sum(a).get()).toEqual(NaN);
  });
});

describe('NDArrayMathCPU unary ops', () => {
  let math: NDArrayMathCPU;
  beforeEach(() => {
    math = new NDArrayMathCPU();
  });

  it('relu', () => {
    const a = Array1D.new([1, -2, 0, 3, -0.1]);
    const result = math.relu(a);
    expect(result.getValues()).toEqual(new Float32Array([1, 0, 0, 3, 0]));
  });

  it('relu propagates NaNs', () => {
    const a = Array1D.new([1, -2, 0, 3, -0.1, NaN]);
    const result = math.relu(a);
    expect(result.getValues()).toEqual(new Float32Array([1, 0, 0, 3, 0, NaN]));
  });

  it('step', () => {
    const a = Array1D.new([1, -2, 0, 3, -0.1]);
    const result = math.step(a);
    expect(result.getValues()).toEqual(new Float32Array([1, 0, 0, 1, 0]));
  });

  it('step propagates NaNs', () => {
    const a = Array1D.new([1, -2, 0, 3, NaN]);
    const result = math.step(a);
    expect(result.getValues()).toEqual(new Float32Array([1, 0, 0, 1, NaN]));
  });

  it('neg', () => {
    const a = Array1D.new([1, -3, 2, 7, -4]);
    expect(math.neg(a).getValues()).toEqual(new Float32Array([
      -1, 3, -2, -7, 4
    ]));
  });

  it('neg propagate NaNs', () => {
    const a = Array1D.new([1, -3, 2, 7, NaN]);
    expect(math.neg(a).getValues()).toEqual(new Float32Array([
      -1, 3, -2, -7, NaN
    ]));
  });

  it('sigmoid', () => {
    const a = Array1D.new([3, 5]);
    const res = math.sigmoid(a).getValues();
    const expected = [3, 5].map(x => 1 / (1 + Math.exp(-x)));
    expect(res).toEqual(new Float32Array(expected));
  });

  it('sigmoid propagates NaNs', () => {
    const a = Array1D.new([3, NaN]);
    const res = math.sigmoid(a).getValues();
    expect(res).toEqual(new Float32Array([1 / (1 + Math.exp(-3)), NaN]));
  });

  it('tanh', () => {
    const a = Array1D.new([4, -3, 0]);
    const res = math.tanh(a).getValues();
    const expected = [util.tanh(4), util.tanh(-3), util.tanh(0)];
    expect(res).toEqual(new Float32Array(expected));
  });

  it('tanh propagates NaNs', () => {
    const a = Array1D.new([4, NaN, 0]);
    const res = math.tanh(a).getValues();
    const expected = [util.tanh(4), NaN, util.tanh(0)];
    expect(res).toEqual(new Float32Array(expected));
  });

  it('sin', () => {
    const a = Array1D.new([4, -3, 0]);
    const res = math.sin(a).getValues();
    const expected = [Math.sin(4), Math.sin(-3), Math.sin(0)];
    expect(res).toEqual(new Float32Array(expected));
  });

  it('sin propagates NaNs', () => {
    const a = Array1D.new([4, NaN, 0]);
    const res = math.sin(a).getValues();
    const expected = [Math.sin(4), NaN, Math.sin(0)];
    expect(res).toEqual(new Float32Array(expected));
  });
});

describe('NDArrayMathCPU scalar OP ndarray', () => {
  let math: NDArrayMathCPU;
  beforeEach(() => {
    math = new NDArrayMathCPU();
  });

  it('c + A', () => {
    const c = Scalar.new(5);
    const a = Array1D.new([1, 2, 3]);
    expect(math.scalarPlusArray(c, a).getValues()).toEqual(new Float32Array([
      6, 7, 8
    ]));
  });

  it('c + A propagates NaNs', () => {
    const c = Scalar.new(NaN);
    const a = Array1D.new([1, 2, 3]);
    const res = math.scalarPlusArray(c, a).getValues();
    expect(res).toEqual(new Float32Array([NaN, NaN, NaN]));
  });

  it('c + A throws when passed non scalar', () => {
    // tslint:disable-next-line:no-any
    const c: any = Array1D.new([1, 2, 3]);
    const a = Array1D.new([1, 2, 3]);
    expect(() => math.scalarPlusArray(c, a)).toThrowError();
  });

  it('c - A', () => {
    const c = Scalar.new(5);
    const a = Array1D.new([1, 2, 3]);
    expect(math.scalarMinusArray(c, a).getValues()).toEqual(new Float32Array([
      4, 3, 2
    ]));
  });

  it('c - A throws when passed non scalar', () => {
    // tslint:disable-next-line:no-any
    const c: any = Array1D.new([1, 2, 3]);
    const a = Array1D.new([1, 2, 3]);
    expect(() => math.scalarMinusArray(c, a)).toThrowError();
  });

  it('A - c', () => {
    const a = Array1D.new([1, 2, 3]);
    const c = Scalar.new(5);
    expect(math.arrayMinusScalar(a, c).getValues()).toEqual(new Float32Array([
      -4, -3, -2
    ]));
  });

  it('A - c propagates NaNs', () => {
    const a = Array1D.new([1, NaN, 3]);
    const c = Scalar.new(5);
    const res = math.arrayMinusScalar(a, c).getValues();
    expect(res).toEqual(new Float32Array([-4, NaN, -2]));
  });

  it('A - c throws when passed non scalar', () => {
    // tslint:disable-next-line:no-any
    const c: any = Array1D.new([1, 2, 3]);
    const a = Array1D.new([1, 2, 3]);
    expect(() => math.arrayMinusScalar(a, c)).toThrowError();
  });
});

describe('NDArrayMathCPU switchDim', () => {
  let math: NDArrayMathCPU;
  beforeEach(() => {
    math = new NDArrayMathCPU();
  });

  it('Switch dim 2D (no change)', () => {
    const t = Array2D.new([2, 4], [1, 11, 2, 22, 3, 33, 4, 44]);
    const t2 = math.switchDim(t, [0, 1]);
    expect(t2.shape).toEqual(t.shape);
    expect(t2.getValues()).toEqual(t.getValues());
  });

  it('Switch dim 2D (transpose)', () => {
    const t = Array2D.new([2, 4], [1, 11, 2, 22, 3, 33, 4, 44]);
    const t2 = math.switchDim(t, [1, 0]);
    expect(t2.shape).toEqual([4, 2]);
    const expected = new Float32Array([1, 3, 11, 33, 2, 4, 22, 44]);
    expect(t2.getValues()).toEqual(expected);
  });

  it('Switch dim 3D [r, c, d] => [d, r, c]', () => {
    const t = Array3D.new([2, 2, 2], [1, 11, 2, 22, 3, 33, 4, 44]);
    const t2 = math.switchDim(t, [2, 0, 1]);
    expect(t2.shape).toEqual([2, 2, 2]);
    const expected = new Float32Array([1, 2, 3, 4, 11, 22, 33, 44]);
    expect(t2.getValues()).toEqual(expected);
  });

  it('Switch dim 3D [r, c, d] => [d, c, r]', () => {
    const t = Array3D.new([2, 2, 2], [1, 11, 2, 22, 3, 33, 4, 44]);
    const t2 = math.switchDim(t, [2, 1, 0]);
    expect(t2.shape).toEqual([2, 2, 2]);
    const expected = new Float32Array([1, 3, 2, 4, 11, 33, 22, 44]);
    expect(t2.getValues()).toEqual(expected);
  });
});

describe('NDArrayMathCPU maxPool', () => {
  let math: NDArrayMathCPU;
  beforeEach(() => {
    math = new NDArrayMathCPU();
  });

  it('1x1x1 in, 1x1 filter, 1 stride: [0] => [0]', () => {
    const a = Array3D.new([1, 1, 1], [0]);
    const result = math.maxPool(a, 1, 1, 0);
    expect(result.getValues()).toBeCloseTo(0);
  });

  it('3x3x1 in, 2x2 filter, 1 stride', () => {
    // Feed forward.
    const a = Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, 9, 8]);
    const result = math.maxPool(a, 2, 1, 0);

    expect(result.shape).toEqual([2, 2, 1]);
    expect(result.getValues()).toEqual(new Float32Array([5, 6, 9, 9]));
  });

  it('3x3x1 in, 2x2 filter, 1 stride, propagates NaNs', () => {
    const a = Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, NaN, 9]);
    const result = math.maxPool(a, 2, 1, 0);

    expect(result.shape).toEqual([2, 2, 1]);
    expect(result.getValues()).toEqual(new Float32Array([5, 6, NaN, NaN]));
  });

  it('3x3x2 in, 2x2 filter, 1 stride', () => {
    // Feed forward.
    const a = Array3D.new(
        [3, 3, 2],
        [1, 99, 2, 88, 3, 77, 4, 66, 5, 55, 6, 44, 7, 33, 9, 22, 8, 11]);
    const result = math.maxPool(a, 2, 1, 0);

    expect(result.shape).toEqual([2, 2, 2]);
    expect(result.getValues()).toEqual(new Float32Array([
      5, 99, 6, 88, 9, 66, 9, 55
    ]));
  });

  it('4x4x1 in, 2x2 filter, 2 stride', () => {
    // Feed forward.
    const a = Array3D.new(
        [4, 4, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    const result = math.maxPool(a, 2, 2, 0);

    expect(result.shape).toEqual([2, 2, 1]);
    expect(result.getValues()).toEqual(new Float32Array([5, 7, 13, 15]));
  });

  it('2x2x1 in, 2x2 filter, 2 stride, pad=1', () => {
    // Feed forward.
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
    const result = math.maxPool(a, 2, 2, 1);

    expect(result.shape).toEqual([2, 2, 1]);
    expect(result.getValues()).toEqual(new Float32Array([1, 2, 3, 4]));
  });
});

describe('NDArrayMathCPU minPool', () => {
  let math: NDArrayMathCPU;
  beforeEach(() => {
    math = new NDArrayMathCPU();
  });

  it('1x1x1 in, 1x1 filter, 1 stride: [0] => [0]', () => {
    const a = Array3D.new([1, 1, 1], [0]);
    const result = math.minPool(a, 1, 1, 0);
    expect(result.getValues()).toBeCloseTo(0);
  });

  it('3x3x1 in, 2x2 filter, 1 stride', () => {
    // Feed forward.
    const a = Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, 9, 8]);
    const result = math.minPool(a, 2, 1, 0);

    expect(result.shape).toEqual([2, 2, 1]);
    expect(result.getValues()).toEqual(new Float32Array([1, 2, 4, 5]));
  });

  it('3x3x1 in, 2x2 filter, 1 stride, propagates NaNs', () => {
    const a = Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, NaN, 8]);
    const result = math.minPool(a, 2, 1, 0);

    expect(result.shape).toEqual([2, 2, 1]);
    expect(result.getValues()).toEqual(new Float32Array([1, 2, NaN, NaN]));
  });

  it('3x3x2 in, 2x2 filter, 1 stride', () => {
    // Feed forward.
    const a = Array3D.new(
        [3, 3, 2],
        [1, 99, 2, 88, 3, 77, 4, 66, 5, 55, 6, 44, 7, 33, 9, 22, 8, 11]);
    const result = math.minPool(a, 2, 1, 0);

    expect(result.shape).toEqual([2, 2, 2]);
    expect(result.getValues()).toEqual(new Float32Array([
      1, 55, 2, 44, 4, 22, 5, 11
    ]));
  });

  it('4x4x1 in, 2x2 filter, 2 stride', () => {
    // Feed forward.
    const a = Array3D.new(
        [4, 4, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    const result = math.minPool(a, 2, 2, 0);

    expect(result.shape).toEqual([2, 2, 1]);
    expect(result.getValues()).toEqual(new Float32Array([0, 2, 8, 10]));
  });

  it('2x2x1 in, 2x2 filter, 2 stride, pad=1', () => {
    // Feed forward.
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
    const result = math.minPool(a, 2, 2, 1);

    expect(result.shape).toEqual([2, 2, 1]);
    expect(result.getValues()).toEqual(new Float32Array([1, 2, 3, 4]));
  });
});

describe('NDArrayMathCPU avgPool', () => {
  let math: NDArrayMathCPU;
  beforeEach(() => {
    math = new NDArrayMathCPU();
  });

  it('1x1x1 in, 1x1 filter, 1 stride: [0] => [0]', () => {
    const a = Array3D.new([1, 1, 1], [0]);
    const result = math.avgPool(a, 1, 1, 0);
    expect(result.getValues()).toBeCloseTo(0);
  });

  it('3x3x1 in, 2x2 filter, 1 stride', () => {
    // Feed forward.
    const a = Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, 9, 8]);
    const result = math.avgPool(a, 2, 1, 0);

    expect(result.shape).toEqual([2, 2, 1]);
    expect(result.getValues()).toEqual(new Float32Array([3, 4, 6.25, 7]));
  });

  it('3x3x1 in, 2x2 filter, 1 stride, propagates NaNs', () => {
    // Feed forward.
    const a = Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, NaN, 8]);
    const result = math.avgPool(a, 2, 1, 0);

    expect(result.shape).toEqual([2, 2, 1]);
    expect(result.getValues()).toEqual(new Float32Array([3, 4, NaN, NaN]));
  });

  it('3x3x2 in, 2x2 filter, 1 stride', () => {
    // Feed forward.
    const a = Array3D.new(
        [3, 3, 2],
        [1, 99, 2, 88, 3, 77, 4, 66, 5, 55, 6, 44, 7, 33, 9, 22, 8, 11]);
    const result = math.avgPool(a, 2, 1, 0);

    expect(result.shape).toEqual([2, 2, 2]);
    expect(result.getValues()).toEqual(new Float32Array([
      3, 77, 4, 66, 6.25, 44, 7, 33
    ]));
  });

  it('4x4x1 in, 2x2 filter, 2 stride', () => {
    // Feed forward.
    const a = Array3D.new(
        [4, 4, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    const result = math.avgPool(a, 2, 2, 0);

    expect(result.shape).toEqual([2, 2, 1]);
    expect(result.getValues()).toEqual(new Float32Array([
      2.5, 4.5, 10.5, 12.5
    ]));
  });

  it('2x2x1 in, 2x2 filter, 2 stride, pad=1', () => {
    // Feed forward.
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
    const result = math.avgPool(a, 2, 2, 1);

    expect(result.shape).toEqual([2, 2, 1]);
    expect(result.getValues()).toEqual(new Float32Array([0.25, 0.5, 0.75, 1]));
  });
});

describe('NDArrayMathCPU maxPoolBackprop', () => {
  let math: NDArrayMathCPU;
  beforeEach(() => {
    math = new NDArrayMathCPU();
  });

  it('x=3x3x1, f=2, s=1, no duplicate max value, test #1', () => {
    const dy = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
    const x = Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const expected = new Float32Array([0, 0, 0, 0, 1, 2, 0, 3, 4]);
    const dx = math.maxPoolBackprop(dy, x, 2, 1, 0);
    expect(dx.getValues()).toEqual(expected);
  });

  it('x=3x3x1, f=2, s=1, no duplicate max value, test #2', () => {
    const dy = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
    const x = Array3D.new([3, 3, 1], [9, 5, 6, 6, 8, 4, 9, 5, 10]);
    const expected = new Float32Array([1, 0, 0, 0, 2, 0, 3, 0, 4]);
    const dx = math.maxPoolBackprop(dy, x, 2, 1, 0);
    expect(dx.getValues()).toEqual(expected);
  });

  it('x=3x3x1, f=2, s=1 duplicate max value, test 1', () => {
    const dy = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
    const x = Array3D.new([3, 3, 1], [0, 0, 0, 0, 5, 0, 0, 0, 0]);
    const expected = new Float32Array([0, 0, 0, 0, 10, 0, 0, 0, 0]);
    const dx = math.maxPoolBackprop(dy, x, 2, 1, 0);
    expect(dx.getValues()).toEqual(expected);
  });

  it('x=3x3x1, f=2, s=1 duplicate max value, test 2', () => {
    const dy = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
    const x = Array3D.new([3, 3, 1], [1, 3, 2, 1, 2, 1, 1, 1, 5]);
    const expected = new Float32Array([0, 3, 0, 0, 3, 0, 0, 0, 4]);
    const dx = math.maxPoolBackprop(dy, x, 2, 1, 0);
    expect(dx.getValues()).toEqual(expected);
  });

  it('x=4x4x1, f=2, s=2, test #1', () => {
    const dy = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
    const x = Array3D.new(
        [4, 4, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    const expected =
        new Float32Array([0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 3, 0, 4]);
    const dx = math.maxPoolBackprop(dy, x, 2, 2, 0);
    expect(dx.getValues()).toEqual(expected);
  });

  it('x=4x4x1, f=2, s=2, test #2', () => {
    const dy = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
    const x = Array3D.new(
        [4, 4, 1], [1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1]);
    const expected =
        new Float32Array([0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0]);
    const dx = math.maxPoolBackprop(dy, x, 2, 2, 0);
    expect(dx.getValues()).toEqual(expected);
  });

  it('x=5x5x1, f=3, s=2 no duplicate max value', () => {
    const dy = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
    const x = Array3D.new([5, 5, 1], [
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
    ]);
    const expected = new Float32Array([
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 4
    ]);
    const dx = math.maxPoolBackprop(dy, x, 3, 2, 0);
    expect(dx.getValues()).toEqual(expected);
  });

  it('x=5x5x1, f=3, s=2 duplicate max value', () => {
    const dy = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
    const x = Array3D.new([5, 5, 1], [
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 24,
      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 12
    ]);
    const expected = new Float32Array([
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]);
    const dx = math.maxPoolBackprop(dy, x, 3, 2, 0);
    expect(dx.getValues()).toEqual(expected);
  });

  // Max pool backprop depth > 1.
  it('x=3x3x2, f=2, s=1, no duplicate max value', () => {
    // This test combines the first two 3x3x1 tests with no duplicates to
    // make depth=2,
    // dy is slightly modified to show the difference.
    const dy = Array3D.new([2, 2, 2], [1, 44, 2, 33, 3, 22, 4, 11]);
    const x = Array3D.new(
        [3, 3, 2],
        [1, 99, 2, 55, 3, 66, 4, 66, 5, 88, 6, 44, 7, 99, 8, 55, 9, 100]);
    const expected = new Float32Array(
        [0, 44, 0, 0, 0, 0, 0, 0, 1, 33, 2, 0, 0, 22, 3, 0, 4, 11]);

    const dx = math.maxPoolBackprop(dy, x, 2, 1, 0);
    expect(dx.getValues()).toEqual(expected);
  });

  it('x=3x3x2, f=2, s=1, duplicate max value', () => {
    // This test combines the first two 3x3x1 tests with duplicates to
    // make depth=2,
    // dy is slightly modified to show the difference.
    const dy = Array3D.new([2, 2, 2], [1, 44, 2, 33, 3, 22, 4, 11]);
    const x = Array3D.new(
        [3, 3, 2], [0, 1, 0, 3, 0, 2, 0, 1, 5, 2, 0, 1, 0, 1, 0, 1, 0, 5]);
    const expected = new Float32Array(
        [0, 0, 0, 77, 0, 0, 0, 0, 10, 22, 0, 0, 0, 0, 0, 0, 0, 11]);

    const dx = math.maxPoolBackprop(dy, x, 2, 1, 0);
    expect(dx.getValues()).toEqual(expected);
  });

  it('x=4x4x2, f=2, s=1', () => {
    // This test combines the first two 4x4x1 tests with duplicates to make
    // depth=2,
    // dy is slightly modified to show the difference.
    const dy = Array3D.new([2, 2, 2], [1, 11, 2, 22, 3, 33, 4, 44]);
    const x = Array3D.new([4, 4, 2], [
      0, 1, 1, 2, 2,  2, 3,  1, 4,  1, 5,  1, 6,  1, 7,  1,
      8, 1, 9, 1, 10, 1, 11, 1, 12, 1, 13, 2, 14, 2, 15, 1
    ]);
    const expected = new Float32Array([
      0, 0, 0, 11, 0, 22, 0, 0, 0, 0, 1, 0,  0, 0,  2, 0,
      0, 0, 0, 0,  0, 0,  0, 0, 0, 0, 3, 33, 0, 44, 4, 0
    ]);
    const dx = math.maxPoolBackprop(dy, x, 2, 2, 0);
    expect(dx.getValues()).toEqual(expected);
  });

  it('x=5x5x2, f=3, s=2 no duplicate max value', () => {
    // This test combines the first two 5x5x1 tests with duplicates to make
    // depth=2,
    // dy is slightly modified to show the difference.
    const dy = Array3D.new([2, 2, 2], [1, 11, 2, 22, 3, 33, 4, 44]);
    const x = Array3D.new([5, 5, 2], [
      0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,
      8,  9,  9,  10, 10, 11, 11, 12, 24, 13, 13, 14, 14, 15, 15, 16, 16,
      17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 12
    ]);
    const expected = new Float32Array([
      0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 110, 0, 0, 2, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 3, 0, 0, 0, 4, 0
    ]);
    const dx = math.maxPoolBackprop(dy, x, 3, 2, 0);
    expect(dx.getValues()).toEqual(expected);
  });
});

describe('NDArrayMathCPU resizeBilinear', () => {
  let math: NDArrayMathCPU;
  beforeEach(() => {
    math = new NDArrayMathCPU();
  });

  it('simple alignCorners=false', () => {
    const input = Array3D.new([2, 2, 1], [2, 2, 4, 4]);
    const output = math.resizeBilinear3D(input, [3, 3], false);

    test_util.expectArraysClose(
        output.getValues(),
        new Float32Array([2, 2, 2, 10 / 3, 10 / 3, 10 / 3, 4, 4, 4]), 1e-4);
  });

  it('simple alignCorners=true', () => {
    const input = Array3D.new([2, 2, 1], [2, 2, 4, 4]);
    const output = math.resizeBilinear3D(input, [3, 3], true);

    test_util.expectArraysClose(
        output.getValues(), new Float32Array([2, 2, 2, 3, 3, 3, 4, 4, 4]),
        1e-4);
  });

  it('matches tensorflow w/ random numbers alignCorners=false', () => {
    const input = Array3D.new([2, 3, 2], [
      1.19074044, 0.91373104, 2.01611669, -0.52270832, 0.38725395, 1.30809779,
      0.61835143, 3.49600659, 2.09230986, 0.56473997, 0.03823943, 1.19864896
    ]);
    const output = math.resizeBilinear3D(input, [4, 5], false);

    test_util.expectArraysClose(
        output.getValues(), new Float32Array([
          1.19074047,  0.91373104, 1.68596613, 0.05186744, 1.69034398,
          -0.15654698, 0.7130264,  0.94193673, 0.38725394, 1.30809784,
          0.9045459,   2.20486879, 1.59434628, 0.89455694, 1.68591988,
          0.26748738,  0.58103991, 1.00690198, 0.21274668, 1.25337338,
          0.6183514,   3.49600649, 1.50272655, 1.73724651, 1.68149579,
          0.69152176,  0.44905344, 1.07186723, 0.03823943, 1.19864893,
          0.6183514,   3.49600649, 1.50272655, 1.73724651, 1.68149579,
          0.69152176,  0.44905344, 1.07186723, 0.03823943, 1.19864893
        ]),
        1e-4);
  });

  it('matches tensorflow w/ random numbers alignCorners=true', () => {
    const input = Array3D.new([2, 3, 2], [
      1.56324531, 2.13817752, 1.44398421, 1.07632684, 0.59306785, -0.36970865,
      1.62451879, 1.8367334, 1.13944798, 2.01993218, 2.01919952, 2.67524054
    ]);
    const output = math.resizeBilinear3D(input, [4, 5], true);

    test_util.expectArraysClose(
        output.getValues(), new Float32Array([
          1.5632453,  2.13817763, 1.50361478, 1.60725224, 1.44398427,
          1.07632685, 1.01852608, 0.35330909, 0.59306782, -0.36970866,
          1.58366978, 2.03769612, 1.46307099, 1.71427906, 1.3424722,
          1.39086199, 1.20545864, 1.01806819, 1.06844509, 0.6452744,
          1.60409427, 1.93721485, 1.42252707, 1.82130599, 1.24096,
          1.70539713, 1.3923912,  1.68282723, 1.54382229, 1.66025746,
          1.62451875, 1.83673346, 1.38198328, 1.92833281, 1.13944793,
          2.01993227, 1.57932377, 2.34758639, 2.01919961, 2.67524052
        ]),
        1e-4);
  });
});

describe('NDArrayMathCPU batchNorm', () => {
  let math: NDArrayMathCPU;
  beforeEach(() => {
    math = new NDArrayMathCPU();
  });

  it('simple batchnorm, no offset or scale, 2x1x2', () => {
    const x = Array3D.new([2, 1, 2], new Float32Array([2, 100, 4, 400]));
    const mean = Array1D.new([1, 2]);
    const variance = Array1D.new([2, 3]);
    const varianceEpsilon = .001;

    const result = math.batchNormalization3D(
        x, mean, variance, varianceEpsilon, undefined, undefined);

    test_util.expectArraysClose(
        result.getValues(), new Float32Array([
          (x.get(0, 0, 0) - mean.get(0)) * 1 /
              Math.sqrt(variance.get(0) + varianceEpsilon),
          (x.get(0, 0, 1) - mean.get(1)) * 1 /
              Math.sqrt(variance.get(1) + varianceEpsilon),
          (x.get(1, 0, 0) - mean.get(0)) * 1 /
              Math.sqrt(variance.get(0) + varianceEpsilon),
          (x.get(1, 0, 1) - mean.get(1)) * 1 /
              Math.sqrt(variance.get(1) + varianceEpsilon)
        ]),
        1e-6);
  });

  it('simple batchnorm, no offset, 2x1x2', () => {
    const x = Array3D.new([2, 1, 2], new Float32Array([2, 100, 4, 400]));
    const mean = Array1D.new([1, 2]);
    const variance = Array1D.new([2, 3]);
    const scale = Array1D.new([4, 5]);
    const varianceEpsilon = .001;

    const result = math.batchNormalization3D(
        x, mean, variance, varianceEpsilon, scale, undefined);

    test_util.expectArraysClose(
        result.getValues(), new Float32Array([
          (x.get(0, 0, 0) - mean.get(0)) * scale.get(0) /
              Math.sqrt(variance.get(0) + varianceEpsilon),
          (x.get(0, 0, 1) - mean.get(1)) * scale.get(1) /
              Math.sqrt(variance.get(1) + varianceEpsilon),
          (x.get(1, 0, 0) - mean.get(0)) * scale.get(0) /
              Math.sqrt(variance.get(0) + varianceEpsilon),
          (x.get(1, 0, 1) - mean.get(1)) * scale.get(1) /
              Math.sqrt(variance.get(1) + varianceEpsilon)
        ]),
        1e-6);
  });

  it('simple batchnorm, no scale, 2x1x2', () => {
    const x = Array3D.new([2, 1, 2], new Float32Array([2, 100, 4, 400]));
    const mean = Array1D.new([1, 2]);
    const variance = Array1D.new([2, 3]);
    const offset = Array1D.new([4, 5]);

    const varianceEpsilon = .001;

    const result = math.batchNormalization3D(
        x, mean, variance, varianceEpsilon, undefined, offset);

    test_util.expectArraysClose(
        result.getValues(), new Float32Array([
          offset.get(0) +
              (x.get(0, 0, 0) - mean.get(0)) * 1 /
                  Math.sqrt(variance.get(0) + varianceEpsilon),
          offset.get(1) +
              (x.get(0, 0, 1) - mean.get(1)) * 1 /
                  Math.sqrt(variance.get(1) + varianceEpsilon),
          offset.get(0) +
              (x.get(1, 0, 0) - mean.get(0)) * 1 /
                  Math.sqrt(variance.get(0) + varianceEpsilon),
          offset.get(1) +
              (x.get(1, 0, 1) - mean.get(1)) * 1 /
                  Math.sqrt(variance.get(1) + varianceEpsilon)
        ]),
        1e-6);
  });

  it('simple batchnorm, 2x1x2', () => {
    const x = Array3D.new([2, 1, 2], new Float32Array([2, 100, 4, 400]));
    const mean = Array1D.new([1, 2]);
    const variance = Array1D.new([2, 3]);
    const offset = Array1D.new([3, 4]);
    const scale = Array1D.new([4, 5]);

    const varianceEpsilon = .001;

    const result = math.batchNormalization3D(
        x, mean, variance, varianceEpsilon, scale, offset);

    test_util.expectArraysClose(
        result.getValues(), new Float32Array([
          offset.get(0) +
              (x.get(0, 0, 0) - mean.get(0)) * scale.get(0) /
                  Math.sqrt(variance.get(0) + varianceEpsilon),
          offset.get(1) +
              (x.get(0, 0, 1) - mean.get(1)) * scale.get(1) /
                  Math.sqrt(variance.get(1) + varianceEpsilon),
          offset.get(0) +
              (x.get(1, 0, 0) - mean.get(0)) * scale.get(0) /
                  Math.sqrt(variance.get(0) + varianceEpsilon),
          offset.get(1) +
              (x.get(1, 0, 1) - mean.get(1)) * scale.get(1) /
                  Math.sqrt(variance.get(1) + varianceEpsilon)
        ]),
        1e-6);
  });

  it('batchnorm matches tensorflow, 2x3x3', () => {
    const x =
        Array3D.new([2, 3, 3], new Float32Array([
                      0.49955603, 0.04158615, -1.09440524, 2.03854165,
                      -0.61578344, 2.87533573, 1.18105987, 0.807462, 1.87888837,
                      2.26563962, -0.37040935, 1.35848753, -0.75347094,
                      0.15683117, 0.91925946, 0.34121279, 0.92717143, 1.89683965
                    ]));
    const mean = Array1D.new([0.39745062, -0.48062894, 0.4847822]);
    const variance = Array1D.new([0.32375343, 0.67117643, 1.08334653]);
    const offset = Array1D.new([0.69398749, -1.29056387, 0.9429723]);
    const scale = Array1D.new([-0.5607271, 0.9878457, 0.25181573]);
    const varianceEpsilon = .001;

    const result = math.batchNormalization3D(
        x, mean, variance, varianceEpsilon, scale, offset);

    test_util.expectArraysClose(
        result.getValues(), new Float32Array([
          0.59352049, -0.66135202, 0.5610874, -0.92077015, -1.45341019,
          1.52106473, -0.07704776, 0.26144429, 1.28010017, -1.14422404,
          -1.15776136, 1.15425493, 1.82644104, -0.52249442, 1.04803919,
          0.74932291, 0.40568101, 1.2844412
        ]),
        1e-5);
  });
});
