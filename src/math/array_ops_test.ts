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

import * as dl from '../index';
import * as test_util from '../test_util';
import {MathTests} from '../test_util';
import * as util from '../util';
import {Array1D, Array2D, Array3D, Array4D, NDArray} from './ndarray';
import {Rank} from './types';

const testsZeros: MathTests = it => {
  it('1D default dtype', () => {
    const a = dl.zeros<Rank.R1>([3]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    test_util.expectArraysClose(a, [0, 0, 0]);
  });

  it('1D float32 dtype', () => {
    const a = dl.zeros<Rank.R1>([3], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    test_util.expectArraysClose(a, [0, 0, 0]);
  });

  it('1D int32 dtype', () => {
    const a = dl.zeros<Rank.R1>([3], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3]);
    test_util.expectArraysEqual(a, [0, 0, 0]);
  });

  it('1D bool dtype', () => {
    const a = dl.zeros<Rank.R1>([3], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([3]);
    test_util.expectArraysEqual(a, [0, 0, 0]);
  });

  it('2D default dtype', () => {
    const a = dl.zeros<Rank.R1>([3, 2]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2]);
    test_util.expectArraysClose(a, [0, 0, 0, 0, 0, 0]);
  });

  it('2D float32 dtype', () => {
    const a = dl.zeros<Rank.R1>([3, 2], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2]);
    test_util.expectArraysClose(a, [0, 0, 0, 0, 0, 0]);
  });

  it('2D int32 dtype', () => {
    const a = dl.zeros<Rank.R1>([3, 2], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3, 2]);
    test_util.expectArraysEqual(a, [0, 0, 0, 0, 0, 0]);
  });

  it('2D bool dtype', () => {
    const a = dl.zeros<Rank.R1>([3, 2], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([3, 2]);
    test_util.expectArraysEqual(a, [0, 0, 0, 0, 0, 0]);
  });

  it('3D default dtype', () => {
    const a = dl.zeros<Rank.R1>([2, 2, 2]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 2]);
    test_util.expectArraysClose(a, [0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it('3D float32 dtype', () => {
    const a = dl.zeros<Rank.R1>([2, 2, 2], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 2]);
    test_util.expectArraysClose(a, [0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it('3D int32 dtype', () => {
    const a = dl.zeros<Rank.R1>([2, 2, 2], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 2]);
    test_util.expectArraysEqual(a, [0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it('3D bool dtype', () => {
    const a = dl.zeros<Rank.R1>([2, 2, 2], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([2, 2, 2]);
    test_util.expectArraysEqual(a, [0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it('4D default dtype', () => {
    const a = dl.zeros<Rank.R1>([3, 2, 1, 1]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    test_util.expectArraysClose(a, [0, 0, 0, 0, 0, 0]);
  });

  it('4D float32 dtype', () => {
    const a = dl.zeros<Rank.R1>([3, 2, 1, 1], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    test_util.expectArraysClose(a, [0, 0, 0, 0, 0, 0]);
  });

  it('4D int32 dtype', () => {
    const a = dl.zeros<Rank.R1>([3, 2, 1, 1], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    test_util.expectArraysEqual(a, [0, 0, 0, 0, 0, 0]);
  });

  it('4D bool dtype', () => {
    const a = dl.zeros<Rank.R1>([3, 2, 1, 1], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    test_util.expectArraysEqual(a, [0, 0, 0, 0, 0, 0]);
  });
};
const testsOnes: MathTests = it => {
  it('1D default dtype', () => {
    const a = dl.ones<Rank.R1>([3]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    test_util.expectArraysClose(a, [1, 1, 1]);
  });

  it('1D float32 dtype', () => {
    const a = dl.ones<Rank.R1>([3], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    test_util.expectArraysClose(a, [1, 1, 1]);
  });

  it('1D int32 dtype', () => {
    const a = dl.ones<Rank.R1>([3], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3]);
    test_util.expectArraysEqual(a, [1, 1, 1]);
  });

  it('1D bool dtype', () => {
    const a = dl.ones<Rank.R1>([3], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([3]);
    test_util.expectArraysEqual(a, [1, 1, 1]);
  });

  it('2D default dtype', () => {
    const a = dl.ones<Rank.R2>([3, 2]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2]);
    test_util.expectArraysClose(a, [1, 1, 1, 1, 1, 1]);
  });

  it('2D float32 dtype', () => {
    const a = dl.ones<Rank.R2>([3, 2], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2]);
    test_util.expectArraysClose(a, [1, 1, 1, 1, 1, 1]);
  });

  it('2D int32 dtype', () => {
    const a = dl.ones<Rank.R2>([3, 2], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3, 2]);
    test_util.expectArraysEqual(a, [1, 1, 1, 1, 1, 1]);
  });

  it('2D bool dtype', () => {
    const a = dl.ones<Rank.R2>([3, 2], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([3, 2]);
    test_util.expectArraysEqual(a, [1, 1, 1, 1, 1, 1]);
  });

  it('3D default dtype', () => {
    const a = dl.ones<Rank.R3>([2, 2, 2]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 2]);
    test_util.expectArraysClose(a, [1, 1, 1, 1, 1, 1, 1, 1]);
  });

  it('3D float32 dtype', () => {
    const a = dl.ones<Rank.R3>([2, 2, 2], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 2]);
    test_util.expectArraysClose(a, [1, 1, 1, 1, 1, 1, 1, 1]);
  });

  it('3D int32 dtype', () => {
    const a = dl.ones<Rank.R3>([2, 2, 2], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 2]);
    test_util.expectArraysEqual(a, [1, 1, 1, 1, 1, 1, 1, 1]);
  });

  it('3D bool dtype', () => {
    const a = dl.ones<Rank.R3>([2, 2, 2], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([2, 2, 2]);
    test_util.expectArraysEqual(a, [1, 1, 1, 1, 1, 1, 1, 1]);
  });

  it('4D default dtype', () => {
    const a = dl.ones<Rank.R4>([3, 2, 1, 1]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    test_util.expectArraysClose(a, [1, 1, 1, 1, 1, 1]);
  });

  it('4D float32 dtype', () => {
    const a = dl.ones<Rank.R4>([3, 2, 1, 1], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    test_util.expectArraysClose(a, [1, 1, 1, 1, 1, 1]);
  });

  it('4D int32 dtype', () => {
    const a = dl.ones<Rank.R4>([3, 2, 1, 1], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    test_util.expectArraysEqual(a, [1, 1, 1, 1, 1, 1]);
  });

  it('4D bool dtype', () => {
    const a = dl.ones<Rank.R4>([3, 2, 1, 1], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    test_util.expectArraysEqual(a, [1, 1, 1, 1, 1, 1]);
  });
};
const testsZerosLike: MathTests = it => {
  it('1D default dtype', () => {
    const a = Array1D.new([1, 2, 3]);
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    test_util.expectArraysClose(b, [0, 0, 0]);
  });

  it('1D float32 dtype', () => {
    const a = Array1D.new([1, 2, 3], 'float32');
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    test_util.expectArraysClose(b, [0, 0, 0]);
  });

  it('1D int32 dtype', () => {
    const a = Array1D.new([1, 2, 3], 'int32');
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([3]);
    test_util.expectArraysEqual(b, [0, 0, 0]);
  });

  it('1D bool dtype', () => {
    const a = Array1D.new([1, 2, 3], 'bool');
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([3]);
    test_util.expectArraysEqual(b, [0, 0, 0]);
  });

  it('2D default dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4]);
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    test_util.expectArraysClose(b, [0, 0, 0, 0]);
  });

  it('2D float32 dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4], 'float32');
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    test_util.expectArraysClose(b, [0, 0, 0, 0]);
  });

  it('2D int32 dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4], 'int32');
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2]);
    test_util.expectArraysEqual(b, [0, 0, 0, 0]);
  });

  it('2D bool dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4], 'bool');
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2]);
    test_util.expectArraysEqual(b, [0, 0, 0, 0]);
  });

  it('3D default dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    test_util.expectArraysClose(b, [0, 0, 0, 0]);
  });

  it('3D float32 dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4], 'float32');
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    test_util.expectArraysClose(b, [0, 0, 0, 0]);
  });

  it('3D int32 dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4], 'int32');
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1]);
    test_util.expectArraysEqual(b, [0, 0, 0, 0]);
  });

  it('3D bool dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4], 'bool');
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1]);
    test_util.expectArraysEqual(b, [0, 0, 0, 0]);
  });

  it('4D default dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4]);
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysClose(b, [0, 0, 0, 0]);
  });

  it('4D float32 dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'float32');
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysClose(b, [0, 0, 0, 0]);
  });

  it('4D int32 dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'int32');
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysEqual(b, [0, 0, 0, 0]);
  });

  it('4D bool dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'bool');
    const b = dl.zerosLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysEqual(b, [0, 0, 0, 0]);
  });
};
const testsOnesLike: MathTests = it => {
  it('1D default dtype', () => {
    const a = Array1D.new([1, 2, 3]);
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    test_util.expectArraysClose(b, [1, 1, 1]);
  });

  it('1D float32 dtype', () => {
    const a = Array1D.new([1, 2, 3], 'float32');
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    test_util.expectArraysClose(b, [1, 1, 1]);
  });

  it('1D int32 dtype', () => {
    const a = Array1D.new([1, 2, 3], 'int32');
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([3]);
    test_util.expectArraysEqual(b, [1, 1, 1]);
  });

  it('1D bool dtype', () => {
    const a = Array1D.new([1, 2, 3], 'bool');
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([3]);
    test_util.expectArraysEqual(b, [1, 1, 1]);
  });

  it('2D default dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4]);
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    test_util.expectArraysClose(b, [1, 1, 1, 1]);
  });

  it('2D float32 dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4], 'float32');
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    test_util.expectArraysClose(b, [1, 1, 1, 1]);
  });

  it('2D int32 dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4], 'int32');
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2]);
    test_util.expectArraysEqual(b, [1, 1, 1, 1]);
  });

  it('2D bool dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4], 'bool');
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2]);
    test_util.expectArraysEqual(b, [1, 1, 1, 1]);
  });

  it('3D default dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    test_util.expectArraysClose(b, [1, 1, 1, 1]);
  });

  it('3D float32 dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4], 'float32');
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    test_util.expectArraysClose(b, [1, 1, 1, 1]);
  });

  it('3D int32 dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4], 'int32');
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1]);
    test_util.expectArraysEqual(b, [1, 1, 1, 1]);
  });

  it('3D bool dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4], 'bool');
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1]);
    test_util.expectArraysEqual(b, [1, 1, 1, 1]);
  });

  it('4D default dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4]);
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysClose(b, [1, 1, 1, 1]);
  });

  it('4D float32 dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'float32');
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysClose(b, [1, 1, 1, 1]);
  });

  it('4D int32 dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'int32');
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysEqual(b, [1, 1, 1, 1]);
  });

  it('4D bool dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'bool');
    const b = dl.onesLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysEqual(b, [1, 1, 1, 1]);
  });
};

const testsRand: MathTests = it => {
  it('should return a random 1D float32 array', () => {
    const shape: [number] = [10];

    // Enusre defaults to float32 w/o type:
    let result = dl.rand(shape, () => util.randUniform(0, 2));
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 2);

    result = dl.rand(shape, () => util.randUniform(0, 1.5));
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 1D int32 array', () => {
    const shape: [number] = [10];
    const result = dl.rand(shape, () => util.randUniform(0, 2), 'int32');
    expect(result.dtype).toBe('int32');
    test_util.expectValuesInRange(result, 0, 2);
  });

  it('should return a random 1D bool array', () => {
    const shape: [number] = [10];
    const result = dl.rand(shape, () => util.randUniform(0, 1), 'bool');
    expect(result.dtype).toBe('bool');
    test_util.expectValuesInRange(result, 0, 1);
  });

  it('should return a random 2D float32 array', () => {
    const shape: [number] = [3, 4];

    // Enusre defaults to float32 w/o type:
    let result = dl.rand(shape, () => util.randUniform(0, 2.5));
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 2.5);

    result = dl.rand(shape, () => util.randUniform(0, 1.5), 'float32');
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 2D int32 array', () => {
    const shape: [number] = [3, 4];
    const result = dl.rand(shape, () => util.randUniform(0, 2), 'int32');
    expect(result.dtype).toBe('int32');
    test_util.expectValuesInRange(result, 0, 2);
  });

  it('should return a random 2D bool array', () => {
    const shape: [number] = [3, 4];
    const result = dl.rand(shape, () => util.randUniform(0, 1), 'bool');
    expect(result.dtype).toBe('bool');
    test_util.expectValuesInRange(result, 0, 1);
  });

  it('should return a random 3D float32 array', () => {
    const shape: [number] = [3, 4, 5];

    // Enusre defaults to float32 w/o type:
    let result = dl.rand(shape, () => util.randUniform(0, 2.5));
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 2.5);

    result = dl.rand(shape, () => util.randUniform(0, 1.5), 'float32');
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 3D int32 array', () => {
    const shape: [number] = [3, 4, 5];
    const result = dl.rand(shape, () => util.randUniform(0, 2), 'int32');
    expect(result.dtype).toBe('int32');
    test_util.expectValuesInRange(result, 0, 2);
  });

  it('should return a random 3D bool array', () => {
    const shape: [number] = [3, 4, 5];
    const result = dl.rand(shape, () => util.randUniform(0, 1), 'bool');
    expect(result.dtype).toBe('bool');
    test_util.expectValuesInRange(result, 0, 1);
  });

  it('should return a random 4D float32 array', () => {
    const shape: [number] = [3, 4, 5, 6];

    // Enusre defaults to float32 w/o type:
    let result = dl.rand(shape, () => util.randUniform(0, 2.5));
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 2.5);

    result = dl.rand(shape, () => util.randUniform(0, 1.5));
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 4D int32 array', () => {
    const shape: [number] = [3, 4, 5, 6];
    const result = dl.rand(shape, () => util.randUniform(0, 2), 'int32');
    expect(result.dtype).toBe('int32');
    test_util.expectValuesInRange(result, 0, 2);
  });

  it('should return a random 4D bool array', () => {
    const shape: [number] = [3, 4, 5, 6];
    const result = dl.rand(shape, () => util.randUniform(0, 1), 'bool');
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
    let result = dl.randNormal([SAMPLES], 0, 0.5, null, SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES]);
    test_util.jarqueBeraNormalityTest(result);
    test_util.expectArrayInMeanStdRange(result, 0, 0.5, EPSILON);

    result = dl.randNormal([SAMPLES], 0, 1.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES]);
    test_util.jarqueBeraNormalityTest(result);
    test_util.expectArrayInMeanStdRange(result, 0, 1.5, EPSILON);
  });

  it('should return a int32 1D of random normal values', () => {
    const SAMPLES = 10000;
    const result = dl.randNormal([SAMPLES], 0, 2, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual([SAMPLES]);
    test_util.jarqueBeraNormalityTest(result);
    test_util.expectArrayInMeanStdRange(result, 0, 2, EPSILON);
  });

  it('should return a float32 2D of random normal values', () => {
    const SAMPLES = 250;

    // Ensure defaults to float32.
    let result = dl.randNormal([SAMPLES, SAMPLES], 0, 2.5, null, SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES]);
    test_util.jarqueBeraNormalityTest(result);
    test_util.expectArrayInMeanStdRange(result, 0, 2.5, EPSILON);

    result = dl.randNormal([SAMPLES, SAMPLES], 0, 3.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES]);
    test_util.jarqueBeraNormalityTest(result);
    test_util.expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);
  });

  it('should return a int32 2D of random normal values', () => {
    const SAMPLES = 100;
    const result = dl.randNormal([SAMPLES, SAMPLES], 0, 2, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES]);
    test_util.jarqueBeraNormalityTest(result);
    test_util.expectArrayInMeanStdRange(result, 0, 2, EPSILON);
  });

  it('should return a float32 3D of random normal values', () => {
    const SAMPLES = 50;

    // Ensure defaults to float32.
    let result = dl.randNormal([SAMPLES, SAMPLES, SAMPLES], 0, 0.5, null, SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES, SAMPLES]);
    test_util.jarqueBeraNormalityTest(result);
    test_util.expectArrayInMeanStdRange(result, 0, 0.5, EPSILON);

    result =
        dl.randNormal([SAMPLES, SAMPLES, SAMPLES], 0, 1.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES, SAMPLES]);
    test_util.jarqueBeraNormalityTest(result);
    test_util.expectArrayInMeanStdRange(result, 0, 1.5, EPSILON);
  });

  it('should return a int32 3D of random normal values', () => {
    const SAMPLES = 50;
    const result =
        dl.randNormal([SAMPLES, SAMPLES, SAMPLES], 0, 2, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES, SAMPLES]);
    test_util.jarqueBeraNormalityTest(result);
    test_util.expectArrayInMeanStdRange(result, 0, 2, EPSILON);
  });

  it('should return a float32 4D of random normal values', () => {
    const SAMPLES = 25;

    // Ensure defaults to float32.
    let result =
        dl.randNormal([SAMPLES, SAMPLES, SAMPLES, SAMPLES], 0, 0.5, null, SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES, SAMPLES, SAMPLES]);
    test_util.jarqueBeraNormalityTest(result);
    test_util.expectArrayInMeanStdRange(result, 0, 0.5, EPSILON);

    result = dl.randNormal(
        [SAMPLES, SAMPLES, SAMPLES, SAMPLES], 0, 1.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES, SAMPLES, SAMPLES]);
    test_util.jarqueBeraNormalityTest(result);
    test_util.expectArrayInMeanStdRange(result, 0, 1.5, EPSILON);
  });

  it('should return a int32 4D of random normal values', () => {
    const SAMPLES = 25;

    const result = dl.randNormal(
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
    let result = dl.randTruncatedNormal(shape, 0, 3.5, null, SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 3.5);
    test_util.expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);

    result = dl.randTruncatedNormal(shape, 0, 4.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 4.5);
    test_util.expectArrayInMeanStdRange(result, 0, 4.5, EPSILON);
  });

  it('should return a randon 1D int32 array', () => {
    const shape: [number] = [1000];
    const result = dl.randTruncatedNormal(shape, 0, 5, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    assertTruncatedValues(result, 0, 5);
    test_util.expectArrayInMeanStdRange(result, 0, 5, EPSILON);
  });

  it('should return a 2D float32 array', () => {
    const shape: [number, number] = [50, 50];

    // Ensure defaults to float32 w/o type:
    let result = dl.randTruncatedNormal(shape, 0, 3.5, null, SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 3.5);
    test_util.expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);

    result = dl.randTruncatedNormal(shape, 0, 4.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 4.5);
    test_util.expectArrayInMeanStdRange(result, 0, 4.5, EPSILON);
  });

  it('should return a 2D int32 array', () => {
    const shape: [number, number] = [50, 50];
    const result = dl.randTruncatedNormal(shape, 0, 5, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    assertTruncatedValues(result, 0, 5);
    test_util.expectArrayInMeanStdRange(result, 0, 5, EPSILON);
  });

  it('should return a 3D float32 array', () => {
    const shape: [number, number, number] = [10, 10, 10];

    // Ensure defaults to float32 w/o type:
    let result = dl.randTruncatedNormal(shape, 0, 3.5, null, SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 3.5);
    test_util.expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);

    result = dl.randTruncatedNormal(shape, 0, 4.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 4.5);
    test_util.expectArrayInMeanStdRange(result, 0, 4.5, EPSILON);
  });

  it('should return a 3D int32 array', () => {
    const shape: [number, number, number] = [10, 10, 10];
    const result = dl.randTruncatedNormal(shape, 0, 5, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    assertTruncatedValues(result, 0, 5);
    test_util.expectArrayInMeanStdRange(result, 0, 5, EPSILON);
  });

  it('should return a 4D float32 array', () => {
    const shape: [number, number, number, number] = [5, 5, 5, 5];

    // Ensure defaults to float32 w/o type:
    let result = dl.randTruncatedNormal(shape, 0, 3.5, null, SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 3.5);
    test_util.expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);

    result = dl.randTruncatedNormal(shape, 0, 4.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(result, 0, 4.5);
    test_util.expectArrayInMeanStdRange(result, 0, 4.5, EPSILON);
  });

  it('should return a 4D int32 array', () => {
    const shape: [number, number, number, number] = [5, 5, 5, 5];
    const result = dl.randTruncatedNormal(shape, 0, 5, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    assertTruncatedValues(result, 0, 5);
    test_util.expectArrayInMeanStdRange(result, 0, 5, EPSILON);
  });
};
const testsRandUniform: MathTests = it => {
  it('should return a random 1D float32 array', () => {
    const shape: [number] = [10];

    // Enusre defaults to float32 w/o type:
    let result = dl.randUniform(shape, 0, 2.5);
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 2.5);

    result = dl.randUniform(shape, 0, 1.5, 'float32');
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 1D int32 array', () => {
    const shape: [number] = [10];
    const result = dl.randUniform(shape, 0, 2, 'int32');
    expect(result.dtype).toBe('int32');
    test_util.expectValuesInRange(result, 0, 2);
  });

  it('should return a random 1D bool array', () => {
    const shape: [number] = [10];
    const result = dl.randUniform(shape, 0, 1, 'bool');
    expect(result.dtype).toBe('bool');
    test_util.expectValuesInRange(result, 0, 1);
  });

  it('should return a random 2D float32 array', () => {
    const shape: [number, number] = [3, 4];

    // Enusre defaults to float32 w/o type:
    let result = dl.randUniform(shape, 0, 2.5);
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 2.5);

    result = dl.randUniform(shape, 0, 1.5, 'float32');
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 2D int32 array', () => {
    const shape: [number, number] = [3, 4];
    const result = dl.randUniform(shape, 0, 2, 'int32');
    expect(result.dtype).toBe('int32');
    test_util.expectValuesInRange(result, 0, 2);
  });

  it('should return a random 2D bool array', () => {
    const shape: [number, number] = [3, 4];
    const result = dl.randUniform(shape, 0, 1, 'bool');
    expect(result.dtype).toBe('bool');
    test_util.expectValuesInRange(result, 0, 1);
  });

  it('should return a random 3D float32 array', () => {
    const shape: [number, number, number] = [3, 4, 5];

    // Enusre defaults to float32 w/o type:
    let result = dl.randUniform(shape, 0, 2.5);
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 2.5);

    result = dl.randUniform(shape, 0, 1.5, 'float32');
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 3D int32 array', () => {
    const shape: [number, number, number] = [3, 4, 5];
    const result = dl.randUniform(shape, 0, 2, 'int32');
    expect(result.dtype).toBe('int32');
    test_util.expectValuesInRange(result, 0, 2);
  });

  it('should return a random 3D bool array', () => {
    const shape: [number, number, number] = [3, 4, 5];
    const result = dl.randUniform(shape, 0, 1, 'bool');
    expect(result.dtype).toBe('bool');
    test_util.expectValuesInRange(result, 0, 1);
  });

  it('should return a random 4D float32 array', () => {
    const shape: [number, number, number, number] = [3, 4, 5, 6];

    // Enusre defaults to float32 w/o type:
    let result = dl.randUniform(shape, 0, 2.5);
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 2.5);

    result = dl.randUniform(shape, 0, 1.5, 'float32');
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result, 0, 1.5);
  });

  it('should return a random 4D int32 array', () => {
    const shape: [number, number, number, number] = [3, 4, 5, 6];
    const result = dl.randUniform(shape, 0, 2, 'int32');
    expect(result.dtype).toBe('int32');
    test_util.expectValuesInRange(result, 0, 2);
  });

  it('should return a random 4D bool array', () => {
    const shape: [number, number, number, number] = [3, 4, 5, 6];
    const result = dl.randUniform(shape, 0, 1, 'bool');
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

    const array = NDArray.fromPixels(pixels, 3);

    test_util.expectArraysEqual(array, [0, 80, 160]);
  });

  it('ImageData 1x1x4', () => {
    const pixels = new ImageData(1, 1);
    pixels.data[0] = 0;
    pixels.data[1] = 80;
    pixels.data[2] = 160;
    pixels.data[3] = 240;

    const array = NDArray.fromPixels(pixels, 4);

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

    const array = NDArray.fromPixels(pixels, 3);

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

    const array = NDArray.fromPixels(pixels, 4);

    test_util.expectArraysClose(
        array,
        new Int32Array(
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]));
  });
};
const testsClone: MathTests = it => {
  it('1D default dtype', () => {
    const a = Array1D.new([1, 2, 3]);
    const b = dl.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    test_util.expectArraysClose(b, [1, 2, 3]);
  });

  it('1D float32 dtype', () => {
    const a = Array1D.new([1, 2, 3], 'float32');
    const b = dl.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    test_util.expectArraysClose(b, [1, 2, 3]);
  });

  it('1D int32 dtype', () => {
    const a = Array1D.new([1, 2, 3], 'int32');
    const b = dl.clone(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([3]);
    test_util.expectArraysEqual(b, [1, 2, 3]);
  });

  it('1D bool dtype', () => {
    const a = Array1D.new([1, 2, 3], 'bool');
    const b = dl.clone(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([3]);
    test_util.expectArraysEqual(b, [1, 1, 1]);
  });

  it('2D default dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4]);
    const b = dl.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    test_util.expectArraysClose(b, [1, 2, 3, 4]);
  });

  it('2D float32 dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4], 'float32');
    const b = dl.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    test_util.expectArraysClose(b, [1, 2, 3, 4]);
  });

  it('2D int32 dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4], 'int32');
    const b = dl.clone(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2]);
    test_util.expectArraysEqual(b, [1, 2, 3, 4]);
  });

  it('2D bool dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4], 'bool');
    const b = dl.clone(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2]);
    test_util.expectArraysEqual(b, [1, 1, 1, 1]);
  });

  it('3D default dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
    const b = dl.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    test_util.expectArraysClose(b, [1, 2, 3, 4]);
  });

  it('3D float32 dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4], 'float32');
    const b = dl.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    test_util.expectArraysClose(b, [1, 2, 3, 4]);
  });

  it('3D int32 dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4], 'int32');
    const b = dl.clone(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1]);
    test_util.expectArraysEqual(b, [1, 2, 3, 4]);
  });

  it('3D bool dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4], 'bool');
    const b = dl.clone(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1]);
    test_util.expectArraysEqual(b, [1, 1, 1, 1]);
  });

  it('4D default dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4]);
    const b = dl.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysClose(b, [1, 2, 3, 4]);
  });

  it('4D float32 dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'float32');
    const b = dl.clone(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysClose(b, [1, 2, 3, 4]);
  });

  it('4D int32 dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'int32');
    const b = dl.clone(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysEqual(b, [1, 2, 3, 4]);
  });

  it('4D bool dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'bool');
    const b = dl.clone(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    test_util.expectArraysEqual(b, [1, 1, 1, 1]);
  });
};

const allTests = [
  testsZeros,
  testsOnes,
  testsZerosLike,
  testsOnesLike,
  testsClone,
  testsRand,
  testsRandNormal,
  testsRandTruncNormal,
  testsRandUniform,
  testsFromPixels,
];

test_util.describeMathCPU('array_ops', allTests);
test_util.describeMathGPU('array_ops', allTests, [
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
]);
