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
import {Array1D, Array2D, Array3D, Array4D, NDArray, Scalar} from './ndarray';
import {Rank} from './types';

// dl.zeros
{
  const tests: MathTests = it => {
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

  test_util.describeMathCPU('zeros', [tests]);
  test_util.describeMathGPU('zeros', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// dl.ones
{
  const tests: MathTests = it => {
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

  test_util.describeMathCPU('ones', [tests]);
  test_util.describeMathGPU('ones', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// dl.zerosLike
{
  const tests: MathTests = it => {
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

  test_util.describeMathCPU('zerosLike', [tests]);
  test_util.describeMathGPU('zerosLike', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// dl.onesLike
{
  const tests: MathTests = it => {
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

  test_util.describeMathCPU('onesLike', [tests]);
  test_util.describeMathGPU('onesLike', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// dl.rand
{
  const tests: MathTests = it => {
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

  test_util.describeMathCPU('rand', [tests]);
  test_util.describeMathGPU('rand', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// dl.randNormal
{
  const tests: MathTests = it => {
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
      let result =
          dl.randNormal([SAMPLES, SAMPLES, SAMPLES], 0, 0.5, null, SEED);
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
      let result = dl.randNormal(
          [SAMPLES, SAMPLES, SAMPLES, SAMPLES], 0, 0.5, null, SEED);
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

  test_util.describeMathCPU('randNormal', [tests]);
  test_util.describeMathGPU('randNormal', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// dl.truncatedNormal
{
  const tests: MathTests = it => {
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
      let result = dl.truncatedNormal(shape, 0, 3.5, null, SEED);
      expect(result.dtype).toBe('float32');
      assertTruncatedValues(result, 0, 3.5);
      test_util.expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);

      result = dl.truncatedNormal(shape, 0, 4.5, 'float32', SEED);
      expect(result.dtype).toBe('float32');
      assertTruncatedValues(result, 0, 4.5);
      test_util.expectArrayInMeanStdRange(result, 0, 4.5, EPSILON);
    });

    it('should return a randon 1D int32 array', () => {
      const shape: [number] = [1000];
      const result = dl.truncatedNormal(shape, 0, 5, 'int32', SEED);
      expect(result.dtype).toBe('int32');
      assertTruncatedValues(result, 0, 5);
      test_util.expectArrayInMeanStdRange(result, 0, 5, EPSILON);
    });

    it('should return a 2D float32 array', () => {
      const shape: [number, number] = [50, 50];

      // Ensure defaults to float32 w/o type:
      let result = dl.truncatedNormal(shape, 0, 3.5, null, SEED);
      expect(result.dtype).toBe('float32');
      assertTruncatedValues(result, 0, 3.5);
      test_util.expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);

      result = dl.truncatedNormal(shape, 0, 4.5, 'float32', SEED);
      expect(result.dtype).toBe('float32');
      assertTruncatedValues(result, 0, 4.5);
      test_util.expectArrayInMeanStdRange(result, 0, 4.5, EPSILON);
    });

    it('should return a 2D int32 array', () => {
      const shape: [number, number] = [50, 50];
      const result = dl.truncatedNormal(shape, 0, 5, 'int32', SEED);
      expect(result.dtype).toBe('int32');
      assertTruncatedValues(result, 0, 5);
      test_util.expectArrayInMeanStdRange(result, 0, 5, EPSILON);
    });

    it('should return a 3D float32 array', () => {
      const shape: [number, number, number] = [10, 10, 10];

      // Ensure defaults to float32 w/o type:
      let result = dl.truncatedNormal(shape, 0, 3.5, null, SEED);
      expect(result.dtype).toBe('float32');
      assertTruncatedValues(result, 0, 3.5);
      test_util.expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);

      result = dl.truncatedNormal(shape, 0, 4.5, 'float32', SEED);
      expect(result.dtype).toBe('float32');
      assertTruncatedValues(result, 0, 4.5);
      test_util.expectArrayInMeanStdRange(result, 0, 4.5, EPSILON);
    });

    it('should return a 3D int32 array', () => {
      const shape: [number, number, number] = [10, 10, 10];
      const result = dl.truncatedNormal(shape, 0, 5, 'int32', SEED);
      expect(result.dtype).toBe('int32');
      assertTruncatedValues(result, 0, 5);
      test_util.expectArrayInMeanStdRange(result, 0, 5, EPSILON);
    });

    it('should return a 4D float32 array', () => {
      const shape: [number, number, number, number] = [5, 5, 5, 5];

      // Ensure defaults to float32 w/o type:
      let result = dl.truncatedNormal(shape, 0, 3.5, null, SEED);
      expect(result.dtype).toBe('float32');
      assertTruncatedValues(result, 0, 3.5);
      test_util.expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);

      result = dl.truncatedNormal(shape, 0, 4.5, 'float32', SEED);
      expect(result.dtype).toBe('float32');
      assertTruncatedValues(result, 0, 4.5);
      test_util.expectArrayInMeanStdRange(result, 0, 4.5, EPSILON);
    });

    it('should return a 4D int32 array', () => {
      const shape: [number, number, number, number] = [5, 5, 5, 5];
      const result = dl.truncatedNormal(shape, 0, 5, 'int32', SEED);
      expect(result.dtype).toBe('int32');
      assertTruncatedValues(result, 0, 5);
      test_util.expectArrayInMeanStdRange(result, 0, 5, EPSILON);
    });
  };

  test_util.describeMathCPU('truncatedNormal', [tests]);
  test_util.describeMathGPU('truncatedNormal', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// dl.randUniform
{
  const tests: MathTests = it => {
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

  test_util.describeMathCPU('randUniform', [tests]);
  test_util.describeMathGPU('randUniform', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// dl.fromPixels
{
  const tests: MathTests = it => {
    it('ImageData 1x1x3', () => {
      const pixels = new ImageData(1, 1);
      pixels.data[0] = 0;
      pixels.data[1] = 80;
      pixels.data[2] = 160;
      pixels.data[3] = 240;

      const array = dl.fromPixels(pixels, 3);

      test_util.expectArraysEqual(array, [0, 80, 160]);
    });

    it('ImageData 1x1x4', () => {
      const pixels = new ImageData(1, 1);
      pixels.data[0] = 0;
      pixels.data[1] = 80;
      pixels.data[2] = 160;
      pixels.data[3] = 240;

      const array = dl.fromPixels(pixels, 4);

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

      const array = dl.fromPixels(pixels, 3);

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

      const array = dl.fromPixels(pixels, 4);

      test_util.expectArraysClose(
          array,
          new Int32Array(
              [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]));
    });

    it('fromPixels, 3 channels', () => {
      const pixels = new ImageData(1, 2);
      pixels.data[0] = 2;
      pixels.data[1] = 3;
      pixels.data[2] = 4;
      pixels.data[3] = 255;  // Not used.
      pixels.data[4] = 5;
      pixels.data[5] = 6;
      pixels.data[6] = 7;
      pixels.data[7] = 255;  // Not used.
      const res = dl.fromPixels(pixels, 3);
      expect(res.shape).toEqual([2, 1, 3]);
      expect(res.dtype).toBe('int32');
      test_util.expectArraysClose(res, [2, 3, 4, 5, 6, 7]);
    });

    it('fromPixels, reshape, then do dl.add()', () => {
      const pixels = new ImageData(1, 1);
      pixels.data[0] = 2;
      pixels.data[1] = 3;
      pixels.data[2] = 4;
      pixels.data[3] = 255;  // Not used.
      const a = dl.fromPixels(pixels, 3).reshape([1, 1, 1, 3]);
      const res = a.add(Scalar.new(2, 'int32'));
      expect(res.shape).toEqual([1, 1, 1, 3]);
      expect(res.dtype).toBe('int32');
      test_util.expectArraysClose(res, [4, 5, 6]);
    });

    it('fromPixels + fromPixels', () => {
      const pixelsA = new ImageData(1, 1);
      pixelsA.data[0] = 255;
      pixelsA.data[1] = 3;
      pixelsA.data[2] = 4;
      pixelsA.data[3] = 255;  // Not used.
      const pixelsB = new ImageData(1, 1);
      pixelsB.data[0] = 5;
      pixelsB.data[1] = 6;
      pixelsB.data[2] = 7;
      pixelsB.data[3] = 255;  // Not used.
      const a = dl.fromPixels(pixelsA, 3).toFloat();
      const b = dl.fromPixels(pixelsB, 3).toFloat();
      const res = a.add(b);
      expect(res.shape).toEqual([1, 1, 3]);
      expect(res.dtype).toBe('float32');
      test_util.expectArraysClose(res, [260, 9, 11]);
    });
  };

  test_util.describeMathCPU('fromPixels', [tests]);
  test_util.describeMathGPU('fromPixels', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// dl.clone
{
  const tests: MathTests = it => {
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

  test_util.describeMathCPU('clone', [tests]);
  test_util.describeMathGPU('clone', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// dl.tile
{
  const tests: MathTests = it => {
    it('1D (tile)', () => {
      const t = Array1D.new([1, 2, 3]);
      const t2 = dl.tile(t, [2]);

      expect(t2.shape).toEqual([6]);
      test_util.expectArraysClose(t2, [1, 2, 3, 1, 2, 3]);
    });

    it('2D (tile)', () => {
      const t = Array2D.new([2, 2], [1, 11, 2, 22]);
      let t2 = dl.tile(t, [1, 2]);

      expect(t2.shape).toEqual([2, 4]);
      test_util.expectArraysClose(t2, [1, 11, 1, 11, 2, 22, 2, 22]);

      t2 = dl.tile(t, [2, 1]);
      expect(t2.shape).toEqual([4, 2]);
      test_util.expectArraysClose(t2, [1, 11, 2, 22, 1, 11, 2, 22]);

      t2 = dl.tile(t, [2, 2]);
      expect(t2.shape).toEqual([4, 4]);
      test_util.expectArraysClose(
          t2, [1, 11, 1, 11, 2, 22, 2, 22, 1, 11, 1, 11, 2, 22, 2, 22]);
    });

    it('3D (tile)', () => {
      const t = Array3D.new([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
      const t2 = dl.tile(t, [1, 2, 1]);

      expect(t2.shape).toEqual([2, 4, 2]);
      test_util.expectArraysClose(
          t2, [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8]);
    });

    it('propagates NaNs', () => {
      const t = Array1D.new([1, 2, NaN]);

      const t2 = dl.tile(t, [2]);

      expect(t2.shape).toEqual([6]);
      test_util.expectArraysClose(t2, [1, 2, NaN, 1, 2, NaN]);
    });

    it('1D bool (tile)', () => {
      const t = Array1D.new([true, false, true], 'bool');
      const t2 = dl.tile(t, [2]);

      expect(t2.shape).toEqual([6]);
      expect(t2.dtype).toBe('bool');
      test_util.expectArraysEqual(t2, [1, 0, 1, 1, 0, 1]);
    });

    it('2D bool (tile)', () => {
      const t = Array2D.new([2, 2], [true, false, true, true], 'bool');
      let t2 = dl.tile(t, [1, 2]);

      expect(t2.shape).toEqual([2, 4]);
      expect(t2.dtype).toBe('bool');
      test_util.expectArraysEqual(t2, [1, 0, 1, 0, 1, 1, 1, 1]);

      t2 = dl.tile(t, [2, 1]);
      expect(t2.shape).toEqual([4, 2]);
      expect(t2.dtype).toBe('bool');
      test_util.expectArraysEqual(t2, [1, 0, 1, 1, 1, 0, 1, 1]);

      t2 = dl.tile(t, [2, 2]);
      expect(t2.shape).toEqual([4, 4]);
      expect(t2.dtype).toBe('bool');
      test_util.expectArraysEqual(
          t2, [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1]);
    });

    it('3D bool (tile)', () => {
      const t = Array3D.new(
          [2, 2, 2], [true, false, true, false, true, false, true, false],
          'bool');
      const t2 = dl.tile(t, [1, 2, 1]);

      expect(t2.shape).toEqual([2, 4, 2]);
      expect(t2.dtype).toBe('bool');
      test_util.expectArraysEqual(
          t2, [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]);
    });

    it('bool propagates NaNs', () => {
      const t = Array1D.new([true, false, NaN] as boolean[], 'bool');
      const t2 = dl.tile(t, [2]);

      expect(t2.shape).toEqual([6]);
      expect(t2.dtype).toBe('bool');
      test_util.expectArraysEqual(
          t2, [1, 0, util.getNaN('bool'), 1, 0, util.getNaN('bool')]);
    });

    it('1D int32 (tile)', () => {
      const t = Array1D.new([1, 2, 5], 'int32');
      const t2 = dl.tile(t, [2]);

      expect(t2.shape).toEqual([6]);
      expect(t2.dtype).toBe('int32');
      test_util.expectArraysEqual(t2, [1, 2, 5, 1, 2, 5]);
    });

    it('2D int32 (tile)', () => {
      const t = Array2D.new([2, 2], [1, 2, 3, 4], 'int32');
      let t2 = dl.tile(t, [1, 2]);

      expect(t2.shape).toEqual([2, 4]);
      expect(t2.dtype).toBe('int32');
      test_util.expectArraysEqual(t2, [1, 2, 1, 2, 3, 4, 3, 4]);

      t2 = dl.tile(t, [2, 1]);
      expect(t2.shape).toEqual([4, 2]);
      expect(t2.dtype).toBe('int32');
      test_util.expectArraysEqual(t2, [1, 2, 3, 4, 1, 2, 3, 4]);

      t2 = dl.tile(t, [2, 2]);
      expect(t2.shape).toEqual([4, 4]);
      expect(t2.dtype).toBe('int32');
      test_util.expectArraysEqual(
          t2, [1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4]);
    });

    it('3D int32 (tile)', () => {
      const t = Array3D.new([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8], 'int32');
      const t2 = dl.tile(t, [1, 2, 1]);

      expect(t2.shape).toEqual([2, 4, 2]);
      expect(t2.dtype).toBe('int32');
      test_util.expectArraysEqual(
          t2, [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8]);
    });

    it('int32 propagates NaNs', () => {
      const t = Array1D.new([1, 3, NaN], 'int32');
      const t2 = dl.tile(t, [2]);

      expect(t2.shape).toEqual([6]);
      expect(t2.dtype).toBe('int32');
      test_util.expectArraysEqual(
          t2, [1, 3, util.getNaN('int32'), 1, 3, util.getNaN('int32')]);
    });
  };

  test_util.describeMathCPU('tile', [tests]);
  test_util.describeMathGPU('tile', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// dl.gather
{
  const tests: MathTests = it => {
    it('1D (gather)', () => {
      const t = Array1D.new([1, 2, 3]);

      const t2 = dl.gather(t, Array1D.new([0, 2, 0, 1], 'int32'), 0);

      expect(t2.shape).toEqual([4]);
      test_util.expectArraysClose(t2, [1, 3, 1, 2]);
    });

    it('2D (gather)', () => {
      const t = Array2D.new([2, 2], [1, 11, 2, 22]);
      let t2 = dl.gather(t, Array1D.new([1, 0, 0, 1], 'int32'), 0);
      expect(t2.shape).toEqual([4, 2]);
      test_util.expectArraysClose(t2, [2, 22, 1, 11, 1, 11, 2, 22]);

      t2 = dl.gather(t, Array1D.new([1, 0, 0, 1], 'int32'), 1);
      expect(t2.shape).toEqual([2, 4]);
      test_util.expectArraysClose(t2, [11, 1, 1, 11, 22, 2, 2, 22]);
    });

    it('3D (gather)', () => {
      const t = Array3D.new([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);

      const t2 = dl.gather(t, Array1D.new([1, 0, 0, 1], 'int32'), 2);

      expect(t2.shape).toEqual([2, 2, 4]);
      test_util.expectArraysClose(
          t2, [2, 1, 1, 2, 4, 3, 3, 4, 6, 5, 5, 6, 8, 7, 7, 8]);
    });

    it('bool (gather)', () => {
      const t = Array1D.new([true, false, true], 'bool');

      const t2 = dl.gather(t, Array1D.new([0, 2, 0, 1], 'int32'), 0);

      expect(t2.shape).toEqual([4]);
      expect(t2.dtype).toBe('bool');
      expect(t2.getValues()).toEqual(new Uint8Array([1, 1, 1, 0]));
    });

    it('int32 (gather)', () => {
      const t = Array1D.new([1, 2, 5], 'int32');

      const t2 = dl.gather(t, Array1D.new([0, 2, 0, 1], 'int32'), 0);

      expect(t2.shape).toEqual([4]);
      expect(t2.dtype).toBe('int32');
      expect(t2.getValues()).toEqual(new Int32Array([1, 5, 1, 2]));
    });

    it('propagates NaNs', () => {
      const t = Array1D.new([1, 2, NaN]);

      const t2 = dl.gather(t, Array1D.new([0, 2, 0, 1], 'int32'), 0);

      expect(t2.shape).toEqual([4]);
      test_util.expectArraysClose(t2, [1, NaN, 1, 2]);
    });
  };

  test_util.describeMathCPU('gather', [tests]);
  test_util.describeMathGPU('gather', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// dl.oneHot
{
  const tests: MathTests = it => {
    it('Depth 1 throws error', () => {
      const indices = Array1D.new([0, 0, 0]);
      expect(() => dl.oneHot(indices, 1)).toThrowError();
    });

    it('Depth 2, diagonal', () => {
      const indices = Array1D.new([0, 1]);
      const res = dl.oneHot(indices, 2);

      expect(res.shape).toEqual([2, 2]);
      test_util.expectArraysClose(res, [1, 0, 0, 1]);
    });

    it('Depth 2, transposed diagonal', () => {
      const indices = Array1D.new([1, 0]);
      const res = dl.oneHot(indices, 2);

      expect(res.shape).toEqual([2, 2]);
      test_util.expectArraysClose(res, [0, 1, 1, 0]);
    });

    it('Depth 3, 4 events', () => {
      const indices = Array1D.new([2, 1, 2, 0]);
      const res = dl.oneHot(indices, 3);

      expect(res.shape).toEqual([4, 3]);
      test_util.expectArraysClose(res, [0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0]);
    });

    it('Depth 2 onValue=3, offValue=-2', () => {
      const indices = Array1D.new([0, 1]);
      const res = dl.oneHot(indices, 2, 3, -2);

      expect(res.shape).toEqual([2, 2]);
      test_util.expectArraysClose(res, [3, -2, -2, 3]);
    });
  };

  test_util.describeMathCPU('oneHot', [tests]);
  test_util.describeMathGPU('oneHot', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
