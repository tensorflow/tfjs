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

import {Array1D, Array2D, Array3D} from './ndarray';

// math.tile
{
  const tests: MathTests = it => {
    it('1D (tile)', math => {
      const t = Array1D.new([1, 2, 3]);
      const t2 = math.tile(t, [2]);

      expect(t2.shape).toEqual([6]);
      test_util.expectArraysClose(t2, [1, 2, 3, 1, 2, 3]);
    });

    it('2D (tile)', math => {
      const t = Array2D.new([2, 2], [1, 11, 2, 22]);
      let t2 = math.tile(t, [1, 2]);

      expect(t2.shape).toEqual([2, 4]);
      test_util.expectArraysClose(t2, [1, 11, 1, 11, 2, 22, 2, 22]);

      t2 = math.tile(t, [2, 1]);
      expect(t2.shape).toEqual([4, 2]);
      test_util.expectArraysClose(t2, [1, 11, 2, 22, 1, 11, 2, 22]);

      t2 = math.tile(t, [2, 2]);
      expect(t2.shape).toEqual([4, 4]);
      test_util.expectArraysClose(
          t2, [1, 11, 1, 11, 2, 22, 2, 22, 1, 11, 1, 11, 2, 22, 2, 22]);
    });

    it('3D (tile)', math => {
      const t = Array3D.new([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
      const t2 = math.tile(t, [1, 2, 1]);

      expect(t2.shape).toEqual([2, 4, 2]);
      test_util.expectArraysClose(
          t2, [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8]);
    });

    it('propagates NaNs', math => {
      const t = Array1D.new([1, 2, NaN]);

      const t2 = math.tile(t, [2]);

      expect(t2.shape).toEqual([6]);
      test_util.expectArraysClose(t2, [1, 2, NaN, 1, 2, NaN]);
    });

    it('1D bool (tile)', math => {
      const t = Array1D.new([true, false, true], 'bool');
      const t2 = math.tile(t, [2]);

      expect(t2.shape).toEqual([6]);
      expect(t2.dtype).toBe('bool');
      test_util.expectArraysEqual(t2, [1, 0, 1, 1, 0, 1]);
    });

    it('2D bool (tile)', math => {
      const t = Array2D.new([2, 2], [true, false, true, true], 'bool');
      let t2 = math.tile(t, [1, 2]);

      expect(t2.shape).toEqual([2, 4]);
      expect(t2.dtype).toBe('bool');
      test_util.expectArraysEqual(t2, [1, 0, 1, 0, 1, 1, 1, 1]);

      t2 = math.tile(t, [2, 1]);
      expect(t2.shape).toEqual([4, 2]);
      expect(t2.dtype).toBe('bool');
      test_util.expectArraysEqual(t2, [1, 0, 1, 1, 1, 0, 1, 1]);

      t2 = math.tile(t, [2, 2]);
      expect(t2.shape).toEqual([4, 4]);
      expect(t2.dtype).toBe('bool');
      test_util.expectArraysEqual(
          t2, [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1]);
    });

    it('3D bool (tile)', math => {
      const t = Array3D.new(
          [2, 2, 2], [true, false, true, false, true, false, true, false],
          'bool');
      const t2 = math.tile(t, [1, 2, 1]);

      expect(t2.shape).toEqual([2, 4, 2]);
      expect(t2.dtype).toBe('bool');
      test_util.expectArraysEqual(
          t2, [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]);
    });

    it('bool propagates NaNs', math => {
      const t = Array1D.new([true, false, NaN] as boolean[], 'bool');
      const t2 = math.tile(t, [2]);

      expect(t2.shape).toEqual([6]);
      expect(t2.dtype).toBe('bool');
      test_util.expectArraysEqual(
          t2, [1, 0, util.getNaN('bool'), 1, 0, util.getNaN('bool')]);
    });

    it('1D int32 (tile)', math => {
      const t = Array1D.new([1, 2, 5], 'int32');
      const t2 = math.tile(t, [2]);

      expect(t2.shape).toEqual([6]);
      expect(t2.dtype).toBe('int32');
      test_util.expectArraysEqual(t2, [1, 2, 5, 1, 2, 5]);
    });

    it('2D int32 (tile)', math => {
      const t = Array2D.new([2, 2], [1, 2, 3, 4], 'int32');
      let t2 = math.tile(t, [1, 2]);

      expect(t2.shape).toEqual([2, 4]);
      expect(t2.dtype).toBe('int32');
      test_util.expectArraysEqual(t2, [1, 2, 1, 2, 3, 4, 3, 4]);

      t2 = math.tile(t, [2, 1]);
      expect(t2.shape).toEqual([4, 2]);
      expect(t2.dtype).toBe('int32');
      test_util.expectArraysEqual(t2, [1, 2, 3, 4, 1, 2, 3, 4]);

      t2 = math.tile(t, [2, 2]);
      expect(t2.shape).toEqual([4, 4]);
      expect(t2.dtype).toBe('int32');
      test_util.expectArraysEqual(
          t2, [1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4]);
    });

    it('3D int32 (tile)', math => {
      const t = Array3D.new([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8], 'int32');
      const t2 = math.tile(t, [1, 2, 1]);

      expect(t2.shape).toEqual([2, 4, 2]);
      expect(t2.dtype).toBe('int32');
      test_util.expectArraysEqual(
          t2, [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8]);
    });

    it('int32 propagates NaNs', math => {
      const t = Array1D.new([1, 3, NaN], 'int32');
      const t2 = math.tile(t, [2]);

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
