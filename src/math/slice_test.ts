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

import {Array1D, Array2D, Array3D, Array4D} from './ndarray';

// math.slice1D
{
  const tests: MathTests = it => {
    it('slices 1x1 into 1x1 (effectively a copy)', math => {
      const a = Array1D.new([5]);
      const result = math.slice1D(a, 0, 1);

      expect(result.shape).toEqual([1]);
      test_util.expectNumbersClose(result.get(0), 5);
    });

    it('slices 5x1 into shape 2x1 starting at 3', math => {
      const a = Array1D.new([1, 2, 3, 4, 5]);
      const result = math.slice1D(a, 3, 2);

      expect(result.shape).toEqual([2]);
      test_util.expectArraysClose(result, [4, 5]);
    });

    it('slices 5x1 into shape 3x1 starting at 1', math => {
      const a = Array1D.new([1, 2, 3, 4, 5]);
      const result = math.slice1D(a, 1, 3);

      expect(result.shape).toEqual([3]);
      test_util.expectArraysClose(result, [2, 3, 4]);
    });
  };

  test_util.describeMathCPU('slice1D', [tests]);
  test_util.describeMathGPU('slice1D', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.slice2D
{
  const tests: MathTests = it => {
    it('slicing a 1x1 from a 1x1 returns a 1x1', math => {
      const a = Array2D.new([1, 1], [0]);
      const b = math.slice2D(a, [0, 0], [1, 1]);
      expect(b.shape).toEqual([1, 1]);
    });

    it('returns a ndarray of slice size', math => {
      const a = Array2D.zeros([100, 100]);
      const b = math.slice2D(a, [0, 0], [12, 34]);
      expect(b.shape).toEqual([12, 34]);
    });

    it('returns the upper-left submatrix when begin is [0, 0]', math => {
      const a = Array2D.randUniform([10, 10], -1, 1);
      const b = math.slice2D(a, [0, 0], [2, 2]);
      const aValues = a.dataSync();

      test_util.expectArraysClose(
          b, [aValues[0], aValues[1], aValues[10], aValues[11]]);
    });

    it('returns the rectangle specified', math => {
      const a = Array2D.new([4, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
      const b = math.slice2D(a, [1, 1], [3, 2]);

      test_util.expectArraysClose(b, [5, 6, 8, 9, 11, 12]);
    });

    it('throws when requesting out of bounds slice', math => {
      const a = Array2D.new([4, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
      expect(() => math.slice2D(a, [1, 1], [10, 10])).toThrowError();
    });
  };

  test_util.describeMathCPU('slice2D', [tests]);
  test_util.describeMathGPU('slice2D', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.slice3D
{
  const tests: MathTests = it => {
    it('slices 1x1x1 into shape 1x1x1 (effectively a copy)', math => {
      const a = Array3D.new([1, 1, 1], [[[5]]]);
      const result = math.slice3D(a, [0, 0, 0], [1, 1, 1]);

      expect(result.shape).toEqual([1, 1, 1]);
      test_util.expectArraysClose(result, [5]);
    });

    it('slices 2x2x2 array into 1x2x2 starting at [1, 0, 0]', math => {
      const a = Array3D.new([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
      const result = math.slice3D(a, [1, 0, 0], [1, 2, 2]);

      expect(result.shape).toEqual([1, 2, 2]);
      test_util.expectArraysClose(result, [5, 6, 7, 8]);
    });

    it('slices 2x2x2 array into 2x1x1 starting at [0, 1, 1]', math => {
      const a = Array3D.new([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
      const result = math.slice3D(a, [0, 1, 1], [2, 1, 1]);

      expect(result.shape).toEqual([2, 1, 1]);
      test_util.expectArraysClose(result, [4, 8]);
    });
  };

  test_util.describeMathCPU('slice3D', [tests]);
  test_util.describeMathGPU('slice3D', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.slice4D
{
  const tests: MathTests = it => {
    it('slices 1x1x1x1 into shape 1x1x1x1 (effectively a copy)', math => {
      const a = Array4D.new([1, 1, 1, 1], [[[[5]]]]);
      const result = math.slice4D(a, [0, 0, 0, 0], [1, 1, 1, 1]);

      expect(result.shape).toEqual([1, 1, 1, 1]);
      test_util.expectArraysClose(result, [5]);
    });

    it('slices 2x2x2x2 array into 1x2x2x2 starting at [1, 0, 0, 0]', math => {
      const a = Array4D.new(
          [2, 2, 2, 2],
          [1, 2, 3, 4, 5, 6, 7, 8, 11, 22, 33, 44, 55, 66, 77, 88]);
      const result = math.slice4D(a, [1, 0, 0, 0], [1, 2, 2, 2]);

      expect(result.shape).toEqual([1, 2, 2, 2]);
      test_util.expectArraysClose(result, [11, 22, 33, 44, 55, 66, 77, 88]);
    });

    it('slices 2x2x2x2 array into 2x1x1x1 starting at [0, 1, 1, 1]', math => {
      const a = Array4D.new(
          [2, 2, 2, 2],
          [1, 2, 3, 4, 5, 6, 7, 8, 11, 22, 33, 44, 55, 66, 77, 88]);
      const result = math.slice4D(a, [0, 1, 1, 1], [2, 1, 1, 1]);

      expect(result.shape).toEqual([2, 1, 1, 1]);
      test_util.expectArraysClose(result, [8, 88]);
    });
  };

  test_util.describeMathCPU('slice4D', [tests]);
  test_util.describeMathGPU('slice4D', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
