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

import {Array2D, Array3D} from './ndarray';

// math.maxPool
{
  const tests: MathTests = it => {
    it('1x1x1 in, 1x1 filter, 1 stride: [0] => [0]', math => {
      const a = Array3D.new([1, 1, 1], [0]);

      const result = math.maxPool(a, 1, 1, 0);

      test_util.expectArraysClose(result.getValues(), new Float32Array([0]));
    });

    it('3x3x1 in, 2x2 filter, 1 stride', math => {
      // Feed forward.
      const a = Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, 9, 8]);

      const result = math.maxPool(a, 2, 1, 0);

      expect(result.shape).toEqual([2, 2, 1]);
      test_util.expectArraysClose(
          result.getValues(), new Float32Array([5, 6, 9, 9]));
    });

    it('3x3x1 in, 2x2 filter, 1 stride, propagates NaNs', math => {
      const a = Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, NaN, 9]);

      const result = math.maxPool(a, 2, 1, 0);

      expect(result.shape).toEqual([2, 2, 1]);
      test_util.expectArraysClose(
          result.getValues(), new Float32Array([5, 6, NaN, NaN]));
    });

    it('3x3x2 in, 2x2 filter, 1 stride', math => {
      // Feed forward.
      const a = Array3D.new(
          [3, 3, 2],
          [1, 99, 2, 88, 3, 77, 4, 66, 5, 55, 6, 44, 7, 33, 9, 22, 8, 11]);

      const result = math.maxPool(a, 2, 1, 0);

      expect(result.shape).toEqual([2, 2, 2]);
      test_util.expectArraysClose(
          result.getValues(), new Float32Array([5, 99, 6, 88, 9, 66, 9, 55]));
    });

    it('4x4x1 in, 2x2 filter, 2 stride', math => {
      // Feed forward.
      const a = Array3D.new(
          [4, 4, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);

      const result = math.maxPool(a, 2, 2, 0);

      expect(result.shape).toEqual([2, 2, 1]);
      test_util.expectArraysClose(
          result.getValues(), new Float32Array([5, 7, 13, 15]));
    });

    it('2x2x1 in, 2x2 filter, 2 stride, pad=1', math => {
      // Feed forward.
      const a = Array3D.new([2, 2, 1], [1, 2, 3, 4]);

      const result = math.maxPool(a, 2, 2, 1);

      expect(result.shape).toEqual([2, 2, 1]);
      test_util.expectArraysClose(
          result.getValues(), new Float32Array([1, 2, 3, 4]));
    });

    it('throws when x is not rank 3', math => {
      // tslint:disable-next-line:no-any
      const a: any = Array2D.new([3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9]);

      expect(() => math.maxPool(a, 2, 1, 0)).toThrowError();

      a.dispose();
    });
  };

  test_util.describeMathCPU('maxPool', [tests]);
  test_util.describeMathGPU('maxPool', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.minPool
{
  const tests: MathTests = it => {
    it('1x1x1 in, 1x1 filter, 1 stride: [0] => [0]', math => {
      const a = Array3D.new([1, 1, 1], [0]);
      const result = math.minPool(a, 1, 1, 0);
      test_util.expectArraysClose(result.getValues(), new Float32Array([0]));
    });

    it('3x3x1 in, 2x2 filter, 1 stride', math => {
      // Feed forward.
      const a = Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, 9, 8]);
      const result = math.minPool(a, 2, 1, 0);

      expect(result.shape).toEqual([2, 2, 1]);
      test_util.expectArraysClose(
          result.getValues(), new Float32Array([1, 2, 4, 5]));
    });

    it('3x3x1 in, 2x2 filter, 1 stride, propagates NaNs', math => {
      const a = Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, NaN, 8]);
      const result = math.minPool(a, 2, 1, 0);

      expect(result.shape).toEqual([2, 2, 1]);
      test_util.expectArraysClose(
          result.getValues(), new Float32Array([1, 2, NaN, NaN]));
    });

    it('3x3x2 in, 2x2 filter, 1 stride', math => {
      // Feed forward.
      const a = Array3D.new(
          [3, 3, 2],
          [1, 99, 2, 88, 3, 77, 4, 66, 5, 55, 6, 44, 7, 33, 9, 22, 8, 11]);
      const result = math.minPool(a, 2, 1, 0);

      expect(result.shape).toEqual([2, 2, 2]);
      test_util.expectArraysClose(
          result.getValues(), new Float32Array([1, 55, 2, 44, 4, 22, 5, 11]));
    });

    it('4x4x1 in, 2x2 filter, 2 stride', math => {
      // Feed forward.
      const a = Array3D.new(
          [4, 4, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
      const result = math.minPool(a, 2, 2, 0);

      expect(result.shape).toEqual([2, 2, 1]);
      test_util.expectArraysClose(
          result.getValues(), new Float32Array([0, 2, 8, 10]));
    });

    it('2x2x1 in, 2x2 filter, 2 stride, pad=1', math => {
      // Feed forward.
      const a = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
      const result = math.minPool(a, 2, 2, 1);

      expect(result.shape).toEqual([2, 2, 1]);
      test_util.expectArraysClose(
          result.getValues(), new Float32Array([1, 2, 3, 4]));
    });
  };

  test_util.describeMathCPU('minPool', [tests]);
  test_util.describeMathGPU('minPool', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.avgPool
{
  const tests: MathTests = it => {
    it('1x1x1 in, 1x1 filter, 1 stride: [0] => [0]', math => {
      const a = Array3D.new([1, 1, 1], [0]);
      const result = math.avgPool(a, 1, 1, 0);
      test_util.expectArraysClose(result.getValues(), new Float32Array([0]));
    });

    it('3x3x1 in, 2x2 filter, 1 stride', math => {
      // Feed forward.
      const a = Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, 9, 8]);
      const result = math.avgPool(a, 2, 1, 0);

      expect(result.shape).toEqual([2, 2, 1]);
      test_util.expectArraysClose(
          result.getValues(), new Float32Array([3, 4, 6.25, 7]));
    });

    it('3x3x1 in, 2x2 filter, 1 stride, propagates NaNs', math => {
      // Feed forward.
      const a = Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, NaN, 8]);
      const result = math.avgPool(a, 2, 1, 0);

      expect(result.shape).toEqual([2, 2, 1]);
      test_util.expectArraysClose(
          result.getValues(), new Float32Array([3, 4, NaN, NaN]));
    });

    it('3x3x2 in, 2x2 filter, 1 stride', math => {
      // Feed forward.
      const a = Array3D.new(
          [3, 3, 2],
          [1, 99, 2, 88, 3, 77, 4, 66, 5, 55, 6, 44, 7, 33, 9, 22, 8, 11]);
      const result = math.avgPool(a, 2, 1, 0);

      expect(result.shape).toEqual([2, 2, 2]);
      test_util.expectArraysClose(
          result.getValues(),
          new Float32Array([3, 77, 4, 66, 6.25, 44, 7, 33]));
    });

    it('4x4x1 in, 2x2 filter, 2 stride', math => {
      // Feed forward.
      const a = Array3D.new(
          [4, 4, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
      const result = math.avgPool(a, 2, 2, 0);

      expect(result.shape).toEqual([2, 2, 1]);
      test_util.expectArraysClose(
          result.getValues(), new Float32Array([2.5, 4.5, 10.5, 12.5]));
    });

    it('2x2x1 in, 2x2 filter, 2 stride, pad=1', math => {
      // Feed forward.
      const a = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
      const result = math.avgPool(a, 2, 2, 1);

      expect(result.shape).toEqual([2, 2, 1]);
      test_util.expectArraysClose(
          result.getValues(), new Float32Array([0.25, 0.5, 0.75, 1]));
    });
  };

  test_util.describeMathCPU('avgPool', [tests]);
  test_util.describeMathGPU('avgPool', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
