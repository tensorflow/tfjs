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

import * as test_util from '../test_util';
import {MathTests} from '../test_util';
import * as util from '../util';

import {Array1D, Array2D, Array3D, Array4D, Scalar} from './ndarray';

// Equal:
{
  const boolNaN = util.getNaN('bool');

  const tests: MathTests = it => {
    // Array1D:
    it('Array1D - int32', math => {
      let a = Array1D.new([1, 4, 5], 'int32');
      let b = Array1D.new([2, 3, 5], 'int32');

      test_util.expectArraysClose(math.equal(a, b), [0, 0, 1]);

      a = Array1D.new([2, 2, 2], 'int32');
      b = Array1D.new([2, 2, 2], 'int32');
      test_util.expectArraysClose(math.equal(a, b), [1, 1, 1]);

      a = Array1D.new([0, 0], 'int32');
      b = Array1D.new([3, 3], 'int32');
      test_util.expectArraysClose(math.equal(a, b), [0, 0]);
    });
    it('Array1D - float32', math => {
      let a = Array1D.new([1.1, 4.1, 5.1], 'float32');
      let b = Array1D.new([2.2, 3.2, 5.1], 'float32');

      test_util.expectArraysClose(math.equal(a, b), [0, 0, 1]);

      a = Array1D.new([2.31, 2.31, 2.31], 'float32');
      b = Array1D.new([2.31, 2.31, 2.31], 'float32');
      test_util.expectArraysClose(math.equal(a, b), [1, 1, 1]);

      a = Array1D.new([0.45, 0.123], 'float32');
      b = Array1D.new([3.123, 3.321], 'float32');
      test_util.expectArraysClose(math.equal(a, b), [0, 0]);
    });
    it('mismatched Array1D shapes - int32', math => {
      const a = Array1D.new([1, 2], 'int32');
      const b = Array1D.new([1, 2, 3], 'int32');
      const f = () => {
        math.equal(a, b);
      };
      expect(f).toThrowError();
    });
    it('mismatched Array1D shapes - float32', math => {
      const a = Array1D.new([1.1, 2.1], 'float32');
      const b = Array1D.new([1.1, 2.1, 3.1], 'float32');
      const f = () => {
        math.equal(a, b);
      };
      expect(f).toThrowError();
    });
    it('NaNs in Array1D - int32', math => {
      const a = Array1D.new([1, NaN, 0], 'int32');
      const b = Array1D.new([0, 0, NaN], 'int32');

      test_util.expectArraysClose(math.equal(a, b), [0, boolNaN, boolNaN]);
    });
    it('NaNs in Array1D - float32', math => {
      const a = Array1D.new([1.1, NaN, 2.1], 'float32');
      const b = Array1D.new([2.1, 3.1, NaN], 'float32');
      test_util.expectArraysClose(math.equal(a, b), [0, boolNaN, boolNaN]);
    });
    it('scalar and 1D broadcast', math => {
      const a = Scalar.new(2);
      const b = Array1D.new([1, 2, 3, 4, 5, 2]);
      const res = math.equal(a, b);
      expect(res.dtype).toBe('bool');
      expect(res.shape).toEqual([6]);
      test_util.expectArraysEqual(res, [0, 1, 0, 0, 0, 1]);
    });

    // Array2D:
    it('Array2D - int32', math => {
      let a = Array2D.new([2, 3], [[1, 4, 5], [8, 9, 12]], 'int32');
      let b = Array2D.new([2, 3], [[2, 3, 6], [7, 10, 11]], 'int32');
      test_util.expectArraysClose(math.equal(a, b), [0, 0, 0, 0, 0, 0]);

      a = Array2D.new([2, 2], [[0, 0], [1, 1]], 'int32');
      b = Array2D.new([2, 2], [[0, 0], [1, 1]], 'int32');
      test_util.expectArraysClose(math.equal(a, b), [1, 1, 1, 1]);
    });
    it('Array2D - float32', math => {
      let a =
          Array2D.new([2, 3], [[1.1, 4.1, 5.1], [8.1, 9.1, 12.1]], 'float32');
      let b =
          Array2D.new([2, 3], [[2.1, 4.1, 5.1], [7.1, 10.1, 11.1]], 'float32');
      test_util.expectArraysClose(math.equal(a, b), [0, 1, 1, 0, 0, 0]);

      a = Array2D.new([2, 2], [[0.2, 0.2], [1.2, 1.2]], 'float32');
      b = Array2D.new([2, 2], [[0.2, 0.2], [1.2, 1.2]], 'float32');
      test_util.expectArraysClose(math.equal(a, b), [1, 1, 1, 1]);
    });
    it('broadcasting Array2D shapes - int32', math => {
      const a = Array2D.new([2, 1], [[3], [7]], 'int32');
      const b = Array2D.new([2, 3], [[2, 3, 4], [7, 8, 9]], 'int32');
      test_util.expectArraysClose(math.equal(a, b), [0, 1, 0, 1, 0, 0]);
    });
    it('broadcasting Array2D shapes - float32', math => {
      const a = Array2D.new([2, 1], [[1.1], [7.1]], 'float32');
      const b =
          Array2D.new([2, 3], [[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], 'float32');
      test_util.expectArraysClose(math.equal(a, b), [0, 1, 0, 1, 0, 0]);
    });
    it('NaNs in Array2D - int32', math => {
      const a = Array2D.new([2, 3], [[1, NaN, 2], [1, NaN, NaN]], 'int32');
      const b = Array2D.new([2, 3], [[0, NaN, NaN], [1, NaN, 3]], 'int32');
      test_util.expectArraysClose(
          math.equal(a, b), [0, boolNaN, boolNaN, 1, boolNaN, boolNaN]);
    });
    it('NaNs in Array2D - float32', math => {
      const a = Array2D.new([2, 2], [[1.1, NaN], [1.1, NaN]], 'float32');
      const b = Array2D.new([2, 2], [[0.1, NaN], [1.1, NaN]], 'float32');
      test_util.expectArraysClose(math.equal(a, b), [0, boolNaN, 1, boolNaN]);
    });
    it('2D and 2D broadcast each with 1 dim', math => {
      const a = Array2D.new([1, 3], [1, 2, 5]);
      const b = Array2D.new([2, 1], [5, 1]);
      const res = math.equal(a, b);
      expect(res.dtype).toBe('bool');
      expect(res.shape).toEqual([2, 3]);
      test_util.expectArraysEqual(res, [0, 0, 1, 1, 0, 0]);
    });
    it('2D and scalar broadcast', math => {
      const a = Array2D.new([2, 3], [1, 2, 3, 2, 5, 6]);
      const b = Scalar.new(2);
      const res = math.equal(a, b);
      expect(res.dtype).toBe('bool');
      expect(res.shape).toEqual([2, 3]);
      test_util.expectArraysEqual(res, [0, 1, 0, 1, 0, 0]);
    });

    // // Array3D:
    it('Array3D - int32', math => {
      let a =
          Array3D.new([2, 3, 1], [[[1], [4], [5]], [[8], [9], [12]]], 'int32');
      let b =
          Array3D.new([2, 3, 1], [[[2], [3], [6]], [[7], [10], [12]]], 'int32');
      test_util.expectArraysClose(math.equal(a, b), [0, 0, 0, 0, 0, 1]);

      a = Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
      b = Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
      test_util.expectArraysClose(math.equal(a, b), [1, 1, 1, 1, 1, 1]);
    });
    it('Array3D - float32', math => {
      let a = Array3D.new(
          [2, 3, 1], [[[1.1], [4.1], [5.1]], [[8.1], [9.1], [12.1]]],
          'float32');
      let b = Array3D.new(
          [2, 3, 1], [[[2.1], [3.1], [6.1]], [[7.1], [10.1], [12.1]]],
          'float32');
      test_util.expectArraysClose(math.equal(a, b), [0, 0, 0, 0, 0, 1]);

      a = Array3D.new(
          [2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
      b = Array3D.new(
          [2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
      test_util.expectArraysClose(math.equal(a, b), [1, 1, 1, 1, 1, 1]);
    });
    it('broadcasting Array3D shapes - int32', math => {
      const a = Array3D.new(
          [2, 3, 2], [[[1, 0], [2, 3], [4, 5]], [[6, 7], [9, 8], [10, 11]]],
          'int32');
      const b =
          Array3D.new([2, 3, 1], [[[1], [2], [3]], [[7], [10], [9]]], 'int32');
      test_util.expectArraysClose(
          math.equal(a, b), [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]);
    });
    it('broadcasting Array3D shapes - float32', math => {
      const a = Array3D.new(
          [2, 3, 2],
          [
            [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
            [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
          ],
          'float32');
      const b = Array3D.new(
          [2, 3, 1], [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]],
          'float32');
      test_util.expectArraysClose(
          math.equal(a, b), [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]);
    });
    it('NaNs in Array3D - int32', math => {
      const a =
          Array3D.new([2, 3, 1], [[[1], [NaN], [1]], [[0], [0], [0]]], 'int32');
      const b =
          Array3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [NaN]]], 'int32');
      test_util.expectArraysClose(
          math.equal(a, b), [0, boolNaN, 1, 0, 1, boolNaN]);
    });
    it('NaNs in Array3D - float32', math => {
      const a = Array3D.new(
          [2, 3, 1], [[[1.1], [NaN], [1.1]], [[0.1], [0.1], [0.1]]], 'float32');
      const b = Array3D.new(
          [2, 3, 1], [[[0.1], [0.1], [1.1]], [[1.1], [0.1], [NaN]]], 'float32');
      test_util.expectArraysClose(
          math.equal(a, b), [0, boolNaN, 1, 0, 1, boolNaN]);
    });
    it('3D and scalar', math => {
      const a = Array3D.new([2, 3, 1], [1, 2, 3, 4, 5, -1]);
      const b = Scalar.new(-1);
      const res = math.equal(a, b);
      expect(res.dtype).toBe('bool');
      expect(res.shape).toEqual([2, 3, 1]);
      test_util.expectArraysEqual(res, [0, 0, 0, 0, 0, 1]);
    });

    // Array4D:
    it('Array4D - int32', math => {
      let a = Array4D.new([2, 2, 1, 1], [1, 4, 5, 8], 'int32');
      let b = Array4D.new([2, 2, 1, 1], [2, 3, 6, 8], 'int32');
      test_util.expectArraysClose(math.equal(a, b), [0, 0, 0, 1]);

      a = Array4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
      b = Array4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
      test_util.expectArraysClose(math.equal(a, b), [1, 1, 1, 1]);

      a = Array4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'int32');
      b = Array4D.new([2, 2, 1, 1], [2, 2, 2, 2], 'int32');
      test_util.expectArraysClose(math.equal(a, b), [0, 0, 0, 0]);
    });
    it('Array4D - float32', math => {
      let a = Array4D.new([2, 2, 1, 1], [1.1, 4.1, 5.1, 8.1], 'float32');
      let b = Array4D.new([2, 2, 1, 1], [2.1, 3.1, 6.1, 8.1], 'float32');
      test_util.expectArraysClose(math.equal(a, b), [0, 0, 0, 1]);

      a = Array4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
      b = Array4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
      test_util.expectArraysClose(math.equal(a, b), [1, 1, 1, 1]);

      a = Array4D.new([2, 2, 1, 1], [0.1, 0.1, 0.1, 0.1], 'float32');
      b = Array4D.new([2, 2, 1, 1], [1.1, 1.1, 1.1, 1.1], 'float32');
      test_util.expectArraysClose(math.equal(a, b), [0, 0, 0, 0]);
    });
    it('broadcasting Array4D shapes - int32', math => {
      const a = Array4D.new([2, 2, 1, 1], [1, 2, 5, 9], 'int32');
      const b = Array4D.new(
          [2, 2, 1, 2], [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], 'int32');
      test_util.expectArraysClose(math.equal(a, b), [1, 0, 0, 0, 1, 0, 0, 0]);
    });
    it('broadcasting Array4D shapes - float32', math => {
      const a = Array4D.new([2, 2, 1, 1], [1.1, 2.1, 5.1, 9.1], 'float32');
      const b = Array4D.new(
          [2, 2, 1, 2],
          [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]],
          'float32');
      test_util.expectArraysClose(math.equal(a, b), [1, 0, 0, 0, 1, 0, 0, 0]);
    });
    it('NaNs in Array4D - int32', math => {
      const a = Array4D.new([2, 2, 1, 1], [1, NaN, 1, 0], 'int32');
      const b = Array4D.new([2, 2, 1, 1], [0, 1, 1, NaN], 'int32');
      test_util.expectArraysClose(math.equal(a, b), [0, boolNaN, 1, boolNaN]);
    });
    it('NaNs in Array4D - float32', math => {
      const a = Array4D.new([2, 2, 1, 1], [1.1, NaN, 1.1, 0.1], 'float32');
      const b = Array4D.new([2, 2, 1, 1], [0.1, 1.1, 1.1, NaN], 'float32');
      test_util.expectArraysClose(math.equal(a, b), [0, boolNaN, 1, boolNaN]);
    });
  };

  test_util.describeMathCPU('equal', [tests]);
  test_util.describeMathGPU('equal', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// EqualStrict:
{
  const boolNaN = util.getNaN('bool');

  const tests: MathTests = it => {
    // Array1D:
    it('Array1D - int32', math => {
      let a = Array1D.new([1, 4, 5], 'int32');
      let b = Array1D.new([2, 3, 5], 'int32');
      test_util.expectArraysClose(math.equalStrict(a, b), [0, 0, 1]);

      a = Array1D.new([2, 2, 2], 'int32');
      b = Array1D.new([2, 2, 2], 'int32');
      test_util.expectArraysClose(math.equalStrict(a, b), [1, 1, 1]);

      a = Array1D.new([0, 0], 'int32');
      b = Array1D.new([3, 3], 'int32');
      test_util.expectArraysClose(math.equalStrict(a, b), [0, 0]);
    });
    it('Array1D - float32', math => {
      let a = Array1D.new([1.1, 4.1, 5.1], 'float32');
      let b = Array1D.new([2.2, 3.2, 5.1], 'float32');
      test_util.expectArraysClose(math.equalStrict(a, b), [0, 0, 1]);

      a = Array1D.new([2.31, 2.31, 2.31], 'float32');
      b = Array1D.new([2.31, 2.31, 2.31], 'float32');
      test_util.expectArraysClose(math.equalStrict(a, b), [1, 1, 1]);

      a = Array1D.new([0.45, 0.123], 'float32');
      b = Array1D.new([3.123, 3.321], 'float32');
      test_util.expectArraysClose(math.equalStrict(a, b), [0, 0]);
    });
    it('mismatched Array1D shapes - int32', math => {
      const a = Array1D.new([1, 2], 'int32');
      const b = Array1D.new([1, 2, 3], 'int32');
      const f = () => {
        math.equalStrict(a, b);
      };
      expect(f).toThrowError();
    });
    it('mismatched Array1D shapes - float32', math => {
      const a = Array1D.new([1.1, 2.1], 'float32');
      const b = Array1D.new([1.1, 2.1, 3.1], 'float32');
      const f = () => {
        math.equalStrict(a, b);
      };
      expect(f).toThrowError();
    });
    it('NaNs in Array1D - int32', math => {
      const a = Array1D.new([1, NaN, 0], 'int32');
      const b = Array1D.new([0, 0, NaN], 'int32');
      test_util.expectArraysClose(
          math.equalStrict(a, b), [0, boolNaN, boolNaN]);
    });
    it('NaNs in Array1D - float32', math => {
      const a = Array1D.new([1.1, NaN, 2.1], 'float32');
      const b = Array1D.new([2.1, 3.1, NaN], 'float32');
      test_util.expectArraysClose(
          math.equalStrict(a, b), [0, boolNaN, boolNaN]);
    });

    // Array2D:
    it('Array2D - int32', math => {
      let a = Array2D.new([2, 3], [[1, 4, 5], [8, 9, 12]], 'int32');
      let b = Array2D.new([2, 3], [[2, 3, 6], [7, 10, 11]], 'int32');
      test_util.expectArraysClose(math.equalStrict(a, b), [0, 0, 0, 0, 0, 0]);

      a = Array2D.new([2, 2], [[0, 0], [1, 1]], 'int32');
      b = Array2D.new([2, 2], [[0, 0], [1, 1]], 'int32');
      test_util.expectArraysClose(math.equalStrict(a, b), [1, 1, 1, 1]);
    });
    it('Array2D - float32', math => {
      let a =
          Array2D.new([2, 3], [[1.1, 4.1, 5.1], [8.1, 9.1, 12.1]], 'float32');
      let b =
          Array2D.new([2, 3], [[2.1, 4.1, 5.1], [7.1, 10.1, 11.1]], 'float32');
      test_util.expectArraysClose(math.equalStrict(a, b), [0, 1, 1, 0, 0, 0]);

      a = Array2D.new([2, 2], [[0.2, 0.2], [1.2, 1.2]], 'float32');
      b = Array2D.new([2, 2], [[0.2, 0.2], [1.2, 1.2]], 'float32');
      test_util.expectArraysClose(math.equalStrict(a, b), [1, 1, 1, 1]);
    });
    it('mismatch Array2D shapes - int32', math => {
      const a = Array2D.new([2, 1], [[3], [7]], 'int32');
      const b = Array2D.new([2, 3], [[2, 3, 4], [7, 8, 9]], 'int32');

      const f = () => {
        math.equalStrict(a, b);
      };
      expect(f).toThrowError();
    });
    it('mismatch Array2D shapes - float32', math => {
      const a = Array2D.new([2, 1], [[1.1], [7.1]], 'float32');
      const b =
          Array2D.new([2, 3], [[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], 'float32');
      const f = () => {
        math.equalStrict(a, b);
      };
      expect(f).toThrowError();
    });
    it('NaNs in Array2D - int32', math => {
      const a = Array2D.new([2, 3], [[1, NaN, 2], [1, NaN, NaN]], 'int32');
      const b = Array2D.new([2, 3], [[0, NaN, NaN], [1, NaN, 3]], 'int32');
      test_util.expectArraysClose(
          math.equalStrict(a, b), [0, boolNaN, boolNaN, 1, boolNaN, boolNaN]);
    });
    it('NaNs in Array2D - float32', math => {
      const a = Array2D.new([2, 2], [[1.1, NaN], [1.1, NaN]], 'float32');
      const b = Array2D.new([2, 2], [[0.1, NaN], [1.1, NaN]], 'float32');
      test_util.expectArraysClose(
          math.equalStrict(a, b), [0, boolNaN, 1, boolNaN]);
    });

    // // Array3D:
    it('Array3D - int32', math => {
      let a =
          Array3D.new([2, 3, 1], [[[1], [4], [5]], [[8], [9], [12]]], 'int32');
      let b =
          Array3D.new([2, 3, 1], [[[2], [3], [6]], [[7], [10], [12]]], 'int32');
      test_util.expectArraysClose(math.equalStrict(a, b), [0, 0, 0, 0, 0, 1]);

      a = Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
      b = Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
      test_util.expectArraysClose(math.equalStrict(a, b), [1, 1, 1, 1, 1, 1]);
    });
    it('Array3D - float32', math => {
      let a = Array3D.new(
          [2, 3, 1], [[[1.1], [4.1], [5.1]], [[8.1], [9.1], [12.1]]],
          'float32');
      let b = Array3D.new(
          [2, 3, 1], [[[2.1], [3.1], [6.1]], [[7.1], [10.1], [12.1]]],
          'float32');
      test_util.expectArraysClose(math.equalStrict(a, b), [0, 0, 0, 0, 0, 1]);

      a = Array3D.new(
          [2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
      b = Array3D.new(
          [2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
      test_util.expectArraysClose(math.equalStrict(a, b), [1, 1, 1, 1, 1, 1]);
    });
    it('mismatch Array3D shapes - int32', math => {
      const a = Array3D.new(
          [2, 3, 2], [[[1, 0], [2, 3], [4, 5]], [[6, 7], [9, 8], [10, 11]]],
          'int32');
      const b =
          Array3D.new([2, 3, 1], [[[1], [2], [3]], [[7], [10], [9]]], 'int32');

      const f = () => {
        math.equalStrict(a, b);
      };
      expect(f).toThrowError();
    });
    it('mismatch Array3D shapes - float32', math => {
      const a = Array3D.new(
          [2, 3, 2],
          [
            [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
            [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
          ],
          'float32');
      const b = Array3D.new(
          [2, 3, 1], [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]],
          'float32');

      const f = () => {
        math.equalStrict(a, b);
      };
      expect(f).toThrowError();
    });
    it('NaNs in Array3D - int32', math => {
      const a =
          Array3D.new([2, 3, 1], [[[1], [NaN], [1]], [[0], [0], [0]]], 'int32');
      const b =
          Array3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [NaN]]], 'int32');
      test_util.expectArraysClose(
          math.equalStrict(a, b), [0, boolNaN, 1, 0, 1, boolNaN]);
    });
    it('NaNs in Array3D - float32', math => {
      const a = Array3D.new(
          [2, 3, 1], [[[1.1], [NaN], [1.1]], [[0.1], [0.1], [0.1]]], 'float32');
      const b = Array3D.new(
          [2, 3, 1], [[[0.1], [0.1], [1.1]], [[1.1], [0.1], [NaN]]], 'float32');
      test_util.expectArraysClose(
          math.equalStrict(a, b), [0, boolNaN, 1, 0, 1, boolNaN]);
    });

    // Array4D:
    it('Array4D - int32', math => {
      let a = Array4D.new([2, 2, 1, 1], [1, 4, 5, 8], 'int32');
      let b = Array4D.new([2, 2, 1, 1], [2, 3, 6, 8], 'int32');
      test_util.expectArraysClose(math.equalStrict(a, b), [0, 0, 0, 1]);

      a = Array4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
      b = Array4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
      test_util.expectArraysClose(math.equalStrict(a, b), [1, 1, 1, 1]);

      a = Array4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'int32');
      b = Array4D.new([2, 2, 1, 1], [2, 2, 2, 2], 'int32');
      test_util.expectArraysClose(math.equalStrict(a, b), [0, 0, 0, 0]);
    });
    it('Array4D - float32', math => {
      let a = Array4D.new([2, 2, 1, 1], [1.1, 4.1, 5.1, 8.1], 'float32');
      let b = Array4D.new([2, 2, 1, 1], [2.1, 3.1, 6.1, 8.1], 'float32');
      test_util.expectArraysClose(math.equalStrict(a, b), [0, 0, 0, 1]);

      a = Array4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
      b = Array4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
      test_util.expectArraysClose(math.equalStrict(a, b), [1, 1, 1, 1]);

      a = Array4D.new([2, 2, 1, 1], [0.1, 0.1, 0.1, 0.1], 'float32');
      b = Array4D.new([2, 2, 1, 1], [1.1, 1.1, 1.1, 1.1], 'float32');
      test_util.expectArraysClose(math.equalStrict(a, b), [0, 0, 0, 0]);
    });
    it('mismatch Array4D shapes - int32', math => {
      const a = Array4D.new([2, 2, 1, 1], [1, 2, 5, 9], 'int32');
      const b = Array4D.new(
          [2, 2, 1, 2], [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], 'int32');

      const f = () => {
        math.equalStrict(a, b);
      };
      expect(f).toThrowError();
    });
    it('mismatch Array4D shapes - float32', math => {
      const a = Array4D.new([2, 2, 1, 1], [1.1, 2.1, 5.1, 9.1], 'float32');
      const b = Array4D.new(
          [2, 2, 1, 2],
          [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]],
          'float32');

      const f = () => {
        math.equalStrict(a, b);
      };
      expect(f).toThrowError();
    });
    it('NaNs in Array4D - int32', math => {
      const a = Array4D.new([2, 2, 1, 1], [1, NaN, 1, 0], 'int32');
      const b = Array4D.new([2, 2, 1, 1], [0, 1, 1, NaN], 'int32');
      test_util.expectArraysClose(
          math.equalStrict(a, b), [0, boolNaN, 1, boolNaN]);
    });
    it('NaNs in Array4D - float32', math => {
      const a = Array4D.new([2, 2, 1, 1], [1.1, NaN, 1.1, 0.1], 'float32');
      const b = Array4D.new([2, 2, 1, 1], [0.1, 1.1, 1.1, NaN], 'float32');
      test_util.expectArraysClose(
          math.equalStrict(a, b), [0, boolNaN, 1, boolNaN]);
    });
  };

  test_util.describeMathCPU('equalStrict', [tests]);
  test_util.describeMathGPU('equalStrict', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// NotEqual:
{
  const tests: MathTests = it => {
    it('propagates NaNs', math => {
      const a = Array1D.new([2, 5, NaN]);
      const b = Array1D.new([4, 5, -1]);

      const res = math.notEqual(a, b);
      expect(res.dtype).toBe('bool');
      test_util.expectArraysEqual(res, [1, 0, util.NAN_BOOL]);
    });

    it('strict version throws when x and y are different shape', math => {
      const a = Array1D.new([2]);
      const b = Array1D.new([4, 2, -1]);

      expect(() => math.notEqualStrict(a, b)).toThrowError();
      expect(() => math.notEqualStrict(b, a)).toThrowError();
    });

    it('2D and scalar broadcast', math => {
      const a = Array2D.new([2, 3], [1, 2, 3, 2, 5, 6]);
      const b = Scalar.new(2);
      const res = math.notEqual(a, b);
      expect(res.dtype).toBe('bool');
      expect(res.shape).toEqual([2, 3]);
      test_util.expectArraysEqual(res, [1, 0, 1, 0, 1, 1]);
    });

    it('scalar and 1D broadcast', math => {
      const a = Scalar.new(2);
      const b = Array1D.new([1, 2, 3, 4, 5, 2]);
      const res = math.notEqual(a, b);
      expect(res.dtype).toBe('bool');
      expect(res.shape).toEqual([6]);
      test_util.expectArraysEqual(res, [1, 0, 1, 1, 1, 0]);
    });

    it('2D and 2D broadcast each with 1 dim', math => {
      const a = Array2D.new([1, 3], [1, 2, 5]);
      const b = Array2D.new([2, 1], [5, 1]);
      const res = math.notEqual(a, b);
      expect(res.dtype).toBe('bool');
      expect(res.shape).toEqual([2, 3]);
      test_util.expectArraysEqual(res, [1, 1, 0, 0, 1, 1]);
    });

    it('3D and scalar', math => {
      const a = Array3D.new([2, 3, 1], [1, 2, 3, 4, 5, -1]);
      const b = Scalar.new(-1);
      const res = math.notEqual(a, b);
      expect(res.dtype).toBe('bool');
      expect(res.shape).toEqual([2, 3, 1]);
      test_util.expectArraysEqual(res, [1, 1, 1, 1, 1, 0]);
    });
  };

  test_util.describeMathCPU('notEqual', [tests]);
  test_util.describeMathGPU('notEqual', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// Less:
{
  const boolNaN = util.getNaN('bool');

  const tests: MathTests = it => {
    // Array1D:
    it('Array1D - int32', math => {
      let a = Array1D.new([1, 4, 5], 'int32');
      let b = Array1D.new([2, 3, 5], 'int32');
      let res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 0]);

      a = Array1D.new([2, 2, 2], 'int32');
      b = Array1D.new([2, 2, 2], 'int32');
      res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0]);

      a = Array1D.new([0, 0], 'int32');
      b = Array1D.new([3, 3], 'int32');
      res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1]);
    });
    it('Array1D - float32', math => {
      let a = Array1D.new([1.1, 4.1, 5.1], 'float32');
      let b = Array1D.new([2.2, 3.2, 5.1], 'float32');
      let res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 0]);

      a = Array1D.new([2.31, 2.31, 2.31], 'float32');
      b = Array1D.new([2.31, 2.31, 2.31], 'float32');
      res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0]);

      a = Array1D.new([0.45, 0.123], 'float32');
      b = Array1D.new([3.123, 3.321], 'float32');
      res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1]);
    });
    it('mismatched Array1D shapes - int32', math => {
      const a = Array1D.new([1, 2], 'int32');
      const b = Array1D.new([1, 2, 3], 'int32');
      const f = () => {
        math.less(a, b);
      };
      expect(f).toThrowError();
    });
    it('mismatched Array1D shapes - float32', math => {
      const a = Array1D.new([1.1, 2.1], 'float32');
      const b = Array1D.new([1.1, 2.1, 3.1], 'float32');
      const f = () => {
        math.less(a, b);
      };
      expect(f).toThrowError();
    });
    it('NaNs in Array1D - int32', math => {
      const a = Array1D.new([1, NaN, 0], 'int32');
      const b = Array1D.new([0, 0, NaN], 'int32');
      const res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, boolNaN, boolNaN]);
    });
    it('NaNs in Array1D - float32', math => {
      const a = Array1D.new([1.1, NaN, 2.1], 'float32');
      const b = Array1D.new([2.1, 3.1, NaN], 'float32');
      const res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, boolNaN, boolNaN]);
    });

    // Array2D:
    it('Array2D - int32', math => {
      let a = Array2D.new([2, 3], [[1, 4, 5], [8, 9, 12]], 'int32');
      let b = Array2D.new([2, 3], [[2, 3, 6], [7, 10, 11]], 'int32');
      let res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1, 0, 1, 0]);

      a = Array2D.new([2, 2], [[0, 0], [1, 1]], 'int32');
      b = Array2D.new([2, 2], [[0, 0], [1, 1]], 'int32');
      res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0]);
    });
    it('Array2D - float32', math => {
      let a =
          Array2D.new([2, 3], [[1.1, 4.1, 5.1], [8.1, 9.1, 12.1]], 'float32');
      let b =
          Array2D.new([2, 3], [[2.1, 3.1, 6.1], [7.1, 10.1, 11.1]], 'float32');
      let res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1, 0, 1, 0]);

      a = Array2D.new([2, 2], [[0.2, 0.2], [1.2, 1.2]], 'float32');
      b = Array2D.new([2, 2], [[0.2, 0.2], [1.2, 1.2]], 'float32');
      res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0]);
    });
    it('broadcasting Array2D shapes - int32', math => {
      const a = Array2D.new([2, 1], [[3], [7]], 'int32');
      const b = Array2D.new([2, 3], [[2, 3, 4], [7, 8, 9]], 'int32');
      const res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 1, 0, 1, 1]);
    });
    it('broadcasting Array2D shapes - float32', math => {
      const a = Array2D.new([2, 1], [[1.1], [7.1]], 'float32');
      const b =
          Array2D.new([2, 3], [[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], 'float32');
      const res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 1, 0, 1, 1]);
    });
    it('NaNs in Array2D - int32', math => {
      const a = Array2D.new([2, 3], [[1, NaN, 2], [0, NaN, NaN]], 'int32');
      const b = Array2D.new([2, 3], [[0, NaN, NaN], [1, NaN, 3]], 'int32');
      const res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(
          res, [0, boolNaN, boolNaN, 1, boolNaN, boolNaN]);
    });
    it('NaNs in Array2D - float32', math => {
      const a = Array2D.new([2, 2], [[1.1, NaN], [0.1, NaN]], 'float32');
      const b = Array2D.new([2, 2], [[0.1, NaN], [1.1, NaN]], 'float32');
      const res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, boolNaN, 1, boolNaN]);
    });

    // Array3D:
    it('Array3D - int32', math => {
      let a =
          Array3D.new([2, 3, 1], [[[1], [4], [5]], [[8], [9], [12]]], 'int32');
      let b =
          Array3D.new([2, 3, 1], [[[2], [3], [6]], [[7], [10], [11]]], 'int32');
      let res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1, 0, 1, 0]);

      a = Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
      b = Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
      res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0, 0, 0]);
    });
    it('Array3D - float32', math => {
      let a = Array3D.new(
          [2, 3, 1], [[[1.1], [4.1], [5.1]], [[8.1], [9.1], [12.1]]],
          'float32');
      let b = Array3D.new(
          [2, 3, 1], [[[2.1], [3.1], [6.1]], [[7.1], [10.1], [11.1]]],
          'float32');
      let res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1, 0, 1, 0]);

      a = Array3D.new(
          [2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.0]]], 'float32');
      b = Array3D.new(
          [2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
      res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0, 0, 1]);
    });
    it('broadcasting Array3D shapes - int32', math => {
      const a = Array3D.new(
          [2, 3, 2], [[[1, 0], [2, 3], [4, 5]], [[6, 7], [9, 8], [10, 11]]],
          'int32');
      const b =
          Array3D.new([2, 3, 1], [[[1], [2], [3]], [[7], [10], [9]]], 'int32');
      const res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0]);
    });
    it('broadcasting Array3D float32', math => {
      const a = Array3D.new(
          [2, 3, 2],
          [
            [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
            [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
          ],
          'float32');
      const b = Array3D.new(
          [2, 3, 1], [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]],
          'float32');
      const res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0]);
    });
    it('NaNs in Array3D - int32', math => {
      const a =
          Array3D.new([2, 3, 1], [[[1], [NaN], [1]], [[0], [0], [0]]], 'int32');
      const b =
          Array3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [NaN]]], 'int32');
      const res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, boolNaN, 0, 1, 0, boolNaN]);
    });
    it('NaNs in Array3D - float32', math => {
      const a = Array3D.new(
          [2, 3, 1], [[[1.1], [NaN], [1.1]], [[0.1], [0.1], [0.1]]], 'float32');
      const b = Array3D.new(
          [2, 3, 1], [[[0.1], [0.1], [1.1]], [[1.1], [0.1], [NaN]]], 'float32');
      const res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, boolNaN, 0, 1, 0, boolNaN]);
    });

    // Array4D:
    it('Array4D - int32', math => {
      let a = Array4D.new([2, 2, 1, 1], [1, 4, 5, 8], 'int32');
      let b = Array4D.new([2, 2, 1, 1], [2, 3, 6, 7], 'int32');
      let res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1, 0]);

      a = Array4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
      b = Array4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
      res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0]);

      a = Array4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'int32');
      b = Array4D.new([2, 2, 1, 1], [2, 2, 2, 2], 'int32');
      res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1]);
    });
    it('Array4D - float32', math => {
      let a = Array4D.new([2, 2, 1, 1], [1.1, 4.1, 5.1, 8.1], 'float32');
      let b = Array4D.new([2, 2, 1, 1], [2.1, 3.1, 6.1, 7.1], 'float32');
      let res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1, 0]);

      a = Array4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
      b = Array4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
      res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0]);

      a = Array4D.new([2, 2, 1, 1], [0.1, 0.1, 0.1, 0.1], 'float32');
      b = Array4D.new([2, 2, 1, 1], [1.1, 1.1, 1.1, 1.1], 'float32');
      res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1]);
    });
    it('broadcasting Array4D shapes - int32', math => {
      const a = Array4D.new([2, 2, 1, 1], [1, 2, 5, 9], 'int32');
      const b = Array4D.new(
          [2, 2, 1, 2], [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], 'int32');
      const res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 1, 1, 0, 1, 0, 0]);
    });
    it('broadcasting Array4D shapes - float32', math => {
      const a = Array4D.new([2, 2, 1, 1], [1.1, 2.1, 5.1, 9.1], 'float32');
      const b = Array4D.new(
          [2, 2, 1, 2],
          [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]],
          'float32');
      const res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 1, 1, 0, 1, 0, 0]);
    });
    it('NaNs in Array4D - int32', math => {
      const a = Array4D.new([2, 2, 1, 1], [1, NaN, 0, 0], 'int32');
      const b = Array4D.new([2, 2, 1, 1], [0, 1, 1, NaN], 'int32');
      const res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, boolNaN, 1, boolNaN]);
    });
    it('NaNs in Array4D - float32', math => {
      const a = Array4D.new([2, 2, 1, 1], [1.1, NaN, 0.1, 0.1], 'float32');
      const b = Array4D.new([2, 2, 1, 1], [0.1, 1.1, 1.1, NaN], 'float32');
      const res = math.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, boolNaN, 1, boolNaN]);
    });
  };

  test_util.describeMathCPU('less', [tests]);
  test_util.describeMathGPU('less', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// LessEqual:
{
  const boolNaN = util.getNaN('bool');

  const tests: MathTests = it => {
    // Array1D:
    it('Array1D - int32', math => {
      let a = Array1D.new([1, 4, 5], 'int32');
      let b = Array1D.new([2, 3, 5], 'int32');
      let res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1]);

      a = Array1D.new([2, 2, 2], 'int32');
      b = Array1D.new([2, 2, 2], 'int32');
      res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1]);

      a = Array1D.new([0, 0], 'int32');
      b = Array1D.new([3, 3], 'int32');
      res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1]);
    });
    it('Array1D - float32', math => {
      let a = Array1D.new([1.1, 4.1, 5.1], 'float32');
      let b = Array1D.new([2.2, 3.2, 5.1], 'float32');
      let res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1]);

      a = Array1D.new([2.31, 2.31, 2.31], 'float32');
      b = Array1D.new([2.31, 2.31, 2.31], 'float32');
      res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1]);

      a = Array1D.new([0.45, 0.123], 'float32');
      b = Array1D.new([3.123, 3.321], 'float32');
      res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1]);
    });
    it('mismatched Array1D shapes - int32', math => {
      const a = Array1D.new([1, 2], 'int32');
      const b = Array1D.new([1, 2, 3], 'int32');
      const f = () => {
        math.lessEqual(a, b);
      };
      expect(f).toThrowError();
    });
    it('mismatched Array1D shapes - float32', math => {
      const a = Array1D.new([1.1, 2.1], 'float32');
      const b = Array1D.new([1.1, 2.1, 3.1], 'float32');
      const f = () => {
        math.lessEqual(a, b);
      };
      expect(f).toThrowError();
    });
    it('NaNs in Array1D - int32', math => {
      const a = Array1D.new([1, NaN, 0], 'int32');
      const b = Array1D.new([0, 0, NaN], 'int32');
      const res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, boolNaN, boolNaN]);
    });
    it('NaNs in Array1D - float32', math => {
      const a = Array1D.new([1.1, NaN, 2.1], 'float32');
      const b = Array1D.new([2.1, 3.1, NaN], 'float32');
      const res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, boolNaN, boolNaN]);
    });

    // Array2D:
    it('Array2D - int32', math => {
      let a = Array2D.new([2, 3], [[1, 4, 5], [8, 9, 12]], 'int32');
      let b = Array2D.new([2, 3], [[2, 3, 6], [7, 10, 11]], 'int32');
      let res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1, 0, 1, 0]);

      a = Array2D.new([2, 2], [[0, 0], [1, 1]], 'int32');
      b = Array2D.new([2, 2], [[0, 0], [1, 1]], 'int32');
      res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1]);
    });
    it('Array2D - float32', math => {
      let a =
          Array2D.new([2, 3], [[1.1, 4.1, 5.1], [8.1, 9.1, 12.1]], 'float32');
      let b =
          Array2D.new([2, 3], [[2.1, 3.1, 6.1], [7.1, 10.1, 11.1]], 'float32');
      let res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1, 0, 1, 0]);

      a = Array2D.new([2, 2], [[0.2, 0.2], [1.2, 1.2]], 'float32');
      b = Array2D.new([2, 2], [[0.2, 0.2], [1.2, 1.2]], 'float32');
      res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1]);
    });
    it('broadcasting Array2D shapes - int32', math => {
      const a = Array2D.new([2, 1], [[3], [7]], 'int32');
      const b = Array2D.new([2, 3], [[2, 3, 4], [7, 8, 9]], 'int32');
      const res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 1, 1, 1, 1]);
    });
    it('broadcasting Array2D shapes - float32', math => {
      const a = Array2D.new([2, 1], [[1.1], [7.1]], 'float32');
      const b =
          Array2D.new([2, 3], [[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], 'float32');
      const res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 1, 1, 1, 1]);
    });
    it('NaNs in Array2D - int32', math => {
      const a = Array2D.new([2, 3], [[1, NaN, 2], [0, NaN, NaN]], 'int32');
      const b = Array2D.new([2, 3], [[0, NaN, NaN], [1, NaN, 3]], 'int32');
      const res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(
          res, [0, boolNaN, boolNaN, 1, boolNaN, boolNaN]);
    });
    it('NaNs in Array2D - float32', math => {
      const a = Array2D.new([2, 2], [[1.1, NaN], [0.1, NaN]], 'float32');
      const b = Array2D.new([2, 2], [[0.1, NaN], [1.1, NaN]], 'float32');
      const res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, boolNaN, 1, boolNaN]);
    });

    // Array3D:
    it('Array3D - int32', math => {
      let a =
          Array3D.new([2, 3, 1], [[[1], [4], [5]], [[8], [9], [12]]], 'int32');
      let b =
          Array3D.new([2, 3, 1], [[[2], [3], [6]], [[7], [10], [11]]], 'int32');
      let res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1, 0, 1, 0]);

      a = Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
      b = Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
      res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1, 1, 1]);
    });
    it('Array3D - float32', math => {
      let a = Array3D.new(
          [2, 3, 1], [[[1.1], [4.1], [5.1]], [[8.1], [9.1], [12.1]]],
          'float32');
      let b = Array3D.new(
          [2, 3, 1], [[[2.1], [3.1], [6.1]], [[7.1], [10.1], [11.1]]],
          'float32');
      let res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1, 0, 1, 0]);

      a = Array3D.new(
          [2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.2]]], 'float32');
      b = Array3D.new(
          [2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
      res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1, 1, 0]);
    });
    it('broadcasting Array3D shapes - int32', math => {
      const a = Array3D.new(
          [2, 3, 2], [[[1, 0], [2, 3], [4, 5]], [[6, 7], [9, 8], [10, 11]]],
          'int32');
      const b =
          Array3D.new([2, 3, 1], [[[1], [2], [3]], [[7], [10], [9]]], 'int32');
      const res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0]);
    });
    it('broadcasting Array3D float32', math => {
      const a = Array3D.new(
          [2, 3, 2],
          [
            [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
            [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
          ],
          'float32');
      const b = Array3D.new(
          [2, 3, 1], [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]],
          'float32');
      const res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0]);
    });
    it('NaNs in Array3D - int32', math => {
      const a =
          Array3D.new([2, 3, 1], [[[1], [NaN], [1]], [[0], [0], [0]]], 'int32');
      const b =
          Array3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [NaN]]], 'int32');
      const res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, boolNaN, 1, 1, 1, boolNaN]);
    });
    it('NaNs in Array3D - float32', math => {
      const a = Array3D.new(
          [2, 3, 1], [[[1.1], [NaN], [1.1]], [[0.1], [0.1], [0.1]]], 'float32');
      const b = Array3D.new(
          [2, 3, 1], [[[0.1], [0.1], [1.1]], [[1.1], [0.1], [NaN]]], 'float32');
      const res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, boolNaN, 1, 1, 1, boolNaN]);
    });

    // Array4D:
    it('Array4D - int32', math => {
      let a = Array4D.new([2, 2, 1, 1], [1, 4, 5, 8], 'int32');
      let b = Array4D.new([2, 2, 1, 1], [2, 3, 6, 7], 'int32');
      let res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1, 0]);

      a = Array4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
      b = Array4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
      res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1]);

      a = Array4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'int32');
      b = Array4D.new([2, 2, 1, 1], [2, 2, 2, 2], 'int32');
      res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1]);
    });
    it('Array4D - float32', math => {
      let a = Array4D.new([2, 2, 1, 1], [1.1, 4.1, 5.1, 8.1], 'float32');
      let b = Array4D.new([2, 2, 1, 1], [2.1, 3.1, 6.1, 7.1], 'float32');
      let res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1, 0]);

      a = Array4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
      b = Array4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
      res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1]);

      a = Array4D.new([2, 2, 1, 1], [0.1, 0.1, 0.1, 0.1], 'float32');
      b = Array4D.new([2, 2, 1, 1], [1.1, 1.1, 1.1, 1.1], 'float32');
      res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1]);
    });
    it('broadcasting Array4D shapes - int32', math => {
      const a = Array4D.new([2, 2, 1, 1], [1, 2, 5, 9], 'int32');
      const b = Array4D.new(
          [2, 2, 1, 2], [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], 'int32');
      const res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1, 1, 1, 0, 0]);
    });
    it('broadcasting Array4D shapes - float32', math => {
      const a = Array4D.new([2, 2, 1, 1], [1.1, 2.1, 5.1, 9.1], 'float32');
      const b = Array4D.new(
          [2, 2, 1, 2],
          [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]],
          'float32');
      const res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1, 1, 1, 0, 0]);
    });
    it('NaNs in Array4D - int32', math => {
      const a = Array4D.new([2, 2, 1, 1], [1, NaN, 0, 0], 'int32');
      const b = Array4D.new([2, 2, 1, 1], [0, 1, 1, NaN], 'int32');
      const res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, boolNaN, 1, boolNaN]);
    });
    it('NaNs in Array4D - float32', math => {
      const a = Array4D.new([2, 2, 1, 1], [1.1, NaN, 0.1, 0.1], 'float32');
      const b = Array4D.new([2, 2, 1, 1], [0.1, 1.1, 1.1, NaN], 'float32');
      const res = math.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, boolNaN, 1, boolNaN]);
    });
  };

  test_util.describeMathCPU('lessEqual', [tests]);
  test_util.describeMathGPU('lessEqual', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// Greater:
{
  const boolNaN = util.getNaN('bool');

  const tests: MathTests = it => {
    // Array1D:
    it('Array1D - int32', math => {
      let a = Array1D.new([1, 4, 5], 'int32');
      let b = Array1D.new([2, 3, 5], 'int32');
      let res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0]);

      a = Array1D.new([2, 2, 2], 'int32');
      b = Array1D.new([2, 2, 2], 'int32');
      res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0]);

      a = Array1D.new([3, 3], 'int32');
      b = Array1D.new([0, 0], 'int32');
      res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1]);
    });
    it('Array1D - float32', math => {
      let a = Array1D.new([1.1, 4.1, 5.1], 'float32');
      let b = Array1D.new([2.2, 3.2, 5.1], 'float32');
      let res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0]);

      a = Array1D.new([2.31, 2.31, 2.31], 'float32');
      b = Array1D.new([2.31, 2.31, 2.31], 'float32');
      res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0]);

      a = Array1D.new([3.123, 3.321], 'float32');
      b = Array1D.new([0.45, 0.123], 'float32');
      res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1]);
    });
    it('mismatched Array1D shapes - int32', math => {
      const a = Array1D.new([1, 2], 'int32');
      const b = Array1D.new([1, 2, 3], 'int32');
      const f = () => {
        math.greater(a, b);
      };
      expect(f).toThrowError();
    });
    it('mismatched Array1D shapes - float32', math => {
      const a = Array1D.new([1.1, 2.1], 'float32');
      const b = Array1D.new([1.1, 2.1, 3.1], 'float32');
      const f = () => {
        math.greater(a, b);
      };
      expect(f).toThrowError();
    });
    it('NaNs in Array1D - int32', math => {
      const a = Array1D.new([1, NaN, 0], 'int32');
      const b = Array1D.new([0, 0, NaN], 'int32');
      const res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, boolNaN, boolNaN]);
    });
    it('NaNs in Array1D - float32', math => {
      const a = Array1D.new([1.1, NaN, 2.1], 'float32');
      const b = Array1D.new([2.1, 3.1, NaN], 'float32');
      const res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, boolNaN, boolNaN]);
    });

    // Array2D:
    it('Array2D - int32', math => {
      let a = Array2D.new([2, 3], [[1, 4, 5], [8, 9, 11]], 'int32');
      let b = Array2D.new([2, 3], [[2, 3, 6], [7, 10, 11]], 'int32');
      let res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0, 1, 0, 0]);

      a = Array2D.new([2, 2], [[0, 0], [1, 1]], 'int32');
      b = Array2D.new([2, 2], [[0, 0], [1, 1]], 'int32');
      res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0]);
    });
    it('Array2D - float32', math => {
      let a =
          Array2D.new([2, 3], [[1.1, 4.1, 5.1], [8.1, 9.1, 11.1]], 'float32');
      let b =
          Array2D.new([2, 3], [[2.1, 3.1, 6.1], [7.1, 10.1, 11.1]], 'float32');
      let res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0, 1, 0, 0]);

      a = Array2D.new([2, 2], [[0.2, 0.2], [1.2, 1.2]], 'float32');
      b = Array2D.new([2, 2], [[0.2, 0.2], [1.2, 1.2]], 'float32');
      res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0]);
    });
    it('broadcasting Array2D shapes - int32', math => {
      const a = Array2D.new([2, 1], [[3], [7]], 'int32');
      const b = Array2D.new([2, 3], [[2, 3, 4], [7, 8, 9]], 'int32');
      const res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 0, 0, 0, 0]);
    });
    it('broadcasting Array2D shapes - float32', math => {
      const a = Array2D.new([2, 1], [[1.1], [7.1]], 'float32');
      const b =
          Array2D.new([2, 3], [[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], 'float32');
      const res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 0, 0, 0, 0]);
    });
    it('NaNs in Array2D - int32', math => {
      const a = Array2D.new([2, 3], [[1, NaN, 2], [0, NaN, NaN]], 'int32');
      const b = Array2D.new([2, 3], [[0, NaN, NaN], [1, NaN, 3]], 'int32');
      const res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(
          res, [1, boolNaN, boolNaN, 0, boolNaN, boolNaN]);
    });
    it('NaNs in Array2D - float32', math => {
      const a = Array2D.new([2, 2], [[1.1, NaN], [0.1, NaN]], 'float32');
      const b = Array2D.new([2, 2], [[0.1, NaN], [1.1, NaN]], 'float32');
      const res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, boolNaN, 0, boolNaN]);
    });

    // Array3D:
    it('Array3D - int32', math => {
      let a =
          Array3D.new([2, 3, 1], [[[1], [4], [5]], [[8], [9], [11]]], 'int32');
      let b =
          Array3D.new([2, 3, 1], [[[2], [3], [6]], [[7], [10], [11]]], 'int32');
      let res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0, 1, 0, 0]);

      a = Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
      b = Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
      res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0, 0, 0]);
    });
    it('Array3D - float32', math => {
      let a = Array3D.new(
          [2, 3, 1], [[[1.1], [4.1], [5.1]], [[8.1], [9.1], [11.1]]],
          'float32');
      let b = Array3D.new(
          [2, 3, 1], [[[2.1], [3.1], [6.1]], [[7.1], [10.1], [11.1]]],
          'float32');
      let res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0, 1, 0, 0]);

      a = Array3D.new(
          [2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.2]]], 'float32');
      b = Array3D.new(
          [2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
      res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0, 0, 1]);
    });
    it('broadcasting Array3D shapes - int32', math => {
      const a = Array3D.new(
          [2, 3, 2], [[[1, 0], [2, 3], [4, 5]], [[6, 7], [9, 8], [10, 11]]],
          'int32');
      const b =
          Array3D.new([2, 3, 1], [[[1], [2], [3]], [[7], [10], [9]]], 'int32');
      const res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]);
    });
    it('broadcasting Array3D float32', math => {
      const a = Array3D.new(
          [2, 3, 2],
          [
            [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
            [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
          ],
          'float32');
      const b = Array3D.new(
          [2, 3, 1], [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]],
          'float32');
      const res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]);
    });
    it('NaNs in Array3D - int32', math => {
      const a =
          Array3D.new([2, 3, 1], [[[1], [NaN], [1]], [[0], [0], [0]]], 'int32');
      const b =
          Array3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [NaN]]], 'int32');
      const res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, boolNaN, 0, 0, 0, boolNaN]);
    });
    it('NaNs in Array3D - float32', math => {
      const a = Array3D.new(
          [2, 3, 1], [[[1.1], [NaN], [1.1]], [[0.1], [0.1], [0.1]]], 'float32');
      const b = Array3D.new(
          [2, 3, 1], [[[0.1], [0.1], [1.1]], [[1.1], [0.1], [NaN]]], 'float32');
      const res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, boolNaN, 0, 0, 0, boolNaN]);
    });

    // Array4D:
    it('Array4D - int32', math => {
      let a = Array4D.new([2, 2, 1, 1], [1, 4, 5, 8], 'int32');
      let b = Array4D.new([2, 2, 1, 1], [2, 3, 6, 8], 'int32');
      let res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0, 0]);

      a = Array4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
      b = Array4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
      res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0]);

      a = Array4D.new([2, 2, 1, 1], [2, 2, 2, 2], 'int32');
      b = Array4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'int32');
      res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1]);
    });
    it('Array4D - float32', math => {
      let a = Array4D.new([2, 2, 1, 1], [1.1, 4.1, 5.1, 8.1], 'float32');
      let b = Array4D.new([2, 2, 1, 1], [2.1, 3.1, 6.1, 8.1], 'float32');
      let res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0, 0]);

      a = Array4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
      b = Array4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
      res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0]);

      a = Array4D.new([2, 2, 1, 1], [1.1, 1.1, 1.1, 1.1], 'float32');
      b = Array4D.new([2, 2, 1, 1], [0.1, 0.1, 0.1, 0.1], 'float32');
      res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1]);
    });
    it('broadcasting Array4D shapes - int32', math => {
      const a = Array4D.new([2, 2, 1, 1], [1, 2, 5, 9], 'int32');
      const b = Array4D.new(
          [2, 2, 1, 2], [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], 'int32');
      const res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0, 0, 0, 1, 1]);
    });
    it('broadcasting Array4D shapes - float32', math => {
      const a = Array4D.new([2, 2, 1, 1], [1.1, 2.1, 5.1, 9.1], 'float32');
      const b = Array4D.new(
          [2, 2, 1, 2],
          [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]],
          'float32');
      const res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0, 0, 0, 1, 1]);
    });
    it('NaNs in Array4D - int32', math => {
      const a = Array4D.new([2, 2, 1, 1], [1, NaN, 0, 0], 'int32');
      const b = Array4D.new([2, 2, 1, 1], [0, 1, 1, NaN], 'int32');
      const res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, boolNaN, 0, boolNaN]);
    });
    it('NaNs in Array4D - float32', math => {
      const a = Array4D.new([2, 2, 1, 1], [1.1, NaN, 0.1, 0.1], 'float32');
      const b = Array4D.new([2, 2, 1, 1], [0.1, 1.1, 1.1, NaN], 'float32');
      const res = math.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, boolNaN, 0, boolNaN]);
    });
  };

  test_util.describeMathCPU('greater', [tests]);
  test_util.describeMathGPU('greater', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// GreaterEqual:
{
  const boolNaN = util.getNaN('bool');

  const tests: MathTests = it => {
    // Array1D:
    it('Array1D - int32', math => {
      let a = Array1D.new([1, 4, 5], 'int32');
      let b = Array1D.new([2, 3, 5], 'int32');
      let res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 1]);

      a = Array1D.new([2, 2, 2], 'int32');
      b = Array1D.new([2, 2, 2], 'int32');
      res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1]);

      a = Array1D.new([0, 0], 'int32');
      b = Array1D.new([3, 3], 'int32');
      res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0]);
    });
    it('Array1D - float32', math => {
      let a = Array1D.new([1.1, 4.1, 5.1], 'float32');
      let b = Array1D.new([2.2, 3.2, 5.1], 'float32');
      let res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 1]);

      a = Array1D.new([2.31, 2.31, 2.31], 'float32');
      b = Array1D.new([2.31, 2.31, 2.31], 'float32');
      res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1]);

      a = Array1D.new([0.45, 0.123], 'float32');
      b = Array1D.new([3.123, 3.321], 'float32');
      res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0]);
    });
    it('mismatched Array1D shapes - int32', math => {
      const a = Array1D.new([1, 2], 'int32');
      const b = Array1D.new([1, 2, 3], 'int32');
      const f = () => {
        math.greaterEqual(a, b);
      };
      expect(f).toThrowError();
    });
    it('mismatched Array1D shapes - float32', math => {
      const a = Array1D.new([1.1, 2.1], 'float32');
      const b = Array1D.new([1.1, 2.1, 3.1], 'float32');
      const f = () => {
        math.greaterEqual(a, b);
      };
      expect(f).toThrowError();
    });
    it('NaNs in Array1D - int32', math => {
      const a = Array1D.new([1, NaN, 0], 'int32');
      const b = Array1D.new([0, 0, NaN], 'int32');
      const res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, boolNaN, boolNaN]);
    });
    it('NaNs in Array1D - float32', math => {
      const a = Array1D.new([1.1, NaN, 2.1], 'float32');
      const b = Array1D.new([2.1, 3.1, NaN], 'float32');
      const res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, boolNaN, boolNaN]);
    });

    // Array2D:
    it('Array2D - int32', math => {
      let a = Array2D.new([2, 3], [[1, 4, 5], [8, 9, 12]], 'int32');
      let b = Array2D.new([2, 3], [[2, 3, 6], [7, 10, 11]], 'int32');
      let res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0, 1, 0, 1]);

      a = Array2D.new([2, 2], [[0, 0], [1, 1]], 'int32');
      b = Array2D.new([2, 2], [[0, 0], [1, 1]], 'int32');
      res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1]);
    });
    it('Array2D - float32', math => {
      let a =
          Array2D.new([2, 3], [[1.1, 4.1, 5.1], [8.1, 9.1, 12.1]], 'float32');
      let b =
          Array2D.new([2, 3], [[2.1, 3.1, 6.1], [7.1, 10.1, 11.1]], 'float32');
      let res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0, 1, 0, 1]);

      a = Array2D.new([2, 2], [[0.2, 0.2], [1.2, 1.2]], 'float32');
      b = Array2D.new([2, 2], [[0.2, 0.2], [1.2, 1.2]], 'float32');
      res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1]);
    });
    it('broadcasting Array2D shapes - int32', math => {
      const a = Array2D.new([2, 1], [[3], [7]], 'int32');
      const b = Array2D.new([2, 3], [[2, 3, 4], [7, 8, 9]], 'int32');
      const res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 0, 1, 0, 0]);
    });
    it('broadcasting Array2D shapes - float32', math => {
      const a = Array2D.new([2, 1], [[1.1], [7.1]], 'float32');
      const b =
          Array2D.new([2, 3], [[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], 'float32');
      const res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 0, 1, 0, 0]);
    });
    it('NaNs in Array2D - int32', math => {
      const a = Array2D.new([2, 3], [[1, NaN, 2], [0, NaN, NaN]], 'int32');
      const b = Array2D.new([2, 3], [[0, NaN, NaN], [1, NaN, 3]], 'int32');
      const res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(
          res, [1, boolNaN, boolNaN, 0, boolNaN, boolNaN]);
    });
    it('NaNs in Array2D - float32', math => {
      const a = Array2D.new([2, 2], [[1.1, NaN], [0.1, NaN]], 'float32');
      const b = Array2D.new([2, 2], [[0.1, NaN], [1.1, NaN]], 'float32');
      const res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, boolNaN, 0, boolNaN]);
    });

    // Array3D:
    it('Array3D - int32', math => {
      let a =
          Array3D.new([2, 3, 1], [[[1], [4], [5]], [[8], [9], [12]]], 'int32');
      let b =
          Array3D.new([2, 3, 1], [[[2], [3], [6]], [[7], [10], [11]]], 'int32');
      let res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0, 1, 0, 1]);

      a = Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
      b = Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
      res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1, 1, 1]);
    });
    it('Array3D - float32', math => {
      let a = Array3D.new(
          [2, 3, 1], [[[1.1], [4.1], [5.1]], [[8.1], [9.1], [12.1]]],
          'float32');
      let b = Array3D.new(
          [2, 3, 1], [[[2.1], [3.1], [6.1]], [[7.1], [10.1], [11.1]]],
          'float32');
      let res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0, 1, 0, 1]);

      a = Array3D.new(
          [2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.2]]], 'float32');
      b = Array3D.new(
          [2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
      res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1, 1, 1]);
    });
    it('broadcasting Array3D shapes - int32', math => {
      const a = Array3D.new(
          [2, 3, 2], [[[1, 0], [2, 3], [4, 5]], [[6, 7], [9, 8], [10, 11]]],
          'int32');
      const b =
          Array3D.new([2, 3, 1], [[[1], [2], [3]], [[7], [10], [9]]], 'int32');
      const res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1]);
    });
    it('broadcasting Array3D float32', math => {
      const a = Array3D.new(
          [2, 3, 2],
          [
            [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
            [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
          ],
          'float32');
      const b = Array3D.new(
          [2, 3, 1], [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]],
          'float32');
      const res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1]);
    });
    it('NaNs in Array3D - int32', math => {
      const a =
          Array3D.new([2, 3, 1], [[[1], [NaN], [1]], [[0], [0], [0]]], 'int32');
      const b =
          Array3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [NaN]]], 'int32');
      const res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, boolNaN, 1, 0, 1, boolNaN]);
    });
    it('NaNs in Array3D - float32', math => {
      const a = Array3D.new(
          [2, 3, 1], [[[1.1], [NaN], [1.1]], [[0.1], [0.1], [0.1]]], 'float32');
      const b = Array3D.new(
          [2, 3, 1], [[[0.1], [0.1], [1.1]], [[1.1], [0.1], [NaN]]], 'float32');
      const res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, boolNaN, 1, 0, 1, boolNaN]);
    });

    // Array4D:
    it('Array4D - int32', math => {
      let a = Array4D.new([2, 2, 1, 1], [1, 4, 5, 8], 'int32');
      let b = Array4D.new([2, 2, 1, 1], [2, 3, 6, 7], 'int32');
      let res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0, 1]);

      a = Array4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
      b = Array4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
      res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1]);

      a = Array4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'int32');
      b = Array4D.new([2, 2, 1, 1], [2, 2, 2, 2], 'int32');
      res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0]);
    });
    it('Array4D - float32', math => {
      let a = Array4D.new([2, 2, 1, 1], [1.1, 4.1, 5.1, 8.1], 'float32');
      let b = Array4D.new([2, 2, 1, 1], [2.1, 3.1, 6.1, 7.1], 'float32');
      let res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0, 1]);

      a = Array4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
      b = Array4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
      res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1]);

      a = Array4D.new([2, 2, 1, 1], [0.1, 0.1, 0.1, 0.1], 'float32');
      b = Array4D.new([2, 2, 1, 1], [1.1, 1.1, 1.1, 1.1], 'float32');
      res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0]);
    });
    it('broadcasting Array4D shapes - int32', math => {
      const a = Array4D.new([2, 2, 1, 1], [1, 2, 5, 9], 'int32');
      const b = Array4D.new(
          [2, 2, 1, 2], [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], 'int32');
      const res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 0, 0, 1, 0, 1, 1]);
    });
    it('broadcasting Array4D shapes - float32', math => {
      const a = Array4D.new([2, 2, 1, 1], [1.1, 2.1, 5.1, 9.1], 'float32');
      const b = Array4D.new(
          [2, 2, 1, 2],
          [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]],
          'float32');
      const res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 0, 0, 1, 0, 1, 1]);
    });
    it('NaNs in Array4D - int32', math => {
      const a = Array4D.new([2, 2, 1, 1], [1, NaN, 0, 0], 'int32');
      const b = Array4D.new([2, 2, 1, 1], [0, 1, 1, NaN], 'int32');
      const res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, boolNaN, 0, boolNaN]);
    });
    it('NaNs in Array4D - float32', math => {
      const a = Array4D.new([2, 2, 1, 1], [1.1, NaN, 0.1, 0.1], 'float32');
      const b = Array4D.new([2, 2, 1, 1], [0.1, 1.1, 1.1, NaN], 'float32');
      const res = math.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, boolNaN, 0, boolNaN]);
    });
  };

  test_util.describeMathCPU('greaterEqual', [tests]);
  test_util.describeMathGPU('greaterEqual', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
