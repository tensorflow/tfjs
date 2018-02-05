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
import {Tensor2D, Tensor3D, Tensor4D} from './tensor';

// Equal:
{
  const boolNaN = util.getNaN('bool');

  const tests: MathTests = it => {
    // Tensor1D:
    it('Tensor1D - int32', () => {
      let a = dl.tensor1d([1, 4, 5], 'int32');
      let b = dl.tensor1d([2, 3, 5], 'int32');

      test_util.expectArraysClose(dl.equal(a, b), [0, 0, 1]);

      a = dl.tensor1d([2, 2, 2], 'int32');
      b = dl.tensor1d([2, 2, 2], 'int32');
      test_util.expectArraysClose(dl.equal(a, b), [1, 1, 1]);

      a = dl.tensor1d([0, 0], 'int32');
      b = dl.tensor1d([3, 3], 'int32');
      test_util.expectArraysClose(dl.equal(a, b), [0, 0]);
    });
    it('Tensor1D - float32', () => {
      let a = dl.tensor1d([1.1, 4.1, 5.1], 'float32');
      let b = dl.tensor1d([2.2, 3.2, 5.1], 'float32');

      test_util.expectArraysClose(dl.equal(a, b), [0, 0, 1]);

      a = dl.tensor1d([2.31, 2.31, 2.31], 'float32');
      b = dl.tensor1d([2.31, 2.31, 2.31], 'float32');
      test_util.expectArraysClose(dl.equal(a, b), [1, 1, 1]);

      a = dl.tensor1d([0.45, 0.123], 'float32');
      b = dl.tensor1d([3.123, 3.321], 'float32');
      test_util.expectArraysClose(dl.equal(a, b), [0, 0]);
    });
    it('mismatched Tensor1D shapes - int32', () => {
      const a = dl.tensor1d([1, 2], 'int32');
      const b = dl.tensor1d([1, 2, 3], 'int32');
      const f = () => {
        dl.equal(a, b);
      };
      expect(f).toThrowError();
    });
    it('mismatched Tensor1D shapes - float32', () => {
      const a = dl.tensor1d([1.1, 2.1], 'float32');
      const b = dl.tensor1d([1.1, 2.1, 3.1], 'float32');
      const f = () => {
        dl.equal(a, b);
      };
      expect(f).toThrowError();
    });
    it('NaNs in Tensor1D - int32', () => {
      const a = dl.tensor1d([1, NaN, 0], 'int32');
      const b = dl.tensor1d([0, 0, NaN], 'int32');

      test_util.expectArraysClose(dl.equal(a, b), [0, boolNaN, boolNaN]);
    });
    it('NaNs in Tensor1D - float32', () => {
      const a = dl.tensor1d([1.1, NaN, 2.1], 'float32');
      const b = dl.tensor1d([2.1, 3.1, NaN], 'float32');
      test_util.expectArraysClose(dl.equal(a, b), [0, boolNaN, boolNaN]);
    });
    it('scalar and 1D broadcast', () => {
      const a = dl.scalar(2);
      const b = dl.tensor1d([1, 2, 3, 4, 5, 2]);
      const res = dl.equal(a, b);
      expect(res.dtype).toBe('bool');
      expect(res.shape).toEqual([6]);
      test_util.expectArraysEqual(res, [0, 1, 0, 0, 0, 1]);
    });

    // Tensor2D:
    it('Tensor2D - int32', () => {
      let a = dl.tensor2d([[1, 4, 5], [8, 9, 12]], [2, 3], 'int32');
      let b = dl.tensor2d([[2, 3, 6], [7, 10, 11]], [2, 3], 'int32');
      test_util.expectArraysClose(dl.equal(a, b), [0, 0, 0, 0, 0, 0]);

      a = dl.tensor2d([[0, 0], [1, 1]], [2, 2], 'int32');
      b = dl.tensor2d([[0, 0], [1, 1]], [2, 2], 'int32');
      test_util.expectArraysClose(dl.equal(a, b), [1, 1, 1, 1]);
    });
    it('Tensor2D - float32', () => {
      let a =
          dl.tensor2d([[1.1, 4.1, 5.1], [8.1, 9.1, 12.1]], [2, 3], 'float32');
      let b =
          dl.tensor2d([[2.1, 4.1, 5.1], [7.1, 10.1, 11.1]], [2, 3], 'float32');
      test_util.expectArraysClose(dl.equal(a, b), [0, 1, 1, 0, 0, 0]);

      a = dl.tensor2d([[0.2, 0.2], [1.2, 1.2]], [2, 2], 'float32');
      b = dl.tensor2d([[0.2, 0.2], [1.2, 1.2]], [2, 2], 'float32');
      test_util.expectArraysClose(dl.equal(a, b), [1, 1, 1, 1]);
    });
    it('broadcasting Tensor2D shapes - int32', () => {
      const a = dl.tensor2d([[3], [7]], [2, 1], 'int32');
      const b = dl.tensor2d([[2, 3, 4], [7, 8, 9]], [2, 3], 'int32');
      test_util.expectArraysClose(dl.equal(a, b), [0, 1, 0, 1, 0, 0]);
    });
    it('broadcasting Tensor2D shapes - float32', () => {
      const a = dl.tensor2d([[1.1], [7.1]], [2, 1], 'float32');
      const b =
          dl.tensor2d([[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], [2, 3], 'float32');
      test_util.expectArraysClose(dl.equal(a, b), [0, 1, 0, 1, 0, 0]);
    });
    it('NaNs in Tensor2D - int32', () => {
      const a = dl.tensor2d([[1, NaN, 2], [1, NaN, NaN]], [2, 3], 'int32');
      const b = dl.tensor2d([[0, NaN, NaN], [1, NaN, 3]], [2, 3], 'int32');
      test_util.expectArraysClose(
          dl.equal(a, b), [0, boolNaN, boolNaN, 1, boolNaN, boolNaN]);
    });
    it('NaNs in Tensor2D - float32', () => {
      const a = dl.tensor2d([[1.1, NaN], [1.1, NaN]], [2, 2], 'float32');
      const b = dl.tensor2d([[0.1, NaN], [1.1, NaN]], [2, 2], 'float32');
      test_util.expectArraysClose(dl.equal(a, b), [0, boolNaN, 1, boolNaN]);
    });
    it('2D and 2D broadcast each with 1 dim', () => {
      const a = dl.tensor2d([1, 2, 5], [1, 3]);
      const b = dl.tensor2d([5, 1], [2, 1]);
      const res = dl.equal(a, b);
      expect(res.dtype).toBe('bool');
      expect(res.shape).toEqual([2, 3]);
      test_util.expectArraysEqual(res, [0, 0, 1, 1, 0, 0]);
    });
    it('2D and scalar broadcast', () => {
      const a = dl.tensor2d([1, 2, 3, 2, 5, 6], [2, 3]);
      const b = dl.scalar(2);
      const res = dl.equal(a, b);
      expect(res.dtype).toBe('bool');
      expect(res.shape).toEqual([2, 3]);
      test_util.expectArraysEqual(res, [0, 1, 0, 1, 0, 0]);
    });

    // Tensor3D:
    it('Tensor3D - int32', () => {
      let a =
          Tensor3D.new([2, 3, 1], [[[1], [4], [5]], [[8], [9], [12]]], 'int32');
      let b =
          Tensor3D.new([2, 3, 1],
            [[[2], [3], [6]], [[7], [10], [12]]], 'int32');
      test_util.expectArraysClose(dl.equal(a, b), [0, 0, 0, 0, 0, 1]);

      a = Tensor3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
      b = Tensor3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
      test_util.expectArraysClose(dl.equal(a, b), [1, 1, 1, 1, 1, 1]);
    });
    it('Tensor3D - float32', () => {
      let a = Tensor3D.new(
          [2, 3, 1], [[[1.1], [4.1], [5.1]], [[8.1], [9.1], [12.1]]],
          'float32');
      let b = Tensor3D.new(
          [2, 3, 1], [[[2.1], [3.1], [6.1]], [[7.1], [10.1], [12.1]]],
          'float32');
      test_util.expectArraysClose(dl.equal(a, b), [0, 0, 0, 0, 0, 1]);

      a = Tensor3D.new(
          [2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
      b = Tensor3D.new(
          [2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
      test_util.expectArraysClose(dl.equal(a, b), [1, 1, 1, 1, 1, 1]);
    });
    it('broadcasting Tensor3D shapes - int32', () => {
      const a = Tensor3D.new(
          [2, 3, 2], [[[1, 0], [2, 3], [4, 5]], [[6, 7], [9, 8], [10, 11]]],
          'int32');
      const b =
          Tensor3D.new([2, 3, 1], [[[1], [2], [3]], [[7], [10], [9]]], 'int32');
      test_util.expectArraysClose(
          dl.equal(a, b), [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]);
    });
    it('broadcasting Tensor3D shapes - float32', () => {
      const a = Tensor3D.new(
          [2, 3, 2],
          [
            [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
            [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
          ],
          'float32');
      const b = Tensor3D.new(
          [2, 3, 1], [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]],
          'float32');
      test_util.expectArraysClose(
          dl.equal(a, b), [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]);
    });
    it('NaNs in Tensor3D - int32', () => {
      const a =
          Tensor3D.new([2, 3, 1],
            [[[1], [NaN], [1]], [[0], [0], [0]]], 'int32');
      const b =
          Tensor3D.new([2, 3, 1],
            [[[0], [0], [1]], [[1], [0], [NaN]]], 'int32');
      test_util.expectArraysClose(
          dl.equal(a, b), [0, boolNaN, 1, 0, 1, boolNaN]);
    });
    it('NaNs in Tensor3D - float32', () => {
      const a = Tensor3D.new(
          [2, 3, 1], [[[1.1], [NaN], [1.1]], [[0.1], [0.1], [0.1]]], 'float32');
      const b = Tensor3D.new(
          [2, 3, 1], [[[0.1], [0.1], [1.1]], [[1.1], [0.1], [NaN]]], 'float32');
      test_util.expectArraysClose(
          dl.equal(a, b), [0, boolNaN, 1, 0, 1, boolNaN]);
    });
    it('3D and scalar', () => {
      const a = Tensor3D.new([2, 3, 1], [1, 2, 3, 4, 5, -1]);
      const b = dl.scalar(-1);
      const res = dl.equal(a, b);
      expect(res.dtype).toBe('bool');
      expect(res.shape).toEqual([2, 3, 1]);
      test_util.expectArraysEqual(res, [0, 0, 0, 0, 0, 1]);
    });

    // Tensor4D:
    it('Tensor4D - int32', () => {
      let a = Tensor4D.new([2, 2, 1, 1], [1, 4, 5, 8], 'int32');
      let b = Tensor4D.new([2, 2, 1, 1], [2, 3, 6, 8], 'int32');
      test_util.expectArraysClose(dl.equal(a, b), [0, 0, 0, 1]);

      a = Tensor4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
      b = Tensor4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
      test_util.expectArraysClose(dl.equal(a, b), [1, 1, 1, 1]);

      a = Tensor4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'int32');
      b = Tensor4D.new([2, 2, 1, 1], [2, 2, 2, 2], 'int32');
      test_util.expectArraysClose(dl.equal(a, b), [0, 0, 0, 0]);
    });
    it('Tensor4D - float32', () => {
      let a = Tensor4D.new([2, 2, 1, 1], [1.1, 4.1, 5.1, 8.1], 'float32');
      let b = Tensor4D.new([2, 2, 1, 1], [2.1, 3.1, 6.1, 8.1], 'float32');
      test_util.expectArraysClose(dl.equal(a, b), [0, 0, 0, 1]);

      a = Tensor4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
      b = Tensor4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
      test_util.expectArraysClose(dl.equal(a, b), [1, 1, 1, 1]);

      a = Tensor4D.new([2, 2, 1, 1], [0.1, 0.1, 0.1, 0.1], 'float32');
      b = Tensor4D.new([2, 2, 1, 1], [1.1, 1.1, 1.1, 1.1], 'float32');
      test_util.expectArraysClose(dl.equal(a, b), [0, 0, 0, 0]);
    });
    it('broadcasting Tensor4D shapes - int32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1, 2, 5, 9], 'int32');
      const b = Tensor4D.new(
          [2, 2, 1, 2], [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], 'int32');
      test_util.expectArraysClose(dl.equal(a, b), [1, 0, 0, 0, 1, 0, 0, 0]);
    });
    it('broadcasting Tensor4D shapes - float32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1.1, 2.1, 5.1, 9.1], 'float32');
      const b = Tensor4D.new(
          [2, 2, 1, 2],
          [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]],
          'float32');
      test_util.expectArraysClose(dl.equal(a, b), [1, 0, 0, 0, 1, 0, 0, 0]);
    });
    it('NaNs in Tensor4D - int32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1, NaN, 1, 0], 'int32');
      const b = Tensor4D.new([2, 2, 1, 1], [0, 1, 1, NaN], 'int32');
      test_util.expectArraysClose(dl.equal(a, b), [0, boolNaN, 1, boolNaN]);
    });
    it('NaNs in Tensor4D - float32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1.1, NaN, 1.1, 0.1], 'float32');
      const b = Tensor4D.new([2, 2, 1, 1], [0.1, 1.1, 1.1, NaN], 'float32');
      test_util.expectArraysClose(dl.equal(a, b), [0, boolNaN, 1, boolNaN]);
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
    // Tensor1D:
    it('Tensor1D - int32', () => {
      let a = dl.tensor1d([1, 4, 5], 'int32');
      let b = dl.tensor1d([2, 3, 5], 'int32');
      test_util.expectArraysClose(dl.equalStrict(a, b), [0, 0, 1]);

      a = dl.tensor1d([2, 2, 2], 'int32');
      b = dl.tensor1d([2, 2, 2], 'int32');
      test_util.expectArraysClose(dl.equalStrict(a, b), [1, 1, 1]);

      a = dl.tensor1d([0, 0], 'int32');
      b = dl.tensor1d([3, 3], 'int32');
      test_util.expectArraysClose(dl.equalStrict(a, b), [0, 0]);
    });
    it('Tensor1D - float32', () => {
      let a = dl.tensor1d([1.1, 4.1, 5.1], 'float32');
      let b = dl.tensor1d([2.2, 3.2, 5.1], 'float32');
      test_util.expectArraysClose(dl.equalStrict(a, b), [0, 0, 1]);

      a = dl.tensor1d([2.31, 2.31, 2.31], 'float32');
      b = dl.tensor1d([2.31, 2.31, 2.31], 'float32');
      test_util.expectArraysClose(dl.equalStrict(a, b), [1, 1, 1]);

      a = dl.tensor1d([0.45, 0.123], 'float32');
      b = dl.tensor1d([3.123, 3.321], 'float32');
      test_util.expectArraysClose(dl.equalStrict(a, b), [0, 0]);
    });
    it('mismatched Tensor1D shapes - int32', () => {
      const a = dl.tensor1d([1, 2], 'int32');
      const b = dl.tensor1d([1, 2, 3], 'int32');
      const f = () => {
        dl.equalStrict(a, b);
      };
      expect(f).toThrowError();
    });
    it('mismatched Tensor1D shapes - float32', () => {
      const a = dl.tensor1d([1.1, 2.1], 'float32');
      const b = dl.tensor1d([1.1, 2.1, 3.1], 'float32');
      const f = () => {
        dl.equalStrict(a, b);
      };
      expect(f).toThrowError();
    });
    it('NaNs in Tensor1D - int32', () => {
      const a = dl.tensor1d([1, NaN, 0], 'int32');
      const b = dl.tensor1d([0, 0, NaN], 'int32');
      test_util.expectArraysClose(
          dl.equalStrict(a, b), [0, boolNaN, boolNaN]);
    });
    it('NaNs in Tensor1D - float32', () => {
      const a = dl.tensor1d([1.1, NaN, 2.1], 'float32');
      const b = dl.tensor1d([2.1, 3.1, NaN], 'float32');
      test_util.expectArraysClose(
          dl.equalStrict(a, b), [0, boolNaN, boolNaN]);
    });

    // Tensor2D:
    it('Tensor2D - int32', () => {
      let a = dl.tensor2d([[1, 4, 5], [8, 9, 12]], [2, 3], 'int32');
      let b = dl.tensor2d([[2, 3, 6], [7, 10, 11]], [2, 3], 'int32');
      test_util.expectArraysClose(dl.equalStrict(a, b), [0, 0, 0, 0, 0, 0]);

      a = dl.tensor2d([[0, 0], [1, 1]], [2, 2], 'int32');
      b = dl.tensor2d([[0, 0], [1, 1]], [2, 2], 'int32');
      test_util.expectArraysClose(dl.equalStrict(a, b), [1, 1, 1, 1]);
    });
    it('Tensor2D - float32', () => {
      let a =
          dl.tensor2d([[1.1, 4.1, 5.1], [8.1, 9.1, 12.1]], [2, 3], 'float32');
      let b =
          dl.tensor2d([[2.1, 4.1, 5.1], [7.1, 10.1, 11.1]], [2, 3], 'float32');
      test_util.expectArraysClose(dl.equalStrict(a, b), [0, 1, 1, 0, 0, 0]);

      a = dl.tensor2d([[0.2, 0.2], [1.2, 1.2]], [2, 2], 'float32');
      b = dl.tensor2d([[0.2, 0.2], [1.2, 1.2]], [2, 2], 'float32');
      test_util.expectArraysClose(dl.equalStrict(a, b), [1, 1, 1, 1]);
    });
    it('mismatch Tensor2D shapes - int32', () => {
      const a = dl.tensor2d([[3], [7]], [2, 1], 'int32');
      const b = dl.tensor2d([[2, 3, 4], [7, 8, 9]], [2, 3], 'int32');

      const f = () => {
        dl.equalStrict(a, b);
      };
      expect(f).toThrowError();
    });
    it('mismatch Tensor2D shapes - float32', () => {
      const a = dl.tensor2d([[1.1], [7.1]], [2, 1], 'float32');
      const b =
          dl.tensor2d([[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], [2, 3], 'float32');
      const f = () => {
        dl.equalStrict(a, b);
      };
      expect(f).toThrowError();
    });
    it('NaNs in Tensor2D - int32', () => {
      const a = dl.tensor2d([[1, NaN, 2], [1, NaN, NaN]], [2, 3], 'int32');
      const b = dl.tensor2d([[0, NaN, NaN], [1, NaN, 3]], [2, 3], 'int32');
      test_util.expectArraysClose(
          dl.equalStrict(a, b), [0, boolNaN, boolNaN, 1, boolNaN, boolNaN]);
    });
    it('NaNs in Tensor2D - float32', () => {
      const a = dl.tensor2d([[1.1, NaN], [1.1, NaN]], [2, 2], 'float32');
      const b = dl.tensor2d([[0.1, NaN], [1.1, NaN]], [2, 2], 'float32');
      test_util.expectArraysClose(
          dl.equalStrict(a, b), [0, boolNaN, 1, boolNaN]);
    });

    // Tensor3D:
    it('Tensor3D - int32', () => {
      let a =
          Tensor3D.new([2, 3, 1], [[[1], [4], [5]], [[8], [9], [12]]], 'int32');
      let b =
          Tensor3D.new([2, 3, 1],
            [[[2], [3], [6]], [[7], [10], [12]]], 'int32');
      test_util.expectArraysClose(dl.equalStrict(a, b), [0, 0, 0, 0, 0, 1]);

      a = Tensor3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
      b = Tensor3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
      test_util.expectArraysClose(dl.equalStrict(a, b), [1, 1, 1, 1, 1, 1]);
    });
    it('Tensor3D - float32', () => {
      let a = Tensor3D.new(
          [2, 3, 1], [[[1.1], [4.1], [5.1]], [[8.1], [9.1], [12.1]]],
          'float32');
      let b = Tensor3D.new(
          [2, 3, 1], [[[2.1], [3.1], [6.1]], [[7.1], [10.1], [12.1]]],
          'float32');
      test_util.expectArraysClose(dl.equalStrict(a, b), [0, 0, 0, 0, 0, 1]);

      a = Tensor3D.new(
          [2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
      b = Tensor3D.new(
          [2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
      test_util.expectArraysClose(dl.equalStrict(a, b), [1, 1, 1, 1, 1, 1]);
    });
    it('mismatch Tensor3D shapes - int32', () => {
      const a = Tensor3D.new(
          [2, 3, 2], [[[1, 0], [2, 3], [4, 5]], [[6, 7], [9, 8], [10, 11]]],
          'int32');
      const b =
          Tensor3D.new([2, 3, 1], [[[1], [2], [3]], [[7], [10], [9]]], 'int32');

      const f = () => {
        dl.equalStrict(a, b);
      };
      expect(f).toThrowError();
    });
    it('mismatch Tensor3D shapes - float32', () => {
      const a = Tensor3D.new(
          [2, 3, 2],
          [
            [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
            [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
          ],
          'float32');
      const b = Tensor3D.new(
          [2, 3, 1], [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]],
          'float32');

      const f = () => {
        dl.equalStrict(a, b);
      };
      expect(f).toThrowError();
    });
    it('NaNs in Tensor3D - int32', () => {
      const a =
          Tensor3D.new([2, 3, 1],
            [[[1], [NaN], [1]], [[0], [0], [0]]], 'int32');
      const b =
          Tensor3D.new([2, 3, 1],
            [[[0], [0], [1]], [[1], [0], [NaN]]], 'int32');
      test_util.expectArraysClose(
          dl.equalStrict(a, b), [0, boolNaN, 1, 0, 1, boolNaN]);
    });
    it('NaNs in Tensor3D - float32', () => {
      const a = Tensor3D.new(
          [2, 3, 1], [[[1.1], [NaN], [1.1]], [[0.1], [0.1], [0.1]]], 'float32');
      const b = Tensor3D.new(
          [2, 3, 1], [[[0.1], [0.1], [1.1]], [[1.1], [0.1], [NaN]]], 'float32');
      test_util.expectArraysClose(
          dl.equalStrict(a, b), [0, boolNaN, 1, 0, 1, boolNaN]);
    });

    // Tensor4D:
    it('Tensor4D - int32', () => {
      let a = Tensor4D.new([2, 2, 1, 1], [1, 4, 5, 8], 'int32');
      let b = Tensor4D.new([2, 2, 1, 1], [2, 3, 6, 8], 'int32');
      test_util.expectArraysClose(dl.equalStrict(a, b), [0, 0, 0, 1]);

      a = Tensor4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
      b = Tensor4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
      test_util.expectArraysClose(dl.equalStrict(a, b), [1, 1, 1, 1]);

      a = Tensor4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'int32');
      b = Tensor4D.new([2, 2, 1, 1], [2, 2, 2, 2], 'int32');
      test_util.expectArraysClose(dl.equalStrict(a, b), [0, 0, 0, 0]);
    });
    it('Tensor4D - float32', () => {
      let a = Tensor4D.new([2, 2, 1, 1], [1.1, 4.1, 5.1, 8.1], 'float32');
      let b = Tensor4D.new([2, 2, 1, 1], [2.1, 3.1, 6.1, 8.1], 'float32');
      test_util.expectArraysClose(dl.equalStrict(a, b), [0, 0, 0, 1]);

      a = Tensor4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
      b = Tensor4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
      test_util.expectArraysClose(dl.equalStrict(a, b), [1, 1, 1, 1]);

      a = Tensor4D.new([2, 2, 1, 1], [0.1, 0.1, 0.1, 0.1], 'float32');
      b = Tensor4D.new([2, 2, 1, 1], [1.1, 1.1, 1.1, 1.1], 'float32');
      test_util.expectArraysClose(dl.equalStrict(a, b), [0, 0, 0, 0]);
    });
    it('mismatch Tensor4D shapes - int32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1, 2, 5, 9], 'int32');
      const b = Tensor4D.new(
          [2, 2, 1, 2], [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], 'int32');

      const f = () => {
        dl.equalStrict(a, b);
      };
      expect(f).toThrowError();
    });
    it('mismatch Tensor4D shapes - float32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1.1, 2.1, 5.1, 9.1], 'float32');
      const b = Tensor4D.new(
          [2, 2, 1, 2],
          [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]],
          'float32');

      const f = () => {
        dl.equalStrict(a, b);
      };
      expect(f).toThrowError();
    });
    it('NaNs in Tensor4D - int32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1, NaN, 1, 0], 'int32');
      const b = Tensor4D.new([2, 2, 1, 1], [0, 1, 1, NaN], 'int32');
      test_util.expectArraysClose(
          dl.equalStrict(a, b), [0, boolNaN, 1, boolNaN]);
    });
    it('NaNs in Tensor4D - float32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1.1, NaN, 1.1, 0.1], 'float32');
      const b = Tensor4D.new([2, 2, 1, 1], [0.1, 1.1, 1.1, NaN], 'float32');
      test_util.expectArraysClose(
          dl.equalStrict(a, b), [0, boolNaN, 1, boolNaN]);
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
  const boolNaN = util.getNaN('bool');

  const tests: MathTests = it => {
    // Tensor1D:
    it('Tensor1D - int32', () => {
      let a = dl.tensor1d([1, 4, 5], 'int32');
      let b = dl.tensor1d([2, 3, 5], 'int32');

      test_util.expectArraysClose(dl.notEqual(a, b), [1, 1, 0]);

      a = dl.tensor1d([2, 2, 2], 'int32');
      b = dl.tensor1d([2, 2, 2], 'int32');
      test_util.expectArraysClose(dl.notEqual(a, b), [0, 0, 0]);

      a = dl.tensor1d([0, 0], 'int32');
      b = dl.tensor1d([3, 3], 'int32');
      test_util.expectArraysClose(dl.notEqual(a, b), [1, 1]);
    });
    it('Tensor1D - float32', () => {
      let a = dl.tensor1d([1.1, 4.1, 5.1], 'float32');
      let b = dl.tensor1d([2.2, 3.2, 5.1], 'float32');

      test_util.expectArraysClose(dl.notEqual(a, b), [1, 1, 0]);

      a = dl.tensor1d([2.31, 2.31, 2.31], 'float32');
      b = dl.tensor1d([2.31, 2.31, 2.31], 'float32');
      test_util.expectArraysClose(dl.notEqual(a, b), [0, 0, 0]);

      a = dl.tensor1d([0.45, 0.123], 'float32');
      b = dl.tensor1d([3.123, 3.321], 'float32');
      test_util.expectArraysClose(dl.notEqual(a, b), [1, 1]);
    });
    it('mismatched Tensor1D shapes - int32', () => {
      const a = dl.tensor1d([1, 2], 'int32');
      const b = dl.tensor1d([1, 2, 3], 'int32');
      const f = () => {
        dl.notEqual(a, b);
      };
      expect(f).toThrowError();
    });
    it('mismatched Tensor1D shapes - float32', () => {
      const a = dl.tensor1d([1.1, 2.1], 'float32');
      const b = dl.tensor1d([1.1, 2.1, 3.1], 'float32');
      const f = () => {
        dl.notEqual(a, b);
      };
      expect(f).toThrowError();
    });
    it('NaNs in Tensor1D - int32', () => {
      const a = dl.tensor1d([1, NaN, 0], 'int32');
      const b = dl.tensor1d([0, 0, NaN], 'int32');

      test_util.expectArraysClose(dl.notEqual(a, b), [1, boolNaN, boolNaN]);
    });
    it('NaNs in Tensor1D - float32', () => {
      const a = dl.tensor1d([1.1, NaN, 2.1], 'float32');
      const b = dl.tensor1d([2.1, 3.1, NaN], 'float32');
      test_util.expectArraysClose(dl.notEqual(a, b), [1, boolNaN, boolNaN]);
    });
    it('propagates NaNs', () => {
      const a = dl.tensor1d([2, 5, NaN]);
      const b = dl.tensor1d([4, 5, -1]);

      const res = dl.notEqual(a, b);
      expect(res.dtype).toBe('bool');
      test_util.expectArraysEqual(res, [1, 0, util.NAN_BOOL]);
    });
    it('scalar and 1D broadcast', () => {
      const a = dl.scalar(2);
      const b = dl.tensor1d([1, 2, 3, 4, 5, 2]);
      const res = dl.notEqual(a, b);
      expect(res.dtype).toBe('bool');
      expect(res.shape).toEqual([6]);
      test_util.expectArraysEqual(res, [1, 0, 1, 1, 1, 0]);
    });

    // Tensor2D:
    it('Tensor2D - int32', () => {
      let a = dl.tensor2d([[1, 4, 5], [8, 9, 12]], [2, 3], 'int32');
      let b = dl.tensor2d([[2, 3, 6], [7, 10, 11]], [2, 3], 'int32');
      test_util.expectArraysClose(dl.notEqual(a, b), [1, 1, 1, 1, 1, 1]);

      a = dl.tensor2d([[0, 0], [1, 1]], [2, 2], 'int32');
      b = dl.tensor2d([[0, 0], [1, 1]], [2, 2], 'int32');
      test_util.expectArraysClose(dl.notEqual(a, b), [0, 0, 0, 0]);
    });
    it('Tensor2D - float32', () => {
      let a =
          dl.tensor2d([[1.1, 4.1, 5.1], [8.1, 9.1, 12.1]], [2, 3], 'float32');
      let b =
          dl.tensor2d([[2.1, 4.1, 5.1], [7.1, 10.1, 11.1]], [2, 3], 'float32');
      test_util.expectArraysClose(dl.notEqual(a, b), [1, 0, 0, 1, 1, 1]);

      a = dl.tensor2d([[0.2, 0.2], [1.2, 1.2]], [2, 2], 'float32');
      b = dl.tensor2d([[0.2, 0.2], [1.2, 1.2]], [2, 2], 'float32');
      test_util.expectArraysClose(dl.notEqual(a, b), [0, 0, 0, 0]);
    });
    it('broadcasting Tensor2D shapes - int32', () => {
      const a = dl.tensor2d([[3], [7]], [2, 1], 'int32');
      const b = dl.tensor2d([[2, 3, 4], [7, 8, 9]], [2, 3], 'int32');
      test_util.expectArraysClose(dl.notEqual(a, b), [1, 0, 1, 0, 1, 1]);
    });
    it('broadcasting Tensor2D shapes - float32', () => {
      const a = dl.tensor2d([[1.1], [7.1]], [2, 1], 'float32');
      const b =
          dl.tensor2d([[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], [2, 3], 'float32');
      test_util.expectArraysClose(dl.notEqual(a, b), [1, 0, 1, 0, 1, 1]);
    });
    it('NaNs in Tensor2D - int32', () => {
      const a = dl.tensor2d([[1, NaN, 2], [1, NaN, NaN]], [2, 3], 'int32');
      const b = dl.tensor2d([[0, NaN, NaN], [1, NaN, 3]], [2, 3], 'int32');
      test_util.expectArraysClose(
          dl.notEqual(a, b), [1, boolNaN, boolNaN, 0, boolNaN, boolNaN]);
    });
    it('NaNs in Tensor2D - float32', () => {
      const a = dl.tensor2d([[1.1, NaN], [1.1, NaN]], [2, 2], 'float32');
      const b = dl.tensor2d([[0.1, NaN], [1.1, NaN]], [2, 2], 'float32');
      test_util.expectArraysClose(
          dl.notEqual(a, b), [1, boolNaN, 0, boolNaN]);
    });
    it('2D and scalar broadcast', () => {
      const a = dl.tensor2d([1, 2, 3, 2, 5, 6], [2, 3]);
      const b = dl.scalar(2);
      const res = dl.notEqual(a, b);
      expect(res.dtype).toBe('bool');
      expect(res.shape).toEqual([2, 3]);
      test_util.expectArraysEqual(res, [1, 0, 1, 0, 1, 1]);
    });
    it('2D and 2D broadcast each with 1 dim', () => {
      const a = dl.tensor2d([1, 2, 5], [1, 3]);
      const b = dl.tensor2d([5, 1], [2, 1]);
      const res = dl.notEqual(a, b);
      expect(res.dtype).toBe('bool');
      expect(res.shape).toEqual([2, 3]);
      test_util.expectArraysEqual(res, [1, 1, 0, 0, 1, 1]);
    });

    // Tensor3D:
    it('Tensor3D - int32', () => {
      let a =
          Tensor3D.new([2, 3, 1], [[[1], [4], [5]], [[8], [9], [12]]], 'int32');
      let b =
          Tensor3D.new([2, 3, 1],
            [[[2], [3], [6]], [[7], [10], [12]]], 'int32');
      test_util.expectArraysClose(dl.notEqual(a, b), [1, 1, 1, 1, 1, 0]);

      a = Tensor3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
      b = Tensor3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
      test_util.expectArraysClose(dl.notEqual(a, b), [0, 0, 0, 0, 0, 0]);
    });
    it('Tensor3D - float32', () => {
      let a = Tensor3D.new(
          [2, 3, 1], [[[1.1], [4.1], [5.1]], [[8.1], [9.1], [12.1]]],
          'float32');
      let b = Tensor3D.new(
          [2, 3, 1], [[[2.1], [3.1], [6.1]], [[7.1], [10.1], [12.1]]],
          'float32');
      test_util.expectArraysClose(dl.notEqual(a, b), [1, 1, 1, 1, 1, 0]);

      a = Tensor3D.new(
          [2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
      b = Tensor3D.new(
          [2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
      test_util.expectArraysClose(dl.notEqual(a, b), [0, 0, 0, 0, 0, 0]);
    });
    it('broadcasting Tensor3D shapes - int32', () => {
      const a = Tensor3D.new(
          [2, 3, 2], [[[1, 0], [2, 3], [4, 5]], [[6, 7], [9, 8], [10, 11]]],
          'int32');
      const b =
          Tensor3D.new([2, 3, 1], [[[1], [2], [3]], [[7], [10], [9]]], 'int32');
      test_util.expectArraysClose(
          dl.notEqual(a, b), [0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1]);
    });
    it('broadcasting Tensor3D shapes - float32', () => {
      const a = Tensor3D.new(
          [2, 3, 2],
          [
            [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
            [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
          ],
          'float32');
      const b = Tensor3D.new(
          [2, 3, 1], [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]],
          'float32');
      test_util.expectArraysClose(
          dl.notEqual(a, b), [0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1]);
    });
    it('NaNs in Tensor3D - int32', () => {
      const a =
          Tensor3D.new([2, 3, 1],
            [[[1], [NaN], [1]], [[0], [0], [0]]], 'int32');
      const b =
          Tensor3D.new([2, 3, 1],
            [[[0], [0], [1]], [[1], [0], [NaN]]], 'int32');
      test_util.expectArraysClose(
          dl.notEqual(a, b), [1, boolNaN, 0, 1, 0, boolNaN]);
    });
    it('NaNs in Tensor3D - float32', () => {
      const a = Tensor3D.new(
          [2, 3, 1], [[[1.1], [NaN], [1.1]], [[0.1], [0.1], [0.1]]], 'float32');
      const b = Tensor3D.new(
          [2, 3, 1], [[[0.1], [0.1], [1.1]], [[1.1], [0.1], [NaN]]], 'float32');
      test_util.expectArraysClose(
          dl.notEqual(a, b), [1, boolNaN, 0, 1, 0, boolNaN]);
    });
    it('3D and scalar', () => {
      const a = Tensor3D.new([2, 3, 1], [1, 2, 3, 4, 5, -1]);
      const b = dl.scalar(-1);
      const res = dl.notEqual(a, b);
      expect(res.dtype).toBe('bool');
      expect(res.shape).toEqual([2, 3, 1]);
      test_util.expectArraysEqual(res, [1, 1, 1, 1, 1, 0]);
    });

    // Tensor4D:
    it('Tensor4D - int32', () => {
      let a = Tensor4D.new([2, 2, 1, 1], [1, 4, 5, 8], 'int32');
      let b = Tensor4D.new([2, 2, 1, 1], [2, 3, 6, 8], 'int32');
      test_util.expectArraysClose(dl.notEqual(a, b), [1, 1, 1, 0]);

      a = Tensor4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
      b = Tensor4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
      test_util.expectArraysClose(dl.notEqual(a, b), [0, 0, 0, 0]);

      a = Tensor4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'int32');
      b = Tensor4D.new([2, 2, 1, 1], [2, 2, 2, 2], 'int32');
      test_util.expectArraysClose(dl.notEqual(a, b), [1, 1, 1, 1]);
    });
    it('Tensor4D - float32', () => {
      let a = Tensor4D.new([2, 2, 1, 1], [1.1, 4.1, 5.1, 8.1], 'float32');
      let b = Tensor4D.new([2, 2, 1, 1], [2.1, 3.1, 6.1, 8.1], 'float32');
      test_util.expectArraysClose(dl.notEqual(a, b), [1, 1, 1, 0]);

      a = Tensor4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
      b = Tensor4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
      test_util.expectArraysClose(dl.notEqual(a, b), [0, 0, 0, 0]);

      a = Tensor4D.new([2, 2, 1, 1], [0.1, 0.1, 0.1, 0.1], 'float32');
      b = Tensor4D.new([2, 2, 1, 1], [1.1, 1.1, 1.1, 1.1], 'float32');
      test_util.expectArraysClose(dl.notEqual(a, b), [1, 1, 1, 1]);
    });
    it('broadcasting Tensor4D shapes - int32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1, 2, 5, 9], 'int32');
      const b = Tensor4D.new(
          [2, 2, 1, 2], [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], 'int32');
      test_util.expectArraysClose(
          dl.notEqual(a, b), [0, 1, 1, 1, 0, 1, 1, 1]);
    });
    it('broadcasting Tensor4D shapes - float32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1.1, 2.1, 5.1, 9.1], 'float32');
      const b = Tensor4D.new(
          [2, 2, 1, 2],
          [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]],
          'float32');
      test_util.expectArraysClose(
          dl.notEqual(a, b), [0, 1, 1, 1, 0, 1, 1, 1]);
    });
    it('NaNs in Tensor4D - int32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1, NaN, 1, 0], 'int32');
      const b = Tensor4D.new([2, 2, 1, 1], [0, 1, 1, NaN], 'int32');
      test_util.expectArraysClose(
          dl.notEqual(a, b), [1, boolNaN, 0, boolNaN]);
    });
    it('NaNs in Tensor4D - float32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1.1, NaN, 1.1, 0.1], 'float32');
      const b = Tensor4D.new([2, 2, 1, 1], [0.1, 1.1, 1.1, NaN], 'float32');
      test_util.expectArraysClose(
          dl.notEqual(a, b), [1, boolNaN, 0, boolNaN]);
    });
  };

  test_util.describeMathCPU('notEqual', [tests]);
  test_util.describeMathGPU('notEqual', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// NotEqualStrict:
{
  const boolNaN = util.getNaN('bool');

  const tests: MathTests = it => {
    // Tensor1D:
    it('Tensor1D - int32', () => {
      let a = dl.tensor1d([1, 4, 5], 'int32');
      let b = dl.tensor1d([2, 3, 5], 'int32');
      test_util.expectArraysClose(dl.notEqualStrict(a, b), [1, 1, 0]);

      a = dl.tensor1d([2, 2, 2], 'int32');
      b = dl.tensor1d([2, 2, 2], 'int32');
      test_util.expectArraysClose(dl.notEqualStrict(a, b), [0, 0, 0]);

      a = dl.tensor1d([0, 0], 'int32');
      b = dl.tensor1d([3, 3], 'int32');
      test_util.expectArraysClose(dl.notEqualStrict(a, b), [1, 1]);
    });
    it('Tensor1D - float32', () => {
      let a = dl.tensor1d([1.1, 4.1, 5.1], 'float32');
      let b = dl.tensor1d([2.2, 3.2, 5.1], 'float32');
      test_util.expectArraysClose(dl.notEqualStrict(a, b), [1, 1, 0]);

      a = dl.tensor1d([2.31, 2.31, 2.31], 'float32');
      b = dl.tensor1d([2.31, 2.31, 2.31], 'float32');
      test_util.expectArraysClose(dl.notEqualStrict(a, b), [0, 0, 0]);

      a = dl.tensor1d([0.45, 0.123], 'float32');
      b = dl.tensor1d([3.123, 3.321], 'float32');
      test_util.expectArraysClose(dl.notEqualStrict(a, b), [1, 1]);
    });
    it('mismatched Tensor1D shapes - int32', () => {
      const a = dl.tensor1d([1, 2], 'int32');
      const b = dl.tensor1d([1, 2, 3], 'int32');
      const f = () => {
        dl.notEqualStrict(a, b);
      };
      expect(f).toThrowError();
    });
    it('mismatched Tensor1D shapes - float32', () => {
      const a = dl.tensor1d([1.1, 2.1], 'float32');
      const b = dl.tensor1d([1.1, 2.1, 3.1], 'float32');
      const f = () => {
        dl.notEqualStrict(a, b);
      };
      expect(f).toThrowError();
    });
    it('NaNs in Tensor1D - int32', () => {
      const a = dl.tensor1d([1, NaN, 0], 'int32');
      const b = dl.tensor1d([0, 0, NaN], 'int32');
      test_util.expectArraysClose(
          dl.notEqualStrict(a, b), [1, boolNaN, boolNaN]);
    });
    it('NaNs in Tensor1D - float32', () => {
      const a = dl.tensor1d([1.1, NaN, 2.1], 'float32');
      const b = dl.tensor1d([2.1, 3.1, NaN], 'float32');
      test_util.expectArraysClose(
          dl.notEqualStrict(a, b), [1, boolNaN, boolNaN]);
    });
    it('strict version throws when x and y are different shape', () => {
      const a = dl.tensor1d([2]);
      const b = dl.tensor1d([4, 2, -1]);

      expect(() => dl.notEqualStrict(a, b)).toThrowError();
      expect(() => dl.notEqualStrict(b, a)).toThrowError();
    });

    // Tensor2D:
    it('Tensor2D - int32', () => {
      let a = dl.tensor2d([[1, 4, 5], [8, 9, 12]], [2, 3], 'int32');
      let b = dl.tensor2d([[2, 3, 6], [7, 10, 11]], [2, 3], 'int32');
      test_util.expectArraysClose(
          dl.notEqualStrict(a, b), [1, 1, 1, 1, 1, 1]);

      a = dl.tensor2d([[0, 0], [1, 1]], [2, 2], 'int32');
      b = dl.tensor2d([[0, 0], [1, 1]], [2, 2], 'int32');
      test_util.expectArraysClose(dl.notEqualStrict(a, b), [0, 0, 0, 0]);
    });
    it('Tensor2D - float32', () => {
      let a =
          dl.tensor2d([[1.1, 4.1, 5.1], [8.1, 9.1, 12.1]], [2, 3], 'float32');
      let b =
          dl.tensor2d([[2.1, 4.1, 5.1], [7.1, 10.1, 11.1]], [2, 3], 'float32');
      test_util.expectArraysClose(
          dl.notEqualStrict(a, b), [1, 0, 0, 1, 1, 1]);

      a = dl.tensor2d([[0.2, 0.2], [1.2, 1.2]], [2, 2], 'float32');
      b = dl.tensor2d([[0.2, 0.2], [1.2, 1.2]], [2, 2], 'float32');
      test_util.expectArraysClose(dl.notEqualStrict(a, b), [0, 0, 0, 0]);
    });
    it('mismatch Tensor2D shapes - int32', () => {
      const a = dl.tensor2d([[3], [7]], [2, 1], 'int32');
      const b = dl.tensor2d([[2, 3, 4], [7, 8, 9]], [2, 3], 'int32');

      const f = () => {
        dl.notEqualStrict(a, b);
      };
      expect(f).toThrowError();
    });
    it('mismatch Tensor2D shapes - float32', () => {
      const a = dl.tensor2d([[1.1], [7.1]], [2, 1], 'float32');
      const b =
          dl.tensor2d([[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], [2, 3], 'float32');
      const f = () => {
        dl.notEqualStrict(a, b);
      };
      expect(f).toThrowError();
    });
    it('NaNs in Tensor2D - int32', () => {
      const a = dl.tensor2d([[1, NaN, 2], [1, NaN, NaN]], [2, 3], 'int32');
      const b = dl.tensor2d([[0, NaN, NaN], [1, NaN, 3]], [2, 3], 'int32');
      test_util.expectArraysClose(
          dl.notEqualStrict(a, b),
          [1, boolNaN, boolNaN, 0, boolNaN, boolNaN]);
    });
    it('NaNs in Tensor2D - float32', () => {
      const a = dl.tensor2d([[1.1, NaN], [1.1, NaN]], [2, 2], 'float32');
      const b = dl.tensor2d([[0.1, NaN], [1.1, NaN]], [2, 2], 'float32');
      test_util.expectArraysClose(
          dl.notEqualStrict(a, b), [1, boolNaN, 0, boolNaN]);
    });

    // Tensor3D:
    it('Tensor3D - int32', () => {
      let a =
          Tensor3D.new([2, 3, 1], [[[1], [4], [5]], [[8], [9], [12]]], 'int32');
      let b =
          Tensor3D.new([2, 3, 1],
            [[[2], [3], [6]], [[7], [10], [12]]], 'int32');
      test_util.expectArraysClose(
          dl.notEqualStrict(a, b), [1, 1, 1, 1, 1, 0]);

      a = Tensor3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
      b = Tensor3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
      test_util.expectArraysClose(
          dl.notEqualStrict(a, b), [0, 0, 0, 0, 0, 0]);
    });
    it('Tensor3D - float32', () => {
      let a = Tensor3D.new(
          [2, 3, 1], [[[1.1], [4.1], [5.1]], [[8.1], [9.1], [12.1]]],
          'float32');
      let b = Tensor3D.new(
          [2, 3, 1], [[[2.1], [3.1], [6.1]], [[7.1], [10.1], [12.1]]],
          'float32');
      test_util.expectArraysClose(
          dl.notEqualStrict(a, b), [1, 1, 1, 1, 1, 0]);

      a = Tensor3D.new(
          [2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
      b = Tensor3D.new(
          [2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
      test_util.expectArraysClose(
          dl.notEqualStrict(a, b), [0, 0, 0, 0, 0, 0]);
    });
    it('mismatch Tensor3D shapes - int32', () => {
      const a = Tensor3D.new(
          [2, 3, 2], [[[1, 0], [2, 3], [4, 5]], [[6, 7], [9, 8], [10, 11]]],
          'int32');
      const b =
          Tensor3D.new([2, 3, 1], [[[1], [2], [3]], [[7], [10], [9]]], 'int32');

      const f = () => {
        dl.notEqualStrict(a, b);
      };
      expect(f).toThrowError();
    });
    it('mismatch Tensor3D shapes - float32', () => {
      const a = Tensor3D.new(
          [2, 3, 2],
          [
            [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
            [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
          ],
          'float32');
      const b = Tensor3D.new(
          [2, 3, 1], [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]],
          'float32');

      const f = () => {
        dl.notEqualStrict(a, b);
      };
      expect(f).toThrowError();
    });
    it('NaNs in Tensor3D - int32', () => {
      const a =
          Tensor3D.new([2, 3, 1],
            [[[1], [NaN], [1]], [[0], [0], [0]]], 'int32');
      const b =
          Tensor3D.new([2, 3, 1],
            [[[0], [0], [1]], [[1], [0], [NaN]]], 'int32');
      test_util.expectArraysClose(
          dl.notEqualStrict(a, b), [1, boolNaN, 0, 1, 0, boolNaN]);
    });
    it('NaNs in Tensor3D - float32', () => {
      const a = Tensor3D.new(
          [2, 3, 1], [[[1.1], [NaN], [1.1]], [[0.1], [0.1], [0.1]]], 'float32');
      const b = Tensor3D.new(
          [2, 3, 1], [[[0.1], [0.1], [1.1]], [[1.1], [0.1], [NaN]]], 'float32');
      test_util.expectArraysClose(
          dl.notEqualStrict(a, b), [1, boolNaN, 0, 1, 0, boolNaN]);
    });

    // Tensor4D:
    it('Tensor4D - int32', () => {
      let a = Tensor4D.new([2, 2, 1, 1], [1, 4, 5, 8], 'int32');
      let b = Tensor4D.new([2, 2, 1, 1], [2, 3, 6, 8], 'int32');
      test_util.expectArraysClose(dl.notEqualStrict(a, b), [1, 1, 1, 0]);

      a = Tensor4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
      b = Tensor4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
      test_util.expectArraysClose(dl.notEqualStrict(a, b), [0, 0, 0, 0]);

      a = Tensor4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'int32');
      b = Tensor4D.new([2, 2, 1, 1], [2, 2, 2, 2], 'int32');
      test_util.expectArraysClose(dl.notEqualStrict(a, b), [1, 1, 1, 1]);
    });
    it('Tensor4D - float32', () => {
      let a = Tensor4D.new([2, 2, 1, 1], [1.1, 4.1, 5.1, 8.1], 'float32');
      let b = Tensor4D.new([2, 2, 1, 1], [2.1, 3.1, 6.1, 8.1], 'float32');
      test_util.expectArraysClose(dl.notEqualStrict(a, b), [1, 1, 1, 0]);

      a = Tensor4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
      b = Tensor4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
      test_util.expectArraysClose(dl.notEqualStrict(a, b), [0, 0, 0, 0]);

      a = Tensor4D.new([2, 2, 1, 1], [0.1, 0.1, 0.1, 0.1], 'float32');
      b = Tensor4D.new([2, 2, 1, 1], [1.1, 1.1, 1.1, 1.1], 'float32');
      test_util.expectArraysClose(dl.notEqualStrict(a, b), [1, 1, 1, 1]);
    });
    it('mismatch Tensor4D shapes - int32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1, 2, 5, 9], 'int32');
      const b = Tensor4D.new(
          [2, 2, 1, 2], [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], 'int32');

      const f = () => {
        dl.notEqualStrict(a, b);
      };
      expect(f).toThrowError();
    });
    it('mismatch Tensor4D shapes - float32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1.1, 2.1, 5.1, 9.1], 'float32');
      const b = Tensor4D.new(
          [2, 2, 1, 2],
          [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]],
          'float32');

      const f = () => {
        dl.notEqualStrict(a, b);
      };
      expect(f).toThrowError();
    });
    it('NaNs in Tensor4D - int32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1, NaN, 1, 0], 'int32');
      const b = Tensor4D.new([2, 2, 1, 1], [0, 1, 1, NaN], 'int32');
      test_util.expectArraysClose(
          dl.notEqualStrict(a, b), [1, boolNaN, 0, boolNaN]);
    });
    it('NaNs in Tensor4D - float32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1.1, NaN, 1.1, 0.1], 'float32');
      const b = Tensor4D.new([2, 2, 1, 1], [0.1, 1.1, 1.1, NaN], 'float32');
      test_util.expectArraysClose(
          dl.notEqualStrict(a, b), [1, boolNaN, 0, boolNaN]);
    });
  };

  test_util.describeMathCPU('notEqualStrict', [tests]);
  test_util.describeMathGPU('notEqualStrict', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// Less:
{
  const boolNaN = util.getNaN('bool');

  const tests: MathTests = it => {
    // Tensor1D:
    it('Tensor1D - int32', () => {
      let a = dl.tensor1d([1, 4, 5], 'int32');
      let b = dl.tensor1d([2, 3, 5], 'int32');
      let res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 0]);

      a = dl.tensor1d([2, 2, 2], 'int32');
      b = dl.tensor1d([2, 2, 2], 'int32');
      res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0]);

      a = dl.tensor1d([0, 0], 'int32');
      b = dl.tensor1d([3, 3], 'int32');
      res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1]);
    });
    it('Tensor1D - float32', () => {
      let a = dl.tensor1d([1.1, 4.1, 5.1], 'float32');
      let b = dl.tensor1d([2.2, 3.2, 5.1], 'float32');
      let res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 0]);

      a = dl.tensor1d([2.31, 2.31, 2.31], 'float32');
      b = dl.tensor1d([2.31, 2.31, 2.31], 'float32');
      res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0]);

      a = dl.tensor1d([0.45, 0.123], 'float32');
      b = dl.tensor1d([3.123, 3.321], 'float32');
      res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1]);
    });
    it('mismatched Tensor1D shapes - int32', () => {
      const a = dl.tensor1d([1, 2], 'int32');
      const b = dl.tensor1d([1, 2, 3], 'int32');
      const f = () => {
        dl.less(a, b);
      };
      expect(f).toThrowError();
    });
    it('mismatched Tensor1D shapes - float32', () => {
      const a = dl.tensor1d([1.1, 2.1], 'float32');
      const b = dl.tensor1d([1.1, 2.1, 3.1], 'float32');
      const f = () => {
        dl.less(a, b);
      };
      expect(f).toThrowError();
    });
    it('NaNs in Tensor1D - int32', () => {
      const a = dl.tensor1d([1, NaN, 0], 'int32');
      const b = dl.tensor1d([0, 0, NaN], 'int32');
      const res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, boolNaN, boolNaN]);
    });
    it('NaNs in Tensor1D - float32', () => {
      const a = dl.tensor1d([1.1, NaN, 2.1], 'float32');
      const b = dl.tensor1d([2.1, 3.1, NaN], 'float32');
      const res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, boolNaN, boolNaN]);
    });

    // Tensor2D:
    it('Tensor2D - int32', () => {
      let a = dl.tensor2d([[1, 4, 5], [8, 9, 12]], [2, 3], 'int32');
      let b = dl.tensor2d([[2, 3, 6], [7, 10, 11]], [2, 3], 'int32');
      let res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1, 0, 1, 0]);

      a = dl.tensor2d([[0, 0], [1, 1]], [2, 2], 'int32');
      b = dl.tensor2d([[0, 0], [1, 1]], [2, 2], 'int32');
      res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0]);
    });
    it('Tensor2D - float32', () => {
      let a =
          dl.tensor2d([[1.1, 4.1, 5.1], [8.1, 9.1, 12.1]], [2, 3], 'float32');
      let b =
          dl.tensor2d([[2.1, 3.1, 6.1], [7.1, 10.1, 11.1]], [2, 3], 'float32');
      let res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1, 0, 1, 0]);

      a = dl.tensor2d([[0.2, 0.2], [1.2, 1.2]], [2, 2], 'float32');
      b = dl.tensor2d([[0.2, 0.2], [1.2, 1.2]], [2, 2], 'float32');
      res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0]);
    });
    it('broadcasting Tensor2D shapes - int32', () => {
      const a = dl.tensor2d([[3], [7]], [2, 1], 'int32');
      const b = dl.tensor2d([[2, 3, 4], [7, 8, 9]], [2, 3], 'int32');
      const res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 1, 0, 1, 1]);
    });
    it('broadcasting Tensor2D shapes - float32', () => {
      const a = dl.tensor2d([[1.1], [7.1]], [2, 1], 'float32');
      const b =
          dl.tensor2d([[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], [2, 3], 'float32');
      const res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 1, 0, 1, 1]);
    });
    it('NaNs in Tensor2D - int32', () => {
      const a = dl.tensor2d([[1, NaN, 2], [0, NaN, NaN]], [2, 3], 'int32');
      const b = dl.tensor2d([[0, NaN, NaN], [1, NaN, 3]], [2, 3], 'int32');
      const res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(
          res, [0, boolNaN, boolNaN, 1, boolNaN, boolNaN]);
    });
    it('NaNs in Tensor2D - float32', () => {
      const a = dl.tensor2d([[1.1, NaN], [0.1, NaN]], [2, 2], 'float32');
      const b = dl.tensor2d([[0.1, NaN], [1.1, NaN]], [2, 2], 'float32');
      const res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, boolNaN, 1, boolNaN]);
    });

    // Tensor3D:
    it('Tensor3D - int32', () => {
      let a =
          Tensor3D.new([2, 3, 1], [[[1], [4], [5]], [[8], [9], [12]]], 'int32');
      let b =
          Tensor3D.new([2, 3, 1],
            [[[2], [3], [6]], [[7], [10], [11]]], 'int32');
      let res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1, 0, 1, 0]);

      a = Tensor3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
      b = Tensor3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
      res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0, 0, 0]);
    });
    it('Tensor3D - float32', () => {
      let a = Tensor3D.new(
          [2, 3, 1], [[[1.1], [4.1], [5.1]], [[8.1], [9.1], [12.1]]],
          'float32');
      let b = Tensor3D.new(
          [2, 3, 1], [[[2.1], [3.1], [6.1]], [[7.1], [10.1], [11.1]]],
          'float32');
      let res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1, 0, 1, 0]);

      a = Tensor3D.new(
          [2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.0]]], 'float32');
      b = Tensor3D.new(
          [2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
      res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0, 0, 1]);
    });
    it('broadcasting Tensor3D shapes - int32', () => {
      const a = Tensor3D.new(
          [2, 3, 2], [[[1, 0], [2, 3], [4, 5]], [[6, 7], [9, 8], [10, 11]]],
          'int32');
      const b =
          Tensor3D.new([2, 3, 1], [[[1], [2], [3]], [[7], [10], [9]]], 'int32');
      const res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0]);
    });
    it('broadcasting Tensor3D float32', () => {
      const a = Tensor3D.new(
          [2, 3, 2],
          [
            [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
            [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
          ],
          'float32');
      const b = Tensor3D.new(
          [2, 3, 1], [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]],
          'float32');
      const res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0]);
    });
    it('NaNs in Tensor3D - int32', () => {
      const a =
          Tensor3D.new([2, 3, 1],
            [[[1], [NaN], [1]], [[0], [0], [0]]], 'int32');
      const b =
          Tensor3D.new([2, 3, 1],
            [[[0], [0], [1]], [[1], [0], [NaN]]], 'int32');
      const res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, boolNaN, 0, 1, 0, boolNaN]);
    });
    it('NaNs in Tensor3D - float32', () => {
      const a = Tensor3D.new(
          [2, 3, 1], [[[1.1], [NaN], [1.1]], [[0.1], [0.1], [0.1]]], 'float32');
      const b = Tensor3D.new(
          [2, 3, 1], [[[0.1], [0.1], [1.1]], [[1.1], [0.1], [NaN]]], 'float32');
      const res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, boolNaN, 0, 1, 0, boolNaN]);
    });

    // Tensor4D:
    it('Tensor4D - int32', () => {
      let a = Tensor4D.new([2, 2, 1, 1], [1, 4, 5, 8], 'int32');
      let b = Tensor4D.new([2, 2, 1, 1], [2, 3, 6, 7], 'int32');
      let res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1, 0]);

      a = Tensor4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
      b = Tensor4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
      res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0]);

      a = Tensor4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'int32');
      b = Tensor4D.new([2, 2, 1, 1], [2, 2, 2, 2], 'int32');
      res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1]);
    });
    it('Tensor4D - float32', () => {
      let a = Tensor4D.new([2, 2, 1, 1], [1.1, 4.1, 5.1, 8.1], 'float32');
      let b = Tensor4D.new([2, 2, 1, 1], [2.1, 3.1, 6.1, 7.1], 'float32');
      let res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1, 0]);

      a = Tensor4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
      b = Tensor4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
      res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0]);

      a = Tensor4D.new([2, 2, 1, 1], [0.1, 0.1, 0.1, 0.1], 'float32');
      b = Tensor4D.new([2, 2, 1, 1], [1.1, 1.1, 1.1, 1.1], 'float32');
      res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1]);
    });
    it('broadcasting Tensor4D shapes - int32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1, 2, 5, 9], 'int32');
      const b = Tensor4D.new(
          [2, 2, 1, 2], [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], 'int32');
      const res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 1, 1, 0, 1, 0, 0]);
    });
    it('broadcasting Tensor4D shapes - float32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1.1, 2.1, 5.1, 9.1], 'float32');
      const b = Tensor4D.new(
          [2, 2, 1, 2],
          [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]],
          'float32');
      const res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 1, 1, 0, 1, 0, 0]);
    });
    it('NaNs in Tensor4D - int32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1, NaN, 0, 0], 'int32');
      const b = Tensor4D.new([2, 2, 1, 1], [0, 1, 1, NaN], 'int32');
      const res = dl.less(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, boolNaN, 1, boolNaN]);
    });
    it('NaNs in Tensor4D - float32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1.1, NaN, 0.1, 0.1], 'float32');
      const b = Tensor4D.new([2, 2, 1, 1], [0.1, 1.1, 1.1, NaN], 'float32');
      const res = dl.less(a, b);

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

// LessStrict:
{

  const tests: MathTests = it => {
    // Tensor1D:
    it('Tensor1D - strict version throws when a and b are different shape',
       () => {
         const a = dl.tensor1d([2]);
         const b = dl.tensor1d([4, 2, -1]);

         expect(() => dl.lessStrict(a, b)).toThrowError();
         expect(() => dl.lessStrict(b, a)).toThrowError();
       });

    // Tensor2D:
    it('Tensor2D - strict version throws when a and b are different shape',
       () => {
         const a = dl.tensor2d([[1.1], [7.1]], [2, 1], 'float32');
         const b =
             Tensor2D.new([2, 3],
              [[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], 'float32');

         expect(() => dl.lessStrict(a, b)).toThrowError();
         expect(() => dl.lessStrict(b, a)).toThrowError();
       });

    // Tensor3D:
    it('Tensor3D - strict version throws when a and b are different shape',
       () => {
         const a = Tensor3D.new(
             [2, 3, 2],
             [
               [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
               [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
             ],
             'float32');
         const b = Tensor3D.new(
             [2, 3, 1], [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]],
             'float32');

         expect(() => dl.lessStrict(a, b)).toThrowError();
         expect(() => dl.lessStrict(b, a)).toThrowError();
       });

    // Tensor4D:
    it('Tensor4D - strict version throws when a and b are different shape',
       () => {
         const a = Tensor4D.new([2, 2, 1, 1], [1.1, 2.1, 5.1, 9.1], 'float32');
         const b = Tensor4D.new(
             [2, 2, 1, 2],
             [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]],
             'float32');

         expect(() => dl.lessStrict(a, b)).toThrowError();
         expect(() => dl.lessStrict(b, a)).toThrowError();
       });
  };

  test_util.describeMathCPU('lessStrict', [tests]);
  test_util.describeMathGPU('lessStrict', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// LessEqual:
{
  const boolNaN = util.getNaN('bool');

  const tests: MathTests = it => {
    // Tensor1D:
    it('Tensor1D - int32', () => {
      let a = dl.tensor1d([1, 4, 5], 'int32');
      let b = dl.tensor1d([2, 3, 5], 'int32');
      let res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1]);

      a = dl.tensor1d([2, 2, 2], 'int32');
      b = dl.tensor1d([2, 2, 2], 'int32');
      res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1]);

      a = dl.tensor1d([0, 0], 'int32');
      b = dl.tensor1d([3, 3], 'int32');
      res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1]);
    });
    it('Tensor1D - float32', () => {
      let a = dl.tensor1d([1.1, 4.1, 5.1], 'float32');
      let b = dl.tensor1d([2.2, 3.2, 5.1], 'float32');
      let res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1]);

      a = dl.tensor1d([2.31, 2.31, 2.31], 'float32');
      b = dl.tensor1d([2.31, 2.31, 2.31], 'float32');
      res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1]);

      a = dl.tensor1d([0.45, 0.123], 'float32');
      b = dl.tensor1d([3.123, 3.321], 'float32');
      res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1]);
    });
    it('mismatched Tensor1D shapes - int32', () => {
      const a = dl.tensor1d([1, 2], 'int32');
      const b = dl.tensor1d([1, 2, 3], 'int32');
      const f = () => {
        dl.lessEqual(a, b);
      };
      expect(f).toThrowError();
    });
    it('mismatched Tensor1D shapes - float32', () => {
      const a = dl.tensor1d([1.1, 2.1], 'float32');
      const b = dl.tensor1d([1.1, 2.1, 3.1], 'float32');
      const f = () => {
        dl.lessEqual(a, b);
      };
      expect(f).toThrowError();
    });
    it('NaNs in Tensor1D - int32', () => {
      const a = dl.tensor1d([1, NaN, 0], 'int32');
      const b = dl.tensor1d([0, 0, NaN], 'int32');
      const res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, boolNaN, boolNaN]);
    });
    it('NaNs in Tensor1D - float32', () => {
      const a = dl.tensor1d([1.1, NaN, 2.1], 'float32');
      const b = dl.tensor1d([2.1, 3.1, NaN], 'float32');
      const res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, boolNaN, boolNaN]);
    });

    // Tensor2D:
    it('Tensor2D - int32', () => {
      let a = dl.tensor2d([[1, 4, 5], [8, 9, 12]], [2, 3], 'int32');
      let b = dl.tensor2d([[2, 3, 6], [7, 10, 11]], [2, 3], 'int32');
      let res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1, 0, 1, 0]);

      a = dl.tensor2d([[0, 0], [1, 1]], [2, 2], 'int32');
      b = dl.tensor2d([[0, 0], [1, 1]], [2, 2], 'int32');
      res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1]);
    });
    it('Tensor2D - float32', () => {
      let a =
          dl.tensor2d([[1.1, 4.1, 5.1], [8.1, 9.1, 12.1]], [2, 3], 'float32');
      let b =
          dl.tensor2d([[2.1, 3.1, 6.1], [7.1, 10.1, 11.1]], [2, 3], 'float32');
      let res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1, 0, 1, 0]);

      a = dl.tensor2d([[0.2, 0.2], [1.2, 1.2]], [2, 2], 'float32');
      b = dl.tensor2d([[0.2, 0.2], [1.2, 1.2]], [2, 2], 'float32');
      res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1]);
    });
    it('broadcasting Tensor2D shapes - int32', () => {
      const a = dl.tensor2d([[3], [7]], [2, 1], 'int32');
      const b = dl.tensor2d([[2, 3, 4], [7, 8, 9]], [2, 3], 'int32');
      const res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 1, 1, 1, 1]);
    });
    it('broadcasting Tensor2D shapes - float32', () => {
      const a = dl.tensor2d([[1.1], [7.1]], [2, 1], 'float32');
      const b =
          dl.tensor2d([[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], [2, 3], 'float32');
      const res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 1, 1, 1, 1]);
    });
    it('NaNs in Tensor2D - int32', () => {
      const a = dl.tensor2d([[1, NaN, 2], [0, NaN, NaN]], [2, 3], 'int32');
      const b = dl.tensor2d([[0, NaN, NaN], [1, NaN, 3]], [2, 3], 'int32');
      const res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(
          res, [0, boolNaN, boolNaN, 1, boolNaN, boolNaN]);
    });
    it('NaNs in Tensor2D - float32', () => {
      const a = dl.tensor2d([[1.1, NaN], [0.1, NaN]], [2, 2], 'float32');
      const b = dl.tensor2d([[0.1, NaN], [1.1, NaN]], [2, 2], 'float32');
      const res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, boolNaN, 1, boolNaN]);
    });

    // Tensor3D:
    it('Tensor3D - int32', () => {
      let a =
          Tensor3D.new([2, 3, 1], [[[1], [4], [5]], [[8], [9], [12]]], 'int32');
      let b =
          Tensor3D.new([2, 3, 1],
            [[[2], [3], [6]], [[7], [10], [11]]], 'int32');
      let res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1, 0, 1, 0]);

      a = Tensor3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
      b = Tensor3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
      res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1, 1, 1]);
    });
    it('Tensor3D - float32', () => {
      let a = Tensor3D.new(
          [2, 3, 1], [[[1.1], [4.1], [5.1]], [[8.1], [9.1], [12.1]]],
          'float32');
      let b = Tensor3D.new(
          [2, 3, 1], [[[2.1], [3.1], [6.1]], [[7.1], [10.1], [11.1]]],
          'float32');
      let res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1, 0, 1, 0]);

      a = Tensor3D.new(
          [2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.2]]], 'float32');
      b = Tensor3D.new(
          [2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
      res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1, 1, 0]);
    });
    it('broadcasting Tensor3D shapes - int32', () => {
      const a = Tensor3D.new(
          [2, 3, 2], [[[1, 0], [2, 3], [4, 5]], [[6, 7], [9, 8], [10, 11]]],
          'int32');
      const b =
          Tensor3D.new([2, 3, 1], [[[1], [2], [3]], [[7], [10], [9]]], 'int32');
      const res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0]);
    });
    it('broadcasting Tensor3D float32', () => {
      const a = Tensor3D.new(
          [2, 3, 2],
          [
            [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
            [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
          ],
          'float32');
      const b = Tensor3D.new(
          [2, 3, 1], [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]],
          'float32');
      const res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0]);
    });
    it('NaNs in Tensor3D - int32', () => {
      const a =
          Tensor3D.new([2, 3, 1],
            [[[1], [NaN], [1]], [[0], [0], [0]]], 'int32');
      const b =
          Tensor3D.new([2, 3, 1],
            [[[0], [0], [1]], [[1], [0], [NaN]]], 'int32');
      const res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, boolNaN, 1, 1, 1, boolNaN]);
    });
    it('NaNs in Tensor3D - float32', () => {
      const a = Tensor3D.new(
          [2, 3, 1], [[[1.1], [NaN], [1.1]], [[0.1], [0.1], [0.1]]], 'float32');
      const b = Tensor3D.new(
          [2, 3, 1], [[[0.1], [0.1], [1.1]], [[1.1], [0.1], [NaN]]], 'float32');
      const res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, boolNaN, 1, 1, 1, boolNaN]);
    });

    // Tensor4D:
    it('Tensor4D - int32', () => {
      let a = Tensor4D.new([2, 2, 1, 1], [1, 4, 5, 8], 'int32');
      let b = Tensor4D.new([2, 2, 1, 1], [2, 3, 6, 7], 'int32');
      let res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1, 0]);

      a = Tensor4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
      b = Tensor4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
      res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1]);

      a = Tensor4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'int32');
      b = Tensor4D.new([2, 2, 1, 1], [2, 2, 2, 2], 'int32');
      res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1]);
    });
    it('Tensor4D - float32', () => {
      let a = Tensor4D.new([2, 2, 1, 1], [1.1, 4.1, 5.1, 8.1], 'float32');
      let b = Tensor4D.new([2, 2, 1, 1], [2.1, 3.1, 6.1, 7.1], 'float32');
      let res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1, 0]);

      a = Tensor4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
      b = Tensor4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
      res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1]);

      a = Tensor4D.new([2, 2, 1, 1], [0.1, 0.1, 0.1, 0.1], 'float32');
      b = Tensor4D.new([2, 2, 1, 1], [1.1, 1.1, 1.1, 1.1], 'float32');
      res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1]);
    });
    it('broadcasting Tensor4D shapes - int32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1, 2, 5, 9], 'int32');
      const b = Tensor4D.new(
          [2, 2, 1, 2], [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], 'int32');
      const res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1, 1, 1, 0, 0]);
    });
    it('broadcasting Tensor4D shapes - float32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1.1, 2.1, 5.1, 9.1], 'float32');
      const b = Tensor4D.new(
          [2, 2, 1, 2],
          [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]],
          'float32');
      const res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1, 1, 1, 0, 0]);
    });
    it('NaNs in Tensor4D - int32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1, NaN, 0, 0], 'int32');
      const b = Tensor4D.new([2, 2, 1, 1], [0, 1, 1, NaN], 'int32');
      const res = dl.lessEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, boolNaN, 1, boolNaN]);
    });
    it('NaNs in Tensor4D - float32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1.1, NaN, 0.1, 0.1], 'float32');
      const b = Tensor4D.new([2, 2, 1, 1], [0.1, 1.1, 1.1, NaN], 'float32');
      const res = dl.lessEqual(a, b);

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

// LessEqualStrict:
{
  const tests: MathTests = it => {
    // Tensor1D:
    it('Tensor1D - strict version throws when a and b are different shape',
       () => {
         const a = dl.tensor1d([2]);
         const b = dl.tensor1d([4, 2, -1]);

         expect(() => dl.lessEqualStrict(a, b)).toThrowError();
         expect(() => dl.lessEqualStrict(b, a)).toThrowError();
       });

    // Tensor2D:
    it('Tensor2D - strict version throws when a and b are different shape',
       () => {
         const a = dl.tensor2d([[1.1], [7.1]], [2, 1], 'float32');
         const b =
             Tensor2D.new([2, 3],
              [[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], 'float32');

         expect(() => dl.lessEqualStrict(a, b)).toThrowError();
         expect(() => dl.lessEqualStrict(b, a)).toThrowError();
       });

    // Tensor3D:
    it('Tensor3D - strict version throws when a and b are different shape',
       () => {
         const a = Tensor3D.new(
             [2, 3, 2],
             [
               [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
               [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
             ],
             'float32');
         const b = Tensor3D.new(
             [2, 3, 1], [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]],
             'float32');

         expect(() => dl.lessEqualStrict(a, b)).toThrowError();
         expect(() => dl.lessEqualStrict(b, a)).toThrowError();
       });

    // Tensor4D:
    it('Tensor4D - strict version throws when a and b are different shape',
       () => {
         const a = Tensor4D.new([2, 2, 1, 1], [1.1, 2.1, 5.1, 9.1], 'float32');
         const b = Tensor4D.new(
             [2, 2, 1, 2],
             [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]],
             'float32');

         expect(() => dl.lessEqualStrict(a, b)).toThrowError();
         expect(() => dl.lessEqualStrict(b, a)).toThrowError();
       });
  };

  test_util.describeMathCPU('lessEqualStrict', [tests]);
  test_util.describeMathGPU('lessEqualStrict', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// Greater:
{
  const boolNaN = util.getNaN('bool');

  const tests: MathTests = it => {
    // Tensor1D:
    it('Tensor1D - int32', () => {
      let a = dl.tensor1d([1, 4, 5], 'int32');
      let b = dl.tensor1d([2, 3, 5], 'int32');
      let res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0]);

      a = dl.tensor1d([2, 2, 2], 'int32');
      b = dl.tensor1d([2, 2, 2], 'int32');
      res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0]);

      a = dl.tensor1d([3, 3], 'int32');
      b = dl.tensor1d([0, 0], 'int32');
      res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1]);
    });
    it('Tensor1D - float32', () => {
      let a = dl.tensor1d([1.1, 4.1, 5.1], 'float32');
      let b = dl.tensor1d([2.2, 3.2, 5.1], 'float32');
      let res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0]);

      a = dl.tensor1d([2.31, 2.31, 2.31], 'float32');
      b = dl.tensor1d([2.31, 2.31, 2.31], 'float32');
      res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0]);

      a = dl.tensor1d([3.123, 3.321], 'float32');
      b = dl.tensor1d([0.45, 0.123], 'float32');
      res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1]);
    });
    it('mismatched Tensor1D shapes - int32', () => {
      const a = dl.tensor1d([1, 2], 'int32');
      const b = dl.tensor1d([1, 2, 3], 'int32');
      const f = () => {
        dl.greater(a, b);
      };
      expect(f).toThrowError();
    });
    it('mismatched Tensor1D shapes - float32', () => {
      const a = dl.tensor1d([1.1, 2.1], 'float32');
      const b = dl.tensor1d([1.1, 2.1, 3.1], 'float32');
      const f = () => {
        dl.greater(a, b);
      };
      expect(f).toThrowError();
    });
    it('NaNs in Tensor1D - int32', () => {
      const a = dl.tensor1d([1, NaN, 0], 'int32');
      const b = dl.tensor1d([0, 0, NaN], 'int32');
      const res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, boolNaN, boolNaN]);
    });
    it('NaNs in Tensor1D - float32', () => {
      const a = dl.tensor1d([1.1, NaN, 2.1], 'float32');
      const b = dl.tensor1d([2.1, 3.1, NaN], 'float32');
      const res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, boolNaN, boolNaN]);
    });

    // Tensor2D:
    it('Tensor2D - int32', () => {
      let a = dl.tensor2d([[1, 4, 5], [8, 9, 11]], [2, 3], 'int32');
      let b = dl.tensor2d([[2, 3, 6], [7, 10, 11]], [2, 3], 'int32');
      let res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0, 1, 0, 0]);

      a = dl.tensor2d([[0, 0], [1, 1]], [2, 2], 'int32');
      b = dl.tensor2d([[0, 0], [1, 1]], [2, 2], 'int32');
      res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0]);
    });
    it('Tensor2D - float32', () => {
      let a =
          dl.tensor2d([[1.1, 4.1, 5.1], [8.1, 9.1, 11.1]], [2, 3], 'float32');
      let b =
          dl.tensor2d([[2.1, 3.1, 6.1], [7.1, 10.1, 11.1]], [2, 3], 'float32');
      let res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0, 1, 0, 0]);

      a = dl.tensor2d([[0.2, 0.2], [1.2, 1.2]], [2, 2], 'float32');
      b = dl.tensor2d([[0.2, 0.2], [1.2, 1.2]], [2, 2], 'float32');
      res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0]);
    });
    it('broadcasting Tensor2D shapes - int32', () => {
      const a = dl.tensor2d([[3], [7]], [2, 1], 'int32');
      const b = dl.tensor2d([[2, 3, 4], [7, 8, 9]], [2, 3], 'int32');
      const res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 0, 0, 0, 0]);
    });
    it('broadcasting Tensor2D shapes - float32', () => {
      const a = dl.tensor2d([[1.1], [7.1]], [2, 1], 'float32');
      const b =
          dl.tensor2d([[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], [2, 3], 'float32');
      const res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 0, 0, 0, 0]);
    });
    it('NaNs in Tensor2D - int32', () => {
      const a = dl.tensor2d([[1, NaN, 2], [0, NaN, NaN]], [2, 3], 'int32');
      const b = dl.tensor2d([[0, NaN, NaN], [1, NaN, 3]], [2, 3], 'int32');
      const res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(
          res, [1, boolNaN, boolNaN, 0, boolNaN, boolNaN]);
    });
    it('NaNs in Tensor2D - float32', () => {
      const a = dl.tensor2d([[1.1, NaN], [0.1, NaN]], [2, 2], 'float32');
      const b = dl.tensor2d([[0.1, NaN], [1.1, NaN]], [2, 2], 'float32');
      const res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, boolNaN, 0, boolNaN]);
    });

    // Tensor3D:
    it('Tensor3D - int32', () => {
      let a =
          Tensor3D.new([2, 3, 1], [[[1], [4], [5]], [[8], [9], [11]]], 'int32');
      let b =
          Tensor3D.new([2, 3, 1],
            [[[2], [3], [6]], [[7], [10], [11]]], 'int32');
      let res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0, 1, 0, 0]);

      a = Tensor3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
      b = Tensor3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
      res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0, 0, 0]);
    });
    it('Tensor3D - float32', () => {
      let a = Tensor3D.new(
          [2, 3, 1], [[[1.1], [4.1], [5.1]], [[8.1], [9.1], [11.1]]],
          'float32');
      let b = Tensor3D.new(
          [2, 3, 1], [[[2.1], [3.1], [6.1]], [[7.1], [10.1], [11.1]]],
          'float32');
      let res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0, 1, 0, 0]);

      a = Tensor3D.new(
          [2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.2]]], 'float32');
      b = Tensor3D.new(
          [2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
      res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0, 0, 1]);
    });
    it('broadcasting Tensor3D shapes - int32', () => {
      const a = Tensor3D.new(
          [2, 3, 2], [[[1, 0], [2, 3], [4, 5]], [[6, 7], [9, 8], [10, 11]]],
          'int32');
      const b =
          Tensor3D.new([2, 3, 1], [[[1], [2], [3]], [[7], [10], [9]]], 'int32');
      const res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]);
    });
    it('broadcasting Tensor3D float32', () => {
      const a = Tensor3D.new(
          [2, 3, 2],
          [
            [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
            [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
          ],
          'float32');
      const b = Tensor3D.new(
          [2, 3, 1], [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]],
          'float32');
      const res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]);
    });
    it('NaNs in Tensor3D - int32', () => {
      const a =
          Tensor3D.new([2, 3, 1],
            [[[1], [NaN], [1]], [[0], [0], [0]]], 'int32');
      const b =
          Tensor3D.new([2, 3, 1],
            [[[0], [0], [1]], [[1], [0], [NaN]]], 'int32');
      const res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, boolNaN, 0, 0, 0, boolNaN]);
    });
    it('NaNs in Tensor3D - float32', () => {
      const a = Tensor3D.new(
          [2, 3, 1], [[[1.1], [NaN], [1.1]], [[0.1], [0.1], [0.1]]], 'float32');
      const b = Tensor3D.new(
          [2, 3, 1], [[[0.1], [0.1], [1.1]], [[1.1], [0.1], [NaN]]], 'float32');
      const res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, boolNaN, 0, 0, 0, boolNaN]);
    });

    // Tensor4D:
    it('Tensor4D - int32', () => {
      let a = Tensor4D.new([2, 2, 1, 1], [1, 4, 5, 8], 'int32');
      let b = Tensor4D.new([2, 2, 1, 1], [2, 3, 6, 8], 'int32');
      let res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0, 0]);

      a = Tensor4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
      b = Tensor4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
      res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0]);

      a = Tensor4D.new([2, 2, 1, 1], [2, 2, 2, 2], 'int32');
      b = Tensor4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'int32');
      res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1]);
    });
    it('Tensor4D - float32', () => {
      let a = Tensor4D.new([2, 2, 1, 1], [1.1, 4.1, 5.1, 8.1], 'float32');
      let b = Tensor4D.new([2, 2, 1, 1], [2.1, 3.1, 6.1, 8.1], 'float32');
      let res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0, 0]);

      a = Tensor4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
      b = Tensor4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
      res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0]);

      a = Tensor4D.new([2, 2, 1, 1], [1.1, 1.1, 1.1, 1.1], 'float32');
      b = Tensor4D.new([2, 2, 1, 1], [0.1, 0.1, 0.1, 0.1], 'float32');
      res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1]);
    });
    it('broadcasting Tensor4D shapes - int32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1, 2, 5, 9], 'int32');
      const b = Tensor4D.new(
          [2, 2, 1, 2], [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], 'int32');
      const res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0, 0, 0, 1, 1]);
    });
    it('broadcasting Tensor4D shapes - float32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1.1, 2.1, 5.1, 9.1], 'float32');
      const b = Tensor4D.new(
          [2, 2, 1, 2],
          [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]],
          'float32');
      const res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0, 0, 0, 1, 1]);
    });
    it('NaNs in Tensor4D - int32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1, NaN, 0, 0], 'int32');
      const b = Tensor4D.new([2, 2, 1, 1], [0, 1, 1, NaN], 'int32');
      const res = dl.greater(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, boolNaN, 0, boolNaN]);
    });
    it('NaNs in Tensor4D - float32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1.1, NaN, 0.1, 0.1], 'float32');
      const b = Tensor4D.new([2, 2, 1, 1], [0.1, 1.1, 1.1, NaN], 'float32');
      const res = dl.greater(a, b);

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

// GreaterStrict:
{
  const tests: MathTests = it => {
    // Tensor1D:
    it('Tensor1D - strict version throws when a and b are different shape',
       () => {
         const a = dl.tensor1d([2]);
         const b = dl.tensor1d([4, 2, -1]);

         expect(() => dl.greaterStrict(a, b)).toThrowError();
         expect(() => dl.greaterStrict(b, a)).toThrowError();
       });

    // Tensor2D:
    it('Tensor2D - strict version throws when a and b are different shape',
       () => {
         const a = dl.tensor2d([[1.1], [7.1]], [2, 1], 'float32');
         const b =
             Tensor2D.new([2, 3],
              [[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], 'float32');

         expect(() => dl.greaterStrict(a, b)).toThrowError();
         expect(() => dl.greaterStrict(b, a)).toThrowError();
       });

    // Tensor3D:
    it('Tensor3D - strict version throws when a and b are different shape',
       () => {
         const a = Tensor3D.new(
             [2, 3, 2],
             [
               [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
               [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
             ],
             'float32');
         const b = Tensor3D.new(
             [2, 3, 1], [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]],
             'float32');

         expect(() => dl.greaterStrict(a, b)).toThrowError();
         expect(() => dl.greaterStrict(b, a)).toThrowError();
       });

    // Tensor4D:
    it('Tensor4D - strict version throws when a and b are different shape',
       () => {
         const a = Tensor4D.new([2, 2, 1, 1], [1.1, 2.1, 5.1, 9.1], 'float32');
         const b = Tensor4D.new(
             [2, 2, 1, 2],
             [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]],
             'float32');

         expect(() => dl.greaterStrict(a, b)).toThrowError();
         expect(() => dl.greaterStrict(b, a)).toThrowError();
       });
  };

  test_util.describeMathCPU('greaterStrict', [tests]);
  test_util.describeMathGPU('greaterStrict', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// GreaterEqual:
{
  const boolNaN = util.getNaN('bool');

  const tests: MathTests = it => {
    // Tensor1D:
    it('Tensor1D - int32', () => {
      let a = dl.tensor1d([1, 4, 5], 'int32');
      let b = dl.tensor1d([2, 3, 5], 'int32');
      let res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 1]);

      a = dl.tensor1d([2, 2, 2], 'int32');
      b = dl.tensor1d([2, 2, 2], 'int32');
      res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1]);

      a = dl.tensor1d([0, 0], 'int32');
      b = dl.tensor1d([3, 3], 'int32');
      res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0]);
    });
    it('Tensor1D - float32', () => {
      let a = dl.tensor1d([1.1, 4.1, 5.1], 'float32');
      let b = dl.tensor1d([2.2, 3.2, 5.1], 'float32');
      let res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 1]);

      a = dl.tensor1d([2.31, 2.31, 2.31], 'float32');
      b = dl.tensor1d([2.31, 2.31, 2.31], 'float32');
      res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1]);

      a = dl.tensor1d([0.45, 0.123], 'float32');
      b = dl.tensor1d([3.123, 3.321], 'float32');
      res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0]);
    });
    it('mismatched Tensor1D shapes - int32', () => {
      const a = dl.tensor1d([1, 2], 'int32');
      const b = dl.tensor1d([1, 2, 3], 'int32');
      const f = () => {
        dl.greaterEqual(a, b);
      };
      expect(f).toThrowError();
    });
    it('mismatched Tensor1D shapes - float32', () => {
      const a = dl.tensor1d([1.1, 2.1], 'float32');
      const b = dl.tensor1d([1.1, 2.1, 3.1], 'float32');
      const f = () => {
        dl.greaterEqual(a, b);
      };
      expect(f).toThrowError();
    });
    it('NaNs in Tensor1D - int32', () => {
      const a = dl.tensor1d([1, NaN, 0], 'int32');
      const b = dl.tensor1d([0, 0, NaN], 'int32');
      const res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, boolNaN, boolNaN]);
    });
    it('NaNs in Tensor1D - float32', () => {
      const a = dl.tensor1d([1.1, NaN, 2.1], 'float32');
      const b = dl.tensor1d([2.1, 3.1, NaN], 'float32');
      const res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, boolNaN, boolNaN]);
    });

    // Tensor2D:
    it('Tensor2D - int32', () => {
      let a = dl.tensor2d([[1, 4, 5], [8, 9, 12]], [2, 3], 'int32');
      let b = dl.tensor2d([[2, 3, 6], [7, 10, 11]], [2, 3], 'int32');
      let res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0, 1, 0, 1]);

      a = dl.tensor2d([[0, 0], [1, 1]], [2, 2], 'int32');
      b = dl.tensor2d([[0, 0], [1, 1]], [2, 2], 'int32');
      res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1]);
    });
    it('Tensor2D - float32', () => {
      let a =
          dl.tensor2d([[1.1, 4.1, 5.1], [8.1, 9.1, 12.1]], [2, 3], 'float32');
      let b =
          dl.tensor2d([[2.1, 3.1, 6.1], [7.1, 10.1, 11.1]], [2, 3], 'float32');
      let res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0, 1, 0, 1]);

      a = dl.tensor2d([[0.2, 0.2], [1.2, 1.2]], [2, 2], 'float32');
      b = dl.tensor2d([[0.2, 0.2], [1.2, 1.2]], [2, 2], 'float32');
      res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1]);
    });
    it('broadcasting Tensor2D shapes - int32', () => {
      const a = dl.tensor2d([[3], [7]], [2, 1], 'int32');
      const b = dl.tensor2d([[2, 3, 4], [7, 8, 9]], [2, 3], 'int32');
      const res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 0, 1, 0, 0]);
    });
    it('broadcasting Tensor2D shapes - float32', () => {
      const a = dl.tensor2d([[1.1], [7.1]], [2, 1], 'float32');
      const b =
          dl.tensor2d([[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], [2, 3], 'float32');
      const res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 0, 1, 0, 0]);
    });
    it('NaNs in Tensor2D - int32', () => {
      const a = dl.tensor2d([[1, NaN, 2], [0, NaN, NaN]], [2, 3], 'int32');
      const b = dl.tensor2d([[0, NaN, NaN], [1, NaN, 3]], [2, 3], 'int32');
      const res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(
          res, [1, boolNaN, boolNaN, 0, boolNaN, boolNaN]);
    });
    it('NaNs in Tensor2D - float32', () => {
      const a = dl.tensor2d([[1.1, NaN], [0.1, NaN]], [2, 2], 'float32');
      const b = dl.tensor2d([[0.1, NaN], [1.1, NaN]], [2, 2], 'float32');
      const res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, boolNaN, 0, boolNaN]);
    });

    // Tensor3D:
    it('Tensor3D - int32', () => {
      let a =
          Tensor3D.new([2, 3, 1], [[[1], [4], [5]], [[8], [9], [12]]], 'int32');
      let b =
          Tensor3D.new([2, 3, 1],
            [[[2], [3], [6]], [[7], [10], [11]]], 'int32');
      let res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0, 1, 0, 1]);

      a = Tensor3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
      b = Tensor3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
      res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1, 1, 1]);
    });
    it('Tensor3D - float32', () => {
      let a = Tensor3D.new(
          [2, 3, 1], [[[1.1], [4.1], [5.1]], [[8.1], [9.1], [12.1]]],
          'float32');
      let b = Tensor3D.new(
          [2, 3, 1], [[[2.1], [3.1], [6.1]], [[7.1], [10.1], [11.1]]],
          'float32');
      let res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0, 1, 0, 1]);

      a = Tensor3D.new(
          [2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.2]]], 'float32');
      b = Tensor3D.new(
          [2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
      res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1, 1, 1]);
    });
    it('broadcasting Tensor3D shapes - int32', () => {
      const a = Tensor3D.new(
          [2, 3, 2], [[[1, 0], [2, 3], [4, 5]], [[6, 7], [9, 8], [10, 11]]],
          'int32');
      const b =
          Tensor3D.new([2, 3, 1], [[[1], [2], [3]], [[7], [10], [9]]], 'int32');
      const res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1]);
    });
    it('broadcasting Tensor3D float32', () => {
      const a = Tensor3D.new(
          [2, 3, 2],
          [
            [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
            [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
          ],
          'float32');
      const b = Tensor3D.new(
          [2, 3, 1], [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]],
          'float32');
      const res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1]);
    });
    it('NaNs in Tensor3D - int32', () => {
      const a =
          Tensor3D.new([2, 3, 1],
            [[[1], [NaN], [1]], [[0], [0], [0]]], 'int32');
      const b =
          Tensor3D.new([2, 3, 1],
            [[[0], [0], [1]], [[1], [0], [NaN]]], 'int32');
      const res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, boolNaN, 1, 0, 1, boolNaN]);
    });
    it('NaNs in Tensor3D - float32', () => {
      const a = Tensor3D.new(
          [2, 3, 1], [[[1.1], [NaN], [1.1]], [[0.1], [0.1], [0.1]]], 'float32');
      const b = Tensor3D.new(
          [2, 3, 1], [[[0.1], [0.1], [1.1]], [[1.1], [0.1], [NaN]]], 'float32');
      const res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, boolNaN, 1, 0, 1, boolNaN]);
    });

    // Tensor4D:
    it('Tensor4D - int32', () => {
      let a = Tensor4D.new([2, 2, 1, 1], [1, 4, 5, 8], 'int32');
      let b = Tensor4D.new([2, 2, 1, 1], [2, 3, 6, 7], 'int32');
      let res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0, 1]);

      a = Tensor4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
      b = Tensor4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
      res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1]);

      a = Tensor4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'int32');
      b = Tensor4D.new([2, 2, 1, 1], [2, 2, 2, 2], 'int32');
      res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0]);
    });
    it('Tensor4D - float32', () => {
      let a = Tensor4D.new([2, 2, 1, 1], [1.1, 4.1, 5.1, 8.1], 'float32');
      let b = Tensor4D.new([2, 2, 1, 1], [2.1, 3.1, 6.1, 7.1], 'float32');
      let res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 1, 0, 1]);

      a = Tensor4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
      b = Tensor4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
      res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 1, 1, 1]);

      a = Tensor4D.new([2, 2, 1, 1], [0.1, 0.1, 0.1, 0.1], 'float32');
      b = Tensor4D.new([2, 2, 1, 1], [1.1, 1.1, 1.1, 1.1], 'float32');
      res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [0, 0, 0, 0]);
    });
    it('broadcasting Tensor4D shapes - int32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1, 2, 5, 9], 'int32');
      const b = Tensor4D.new(
          [2, 2, 1, 2], [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], 'int32');
      const res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 0, 0, 1, 0, 1, 1]);
    });
    it('broadcasting Tensor4D shapes - float32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1.1, 2.1, 5.1, 9.1], 'float32');
      const b = Tensor4D.new(
          [2, 2, 1, 2],
          [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]],
          'float32');
      const res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, 0, 0, 0, 1, 0, 1, 1]);
    });
    it('NaNs in Tensor4D - int32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1, NaN, 0, 0], 'int32');
      const b = Tensor4D.new([2, 2, 1, 1], [0, 1, 1, NaN], 'int32');
      const res = dl.greaterEqual(a, b);

      expect(res.dtype).toBe('bool');
      test_util.expectArraysClose(res, [1, boolNaN, 0, boolNaN]);
    });
    it('NaNs in Tensor4D - float32', () => {
      const a = Tensor4D.new([2, 2, 1, 1], [1.1, NaN, 0.1, 0.1], 'float32');
      const b = Tensor4D.new([2, 2, 1, 1], [0.1, 1.1, 1.1, NaN], 'float32');
      const res = dl.greaterEqual(a, b);

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

// GreaterEqualStrict:
{
  const tests: MathTests = it => {
    // Tensor1D:
    it('Tensor1D - strict version throws when a and b are different shape',
       () => {
         const a = dl.tensor1d([2]);
         const b = dl.tensor1d([4, 2, -1]);

         expect(() => dl.greaterEqualStrict(a, b)).toThrowError();
         expect(() => dl.greaterEqualStrict(b, a)).toThrowError();
       });

    // Tensor2D:
    it('Tensor2D - strict version throws when a and b are different shape',
       () => {
         const a = dl.tensor2d([[1.1], [7.1]], [2, 1], 'float32');
         const b =
             Tensor2D.new([2, 3],
              [[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], 'float32');

         expect(() => dl.greaterEqualStrict(a, b)).toThrowError();
         expect(() => dl.greaterEqualStrict(b, a)).toThrowError();
       });

    // Tensor3D:
    it('Tensor3D - strict version throws when a and b are different shape',
       () => {
         const a = Tensor3D.new(
             [2, 3, 2],
             [
               [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
               [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
             ],
             'float32');
         const b = Tensor3D.new(
             [2, 3, 1], [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]],
             'float32');

         expect(() => dl.greaterEqualStrict(a, b)).toThrowError();
         expect(() => dl.greaterEqualStrict(b, a)).toThrowError();
       });

    // Tensor4D:
    it('Tensor4D - strict version throws when a and b are different shape',
       () => {
         const a = Tensor4D.new([2, 2, 1, 1], [1.1, 2.1, 5.1, 9.1], 'float32');
         const b = Tensor4D.new(
             [2, 2, 1, 2],
             [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]],
             'float32');

         expect(() => dl.greaterEqualStrict(a, b)).toThrowError();
         expect(() => dl.greaterEqualStrict(b, a)).toThrowError();
       });
  };

  test_util.describeMathCPU('greaterEqualStrict', [tests]);
  test_util.describeMathGPU('greaterEqualStrict', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
