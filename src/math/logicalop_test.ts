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
import * as dl from '../index';

// LogicalNot:
{
  const boolNaN = util.getNaN('bool');

  const tests: MathTests = it => {
    it('Tensor1D.', math => {
      let a = dl.tensor1d([1, 0, 0], 'bool');
      test_util.expectArraysClose(math.logicalNot(a), [0, 1, 1]);

      a = dl.tensor1d([0, 0, 0], 'bool');
      test_util.expectArraysClose(math.logicalNot(a), [1, 1, 1]);

      a = dl.tensor1d([1, 1], 'bool');
      test_util.expectArraysClose(math.logicalNot(a), [0, 0]);
    });
    it('NaNs in Tensor1D', math => {
      const a = dl.tensor1d([1, NaN, 0], 'bool');
      test_util.expectArraysClose(math.logicalNot(a), [0, boolNaN, 1]);
    });

    it('Tensor2D', math => {
      let a = dl.tensor2d([[1, 0, 1], [0, 0, 0]], [2, 3], 'bool');
      test_util.expectArraysClose(math.logicalNot(a), [0, 1, 0, 1, 1, 1]);

      a = dl.tensor2d([[0, 0, 0], [1, 1, 1]], [2, 3], 'bool');
      test_util.expectArraysClose(math.logicalNot(a), [1, 1, 1, 0, 0, 0]);
    });
    it('NaNs in Tensor2D', math => {
      const a = dl.tensor2d([[1, NaN], [0, NaN]], [2, 2], 'bool');
      test_util.expectArraysClose(math.logicalNot(a), [0, boolNaN, 1, boolNaN]);
    });

    it('Tensor3D', math => {
      let a =
          dl.tensor3d([[[1], [0], [1]], [[0], [0], [0]]], [2, 3, 1], 'bool');
      test_util.expectArraysClose(math.logicalNot(a), [0, 1, 0, 1, 1, 1]);

      a = dl.tensor3d([[[0], [0], [0]], [[1], [1], [1]]], [2, 3, 1], 'bool');
      test_util.expectArraysClose(math.logicalNot(a), [1, 1, 1, 0, 0, 0]);
    });
    it('NaNs in Tensor3D', math => {
      const a =
          dl.tensor3d([[[1], [NaN], [1]], [[0], [0], [0]]], [2, 3, 1], 'bool');
      test_util.expectArraysClose(math.logicalNot(a), [0, boolNaN, 0, 1, 1, 1]);
    });

    it('Tensor4D', math => {
      let a = dl.tensor4d([1, 0, 1, 0], [2, 2, 1, 1], 'bool');
      test_util.expectArraysClose(math.logicalNot(a), [0, 1, 0, 1]);

      a = dl.tensor4d([0, 0, 0, 0], [2, 2, 1, 1], 'bool');
      test_util.expectArraysClose(math.logicalNot(a), [1, 1, 1, 1]);

      a = dl.tensor4d([1, 1, 1, 1], [2, 2, 1, 1], 'bool');
      test_util.expectArraysClose(math.logicalNot(a), [0, 0, 0, 0]);
    });
    it('NaNs in Tensor4D', math => {
      const a = dl.tensor4d([1, NaN, 1, 0], [2, 2, 1, 1], 'bool');
      test_util.expectArraysClose(math.logicalNot(a), [0, boolNaN, 0, 1]);
    });
  };

  test_util.describeMathCPU('logicalNot', [tests]);
  test_util.describeMathGPU('logicalNot', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// LogicalAnd:
{
  const boolNaN = util.getNaN('bool');

  const tests: MathTests = it => {
    it('Tensor1D.', math => {
      let a = dl.tensor1d([1, 0, 0], 'bool');
      let b = dl.tensor1d([0, 1, 0], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 0]);

      a = dl.tensor1d([0, 0, 0], 'bool');
      b = dl.tensor1d([0, 0, 0], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 0]);

      a = dl.tensor1d([1, 1], 'bool');
      b = dl.tensor1d([1, 1], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [1, 1]);
    });
    it('mismatched Tensor1D shapes', math => {
      const a = dl.tensor1d([1, 0], 'bool');
      const b = dl.tensor1d([0, 1, 0], 'bool');
      const f = () => {
        math.logicalAnd(a, b);
      };
      expect(f).toThrowError();
    });
    it('NaNs in Tensor1D', math => {
      const a = dl.tensor1d([1, NaN, 0], 'bool');
      const b = dl.tensor1d([0, 0, NaN], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [0, boolNaN, boolNaN]);
    });

    it('Tensor2D', math => {
      let a = dl.tensor2d([[1, 0, 1], [0, 0, 0]], [2, 3], 'bool');
      let b = dl.tensor2d([[0, 0, 0], [0, 1, 0]], [2, 3], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 0, 0, 0, 0]);

      a = dl.tensor2d([[0, 0, 0], [1, 1, 1]], [2, 3], 'bool');
      b = dl.tensor2d([[0, 0, 0], [1, 1, 1]], [2, 3], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 0, 1, 1, 1]);
    });
    it('broadcasting Tensor2D shapes', math => {
      const a = dl.tensor2d([[1], [0]], [2, 1], 'bool');
      const b = dl.tensor2d([[0, 1, 0], [0, 1, 0]], [2, 3], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [0, 1, 0, 0, 0, 0]);
    });
    it('NaNs in Tensor2D', math => {
      const a = dl.tensor2d([[1, NaN], [0, NaN]], [2, 2], 'bool');
      const b = dl.tensor2d([[0, NaN], [1, NaN]], [2, 2], 'bool');
      test_util.expectArraysClose(
          math.logicalAnd(a, b), [0, boolNaN, 0, boolNaN]);
    });

    it('Tensor3D', math => {
      let a =
          dl.tensor3d([[[1], [0], [1]], [[0], [0], [1]]], [2, 3, 1], 'bool');
      let b =
          dl.tensor3d([[[0], [0], [1]], [[1], [0], [0]]], [2, 3, 1], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 1, 0, 0, 0]);

      a = dl.tensor3d([[[0], [0], [0]], [[1], [1], [1]]], [2, 3, 1], 'bool');
      b = dl.tensor3d([[[0], [0], [0]], [[1], [1], [1]]], [2, 3, 1], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 0, 1, 1, 1]);
    });
    it('broadcasting Tensor3D shapes', math => {
      const a = dl.tensor3d(
          [[[1, 0], [0, 0], [1, 1]], [[0, 0], [0, 1], [0, 0]]],
          [2, 3, 2],
          'bool');
      const b =
          dl.tensor3d([[[0], [0], [1]], [[1], [0], [0]]], [2, 3, 1], 'bool');
      test_util.expectArraysClose(
          math.logicalAnd(a, b), [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]);
    });
    it('NaNs in Tensor3D', math => {
      const a =
          dl.tensor3d([[[1], [NaN], [1]], [[0], [0], [0]]], [2, 3, 1], 'bool');
      const b =
          dl.tensor3d([[[0], [0], [1]], [[1], [0], [NaN]]], [2, 3, 1], 'bool');
      test_util.expectArraysClose(
          math.logicalAnd(a, b), [0, boolNaN, 1, 0, 0, boolNaN]);
    });

    it('Tensor4D', math => {
      let a = dl.tensor4d([1, 0, 1, 0], [2, 2, 1, 1], 'bool');
      let b = dl.tensor4d([0, 1, 1, 0], [2, 2, 1, 1], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 1, 0]);

      a = dl.tensor4d([0, 0, 0, 0], [2, 2, 1, 1], 'bool');
      b = dl.tensor4d([0, 0, 0, 0], [2, 2, 1, 1], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 0, 0]);

      a = dl.tensor4d([1, 1, 1, 1], [2, 2, 1, 1], 'bool');
      b = dl.tensor4d([1, 1, 1, 1], [2, 2, 1, 1], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [1, 1, 1, 1]);
    });
    it('broadcasting Tensor4D shapes', math => {
      const a = dl.tensor4d([1, 0, 1, 0], [2, 2, 1, 1], 'bool');
      const b = dl.tensor4d(
          [[[[1, 0]], [[0, 0]]], [[[0, 0]], [[1, 1]]]], [2, 2, 1, 2], 'bool');
      test_util.expectArraysClose(
          math.logicalAnd(a, b), [1, 0, 0, 0, 0, 0, 0, 0]);
    });
    it('NaNs in Tensor4D', math => {
      const a = dl.tensor4d([1, NaN, 1, 0], [2, 2, 1, 1], 'bool');
      const b = dl.tensor4d([0, 1, 0, NaN], [2, 2, 1, 1], 'bool');
      test_util.expectArraysClose(
          math.logicalAnd(a, b), [0, boolNaN, 0, boolNaN]);
    });
  };

  test_util.describeMathCPU('logicalAnd', [tests]);
  test_util.describeMathGPU('logicalAnd', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// LogicalOr:
{
  const boolNaN = util.getNaN('bool');

  const tests: MathTests = it => {
    it('Tensor1D.', math => {
      let a = dl.tensor1d([1, 0, 0], 'bool');
      let b = dl.tensor1d([0, 1, 0], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [1, 1, 0]);

      a = dl.tensor1d([0, 0, 0], 'bool');
      b = dl.tensor1d([0, 0, 0], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [0, 0, 0]);

      a = dl.tensor1d([1, 1], 'bool');
      b = dl.tensor1d([1, 1], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [1, 1]);
    });
    it('mismatched Tensor1D shapes', math => {
      const a = dl.tensor1d([1, 0], 'bool');
      const b = dl.tensor1d([0, 1, 0], 'bool');
      const f = () => {
        math.logicalOr(a, b);
      };
      expect(f).toThrowError();
    });
    it('NaNs in Tensor1D', math => {
      const a = dl.tensor1d([1, NaN, 0], 'bool');
      const b = dl.tensor1d([0, 0, NaN], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [1, boolNaN, boolNaN]);
    });

    it('Tensor2D', math => {
      let a = dl.tensor2d([[1, 0, 1], [0, 0, 0]], [2, 3], 'bool');
      let b = dl.tensor2d([[0, 0, 0], [0, 1, 0]], [2, 3], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [1, 0, 1, 0, 1, 0]);

      a = dl.tensor2d([[0, 0, 0], [1, 1, 1]], [2, 3], 'bool');
      b = dl.tensor2d([[0, 0, 0], [1, 1, 1]], [2, 3], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [0, 0, 0, 1, 1, 1]);
    });
    it('broadcasting Tensor2D shapes', math => {
      const a = dl.tensor2d([[1], [0]], [2, 1], 'bool');
      const b = dl.tensor2d([[0, 0, 0], [0, 1, 0]], [2, 3], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [1, 1, 1, 0, 1, 0]);
    });
    it('NaNs in Tensor2D', math => {
      const a = dl.tensor2d([[1, NaN], [0, NaN]], [2, 2], 'bool');
      const b = dl.tensor2d([[0, NaN], [1, NaN]], [2, 2], 'bool');
      test_util.expectArraysClose(
          math.logicalOr(a, b), [1, boolNaN, 1, boolNaN]);
    });

    it('Tensor3D', math => {
      let a =
          dl.tensor3d([[[1], [0], [1]], [[0], [0], [0]]], [2, 3, 1], 'bool');
      let b =
          dl.tensor3d([[[0], [0], [1]], [[1], [0], [0]]], [2, 3, 1], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [1, 0, 1, 1, 0, 0]);

      a = dl.tensor3d([[[0], [0], [0]], [[1], [1], [1]]], [2, 3, 1], 'bool');
      b = dl.tensor3d([[[0], [0], [0]], [[1], [1], [1]]], [2, 3, 1], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [0, 0, 0, 1, 1, 1]);
    });
    it('broadcasting Tensor3D shapes', math => {
      const a = dl.tensor3d(
          [[[1, 0], [0, 0], [1, 1]], [[0, 0], [0, 1], [0, 0]]],
          [2, 3, 2],
          'bool');
      const b =
          dl.tensor3d([[[0], [0], [1]], [[1], [0], [0]]], [2, 3, 1], 'bool');
      test_util.expectArraysClose(
          math.logicalOr(a, b), [1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0]);
    });
    it('NaNs in Tensor3D', math => {
      const a =
          dl.tensor3d([[[1], [NaN], [1]], [[0], [0], [0]]], [2, 3, 1], 'bool');
      const b =
          dl.tensor3d([[[0], [0], [1]], [[1], [0], [NaN]]], [2, 3, 1], 'bool');
      test_util.expectArraysClose(
          math.logicalOr(a, b), [1, boolNaN, 1, 1, 0, boolNaN]);
    });

    it('Tensor4D', math => {
      let a = dl.tensor4d([1, 0, 1, 0], [2, 2, 1, 1], 'bool');
      let b = dl.tensor4d([0, 1, 0, 0], [2, 2, 1, 1], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [1, 1, 1, 0]);

      a = dl.tensor4d([0, 0, 0, 0], [2, 2, 1, 1], 'bool');
      b = dl.tensor4d([0, 0, 0, 0], [2, 2, 1, 1], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [0, 0, 0, 0]);

      a = dl.tensor4d([1, 1, 1, 1], [2, 2, 1, 1], 'bool');
      b = dl.tensor4d([1, 1, 1, 1], [2, 2, 1, 1], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [1, 1, 1, 1]);
    });
    it('broadcasting Tensor4D shapes', math => {
      const a = dl.tensor4d([1, 0, 1, 0], [2, 2, 1, 1], 'bool');
      const b = dl.tensor4d(
          [[[[1, 0]], [[0, 0]]], [[[0, 0]], [[1, 1]]]], [2, 2, 1, 2], 'bool');
      test_util.expectArraysClose(
          math.logicalOr(a, b), [1, 1, 0, 0, 1, 1, 1, 1]);
    });
    it('NaNs in Tensor4D', math => {
      const a = dl.tensor4d([1, NaN, 1, 0], [2, 2, 1, 1], 'bool');
      const b = dl.tensor4d([0, 1, 0, NaN], [2, 2, 1, 1], 'bool');
      test_util.expectArraysClose(
          math.logicalOr(a, b), [1, boolNaN, 1, boolNaN]);
    });
  };

  test_util.describeMathCPU('logicalOr', [tests]);
  test_util.describeMathGPU('logicalOr', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// LogicalXor:
{
  const boolNaN = util.getNaN('bool');

  const tests: MathTests = it => {
    // Tensor1D:
    it('Tensor1D.', math => {
      let a = dl.tensor1d([1, 0, 0], 'bool');
      let b = dl.tensor1d([0, 1, 0], 'bool');
      test_util.expectArraysClose(math.logicalXor(a, b), [1, 1, 0]);

      a = dl.tensor1d([0, 0, 0], 'bool');
      b = dl.tensor1d([0, 0, 0], 'bool');
      test_util.expectArraysClose(math.logicalXor(a, b), [0, 0, 0]);

      a = dl.tensor1d([1, 1], 'bool');
      b = dl.tensor1d([1, 1], 'bool');
      test_util.expectArraysClose(math.logicalXor(a, b), [0, 0]);
    });
    it('mismatched Tensor1D shapes', math => {
      const a = dl.tensor1d([1, 0], 'bool');
      const b = dl.tensor1d([0, 1, 0], 'bool');
      const f = () => {
        math.logicalXor(a, b);
      };
      expect(f).toThrowError();
    });
    it('NaNs in Tensor1D', math => {
      const a = dl.tensor1d([1, NaN, 0], 'bool');
      const b = dl.tensor1d([0, 0, NaN], 'bool');
      test_util.expectArraysClose(math.logicalXor(a, b), [1, boolNaN, boolNaN]);
    });

    // Tensor2D:
    it('Tensor2D', math => {
      let a = dl.tensor2d([[1, 0, 1], [0, 0, 0]], [2, 3], 'bool');
      let b = dl.tensor2d([[0, 0, 0], [0, 1, 0]], [2, 3], 'bool');
      test_util.expectArraysClose(math.logicalXor(a, b), [1, 0, 1, 0, 1, 0]);

      a = dl.tensor2d([[0, 0, 0], [1, 1, 1]], [2, 3], 'bool');
      b = dl.tensor2d([[0, 0, 0], [1, 1, 1]], [2, 3], 'bool');
      test_util.expectArraysClose(math.logicalXor(a, b), [0, 0, 0, 0, 0, 0]);
    });
    it('broadcasting Tensor2D shapes', math => {
      const a = dl.tensor2d([[1], [0]], [2, 1], 'bool');
      const b = dl.tensor2d([[0, 0, 0], [0, 1, 0]], [2, 3], 'bool');
      test_util.expectArraysClose(math.logicalXor(a, b), [1, 1, 1, 0, 1, 0]);
    });
    it('NaNs in Tensor2D', math => {
      const a = dl.tensor2d([[1, NaN], [0, NaN]], [2, 2], 'bool');
      const b = dl.tensor2d([[0, NaN], [1, NaN]], [2, 2], 'bool');
      test_util.expectArraysClose(
          math.logicalXor(a, b), [1, boolNaN, 1, boolNaN]);
    });

    // Tensor3D:
    it('Tensor3D', math => {
      let a =
          dl.tensor3d([[[1], [0], [1]], [[0], [0], [0]]], [2, 3, 1], 'bool');
      let b =
          dl.tensor3d([[[0], [0], [1]], [[1], [0], [0]]], [2, 3, 1], 'bool');
      test_util.expectArraysClose(math.logicalXor(a, b), [1, 0, 0, 1, 0, 0]);

      a = dl.tensor3d([[[0], [0], [0]], [[1], [1], [1]]], [2, 3, 1], 'bool');
      b = dl.tensor3d([[[0], [0], [0]], [[1], [1], [1]]], [2, 3, 1], 'bool');
      test_util.expectArraysClose(math.logicalXor(a, b), [0, 0, 0, 0, 0, 0]);
    });
    it('broadcasting Tensor3D shapes', math => {
      const a = dl.tensor3d(
          [[[1, 0], [0, 0], [1, 1]], [[0, 0], [0, 1], [0, 0]]],
          [2, 3, 2],
          'bool');
      const b =
          dl.tensor3d([[[0], [0], [1]], [[1], [0], [0]]], [2, 3, 1], 'bool');
      test_util.expectArraysClose(
          math.logicalXor(a, b), [1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]);
    });
    it('NaNs in Tensor3D', math => {
      const a =
          dl.tensor3d([[[1], [NaN], [1]], [[0], [0], [0]]], [2, 3, 1], 'bool');
      const b =
          dl.tensor3d([[[0], [0], [1]], [[1], [0], [NaN]]], [2, 3, 1], 'bool');
      test_util.expectArraysClose(
          math.logicalXor(a, b), [1, boolNaN, 0, 1, 0, boolNaN]);
    });

    // Tensor4D:
    it('Tensor4D', math => {
      let a = dl.tensor4d([1, 0, 1, 0], [2, 2, 1, 1], 'bool');
      let b = dl.tensor4d([0, 1, 1, 0], [2, 2, 1, 1], 'bool');
      test_util.expectArraysClose(math.logicalXor(a, b), [1, 1, 0, 0]);

      a = dl.tensor4d([0, 0, 0, 0], [2, 2, 1, 1], 'bool');
      b = dl.tensor4d([0, 0, 0, 0], [2, 2, 1, 1], 'bool');
      test_util.expectArraysClose(math.logicalXor(a, b), [0, 0, 0, 0]);

      a = dl.tensor4d([1, 1, 1, 1], [2, 2, 1, 1], 'bool');
      b = dl.tensor4d([1, 1, 1, 1], [2, 2, 1, 1], 'bool');
      test_util.expectArraysClose(math.logicalXor(a, b), [0, 0, 0, 0]);
    });
    it('broadcasting Tensor4D shapes', math => {
      const a = dl.tensor4d([1, 0, 1, 0], [2, 2, 1, 1], 'bool');
      const b = dl.tensor4d(
          [[[[1, 0]], [[0, 0]]], [[[0, 0]], [[1, 1]]]], [2, 2, 1, 2], 'bool');
      test_util.expectArraysClose(
          math.logicalXor(a, b), [0, 1, 0, 0, 1, 1, 1, 1]);
    });
    it('NaNs in Tensor4D', math => {
      const a = dl.tensor4d([1, NaN, 1, 0], [2, 2, 1, 1], 'bool');
      const b = dl.tensor4d([0, 1, 0, NaN], [2, 2, 1, 1], 'bool');
      test_util.expectArraysClose(
          math.logicalXor(a, b), [1, boolNaN, 1, boolNaN]);
    });
  };

  test_util.describeMathCPU('logicalXor', [tests]);
  test_util.describeMathGPU('logicalXor', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// Where
{
  const tests: MathTests = it => {
    it('Tensor1D', math => {
      const c = dl.tensor1d([1, 0, 1, 0], 'bool');
      const a = dl.tensor1d([10, 10, 10, 10]);
      const b = dl.tensor1d([20, 20, 20, 20]);
      test_util.expectArraysClose(math.where(c, a, b), [10, 20, 10, 20]);
    });

    it('Tensor1D different a/b shapes', math => {
      let c = dl.tensor1d([1, 0, 1, 0], 'bool');
      let a = dl.tensor1d([10, 10, 10]);
      let b = dl.tensor1d([20, 20, 20, 20]);
      let f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();

      c = dl.tensor1d([1, 0, 1, 0], 'bool');
      a = dl.tensor1d([10, 10, 10, 10]);
      b = dl.tensor1d([20, 20, 20]);
      f = () => {
        math.where(c, a, b);
      };
    });

    it('Tensor1D different condition/a shapes', math => {
      const c = dl.tensor1d([1, 0, 1, 0], 'bool');
      const a = dl.tensor1d([10, 10, 10]);
      const b = dl.tensor1d([20, 20, 20]);
      const f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();
    });

    it('Tensor2D', math => {
      const c = dl.tensor2d([[1, 0], [0, 1]], [2, 2], 'bool');
      const a = dl.tensor2d([[10, 10], [10, 10]], [2, 2]);
      const b = dl.tensor2d([[5, 5], [5, 5]], [2, 2]);
      test_util.expectArraysClose(math.where(c, a, b), [10, 5, 5, 10]);
    });

    it('Tensor2D different a/b shapes', math => {
      let c = dl.tensor2d([[1, 1], [0, 0]], [2, 2], 'bool');
      let a = dl.tensor2d([[5, 5, 5], [5, 5, 5]], [2, 3]);
      let b = dl.tensor2d([[4, 4], [4, 4]], [2, 2]);
      let f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();

      c = dl.tensor2d([[1, 1], [0, 0]], [2, 2], 'bool');
      a = dl.tensor2d([[5, 5], [5, 5]], [2, 2]);
      b = dl.tensor2d([[4, 4, 4], [4, 4, 4]], [2, 3]);
      f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();
    });

    it('Tensor2D different condition/a shapes', math => {
      const c = dl.tensor2d([[1, 0], [0, 1]], [2, 2], 'bool');
      const a = dl.tensor2d([[10, 10, 10], [10, 10, 10]], [2, 3]);
      const b = dl.tensor2d([[5, 5, 5], [5, 5, 5]], [2, 3]);
      const f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();
    });

    it('Tensor2D different `a` dimension w/ condition rank=1', math => {
      const c = dl.tensor1d([1, 0, 1, 0], 'bool');
      let a = dl.tensor2d([[10, 10], [10, 10]], [2, 2]);
      let b = dl.tensor2d([[5, 5], [5, 5]], [2, 2]);
      const f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();

      a = dl.tensor2d([[10], [10], [10], [10]], [4, 1]);
      b = dl.tensor2d([[5], [5], [5], [5]], [4, 1]);
      test_util.expectArraysClose(math.where(c, a, b), [10, 5, 10, 5]);

      a = dl.tensor2d([[10, 10], [10, 10], [10, 10], [10, 10]], [4, 2]);
      b = dl.tensor2d([[5, 5], [5, 5], [5, 5], [5, 5]], [4, 2]);
      test_util.expectArraysClose(
          math.where(c, a, b), [10, 10, 5, 5, 10, 10, 5, 5]);
    });

    it('Tensor3D', math => {
      const c =
          dl.tensor3d([[[1], [0], [1]], [[0], [0], [0]]], [2, 3, 1], 'bool');
      const a = dl.tensor3d([[[5], [5], [5]], [[5], [5], [5]]], [2, 3, 1]);
      const b = dl.tensor3d([[[3], [3], [3]], [[3], [3], [3]]], [2, 3, 1]);
      test_util.expectArraysClose(math.where(c, a, b), [5, 3, 5, 3, 3, 3]);
    });

    it('Tensor3D different a/b shapes', math => {
      const c =
          dl.tensor3d([[[1], [0], [1]], [[0], [0], [0]]], [2, 3, 1], 'bool');
      let a = dl.tensor3d([[[5], [5]], [[5], [5]]], [2, 2, 1]);
      let b = dl.tensor3d([[[3], [3], [3]], [[3], [3], [3]]], [2, 3, 1]);
      let f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();

      a = dl.tensor3d([[[5], [5], [5]], [[5], [5], [5]]], [2, 3, 1]);
      b = dl.tensor3d([[[3], [3]], [[3], [3]]], [2, 2, 1]);
      f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();
    });

    it('Tensor3D different condition/a shapes', math => {
      const c = dl.tensor3d([[[1], [0]], [[0], [0]]], [2, 2, 1], 'bool');
      const a = dl.tensor3d([[[5], [5], [5]], [[5], [5], [5]]], [2, 3, 1]);
      const b = dl.tensor3d([[[3], [3], [3]], [[3], [3], [3]]], [2, 3, 1]);
      const f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();
    });

    it('Tensor3D different `a` dimension w/ condition rank=1', math => {
      const c = dl.tensor1d([1, 0, 1, 0], 'bool');
      let a = dl.tensor3d([[[9, 9], [9, 9]], [[9, 9], [9, 9]]], [2, 2, 2]);
      let b = dl.tensor3d([[[8, 8], [8, 8]], [[8, 8], [8, 8]]], [2, 2, 2]);
      const f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();

      a = dl.tensor3d([[[9]], [[9]], [[9]], [[9]]], [4, 1, 1]);
      b = dl.tensor3d([[[8]], [[8]], [[8]], [[8]]], [4, 1, 1]);
      test_util.expectArraysClose(math.where(c, a, b), [9, 8, 9, 8]);

      a = dl.tensor3d(
          [[[9], [9]], [[9], [9]], [[9], [9]], [[9], [9]]], [4, 2, 1]);
      b = dl.tensor3d(
          [[[8], [8]], [[8], [8]], [[8], [8]], [[8], [8]]], [4, 2, 1]);
      test_util.expectArraysClose(
          math.where(c, a, b), [9, 9, 8, 8, 9, 9, 8, 8]);
    });

    it('Tensor4D', math => {
      const c = dl.tensor4d([1, 0, 1, 1], [2, 2, 1, 1], 'bool');
      const a = dl.tensor4d([7, 7, 7, 7], [2, 2, 1, 1]);
      const b = dl.tensor4d([3, 3, 3, 3], [2, 2, 1, 1]);
      test_util.expectArraysClose(math.where(c, a, b), [7, 3, 7, 7]);
    });

    it('Tensor4D different a/b shapes', math => {
      const c = dl.tensor4d([1, 0, 1, 1], [2, 2, 1, 1], 'bool');
      let a = dl.tensor4d([7, 7, 7, 7, 7, 7, 7, 7], [2, 2, 2, 1]);
      let b = dl.tensor4d([3, 3, 3, 3], [2, 2, 1, 1]);
      let f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();

      a = dl.tensor4d([7, 7, 7, 7], [2, 2, 1, 1]);
      b = dl.tensor4d([3, 3, 3, 3, 3, 3, 3, 3], [2, 2, 2, 1]);
      f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();
    });

    it('Tensor4D different condition/a shapes', math => {
      const c = dl.tensor4d([1, 0, 1, 1, 1, 0, 1, 1], [2, 2, 2, 1], 'bool');
      const a = dl.tensor4d([7, 7, 7, 7], [2, 2, 1, 1]);
      const b = dl.tensor4d([3, 3, 3, 3], [2, 2, 1, 1]);
      const f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();
    });

    it('Tensor4D different `a` dimension w/ condition rank=1', math => {
      const c = dl.tensor1d([1, 0, 1, 0], 'bool');
      let a = dl.tensor4d([7, 7, 7, 7, 7, 7, 7, 7], [2, 2, 2, 1]);
      let b = dl.tensor4d([3, 3, 3, 3, 3, 3, 3, 3], [2, 2, 2, 1]);
      const f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();

      a = dl.tensor4d([7, 7, 7, 7], [4, 1, 1, 1]);
      b = dl.tensor4d([3, 3, 3, 3], [4, 1, 1, 1]);
      test_util.expectArraysClose(math.where(c, a, b), [7, 3, 7, 3]);

      a = dl.tensor4d([7, 7, 7, 7, 7, 7, 7, 7], [4, 2, 1, 1]);
      b = dl.tensor4d([3, 3, 3, 3, 3, 3, 3, 3], [4, 2, 1, 1]);
      test_util.expectArraysClose(
          math.where(c, a, b), [7, 7, 3, 3, 7, 7, 3, 3]);
    });
  };

  test_util.describeMathCPU('where', [tests]);
  test_util.describeMathGPU('where', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
