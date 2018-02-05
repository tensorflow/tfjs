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

import {Tensor1D, Tensor2D, Tensor3D, Tensor4D} from './tensor';

// LogicalNot:
{
  const boolNaN = util.getNaN('bool');

  const tests: MathTests = it => {
    it('Tensor1D.', math => {
      let a = Tensor1D.new([1, 0, 0], 'bool');
      test_util.expectArraysClose(math.logicalNot(a), [0, 1, 1]);

      a = Tensor1D.new([0, 0, 0], 'bool');
      test_util.expectArraysClose(math.logicalNot(a), [1, 1, 1]);

      a = Tensor1D.new([1, 1], 'bool');
      test_util.expectArraysClose(math.logicalNot(a), [0, 0]);
    });
    it('NaNs in Tensor1D', math => {
      const a = Tensor1D.new([1, NaN, 0], 'bool');
      test_util.expectArraysClose(math.logicalNot(a), [0, boolNaN, 1]);
    });

    it('Tensor2D', math => {
      let a = Tensor2D.new([2, 3], [[1, 0, 1], [0, 0, 0]], 'bool');
      test_util.expectArraysClose(math.logicalNot(a), [0, 1, 0, 1, 1, 1]);

      a = Tensor2D.new([2, 3], [[0, 0, 0], [1, 1, 1]], 'bool');
      test_util.expectArraysClose(math.logicalNot(a), [1, 1, 1, 0, 0, 0]);
    });
    it('NaNs in Tensor2D', math => {
      const a = Tensor2D.new([2, 2], [[1, NaN], [0, NaN]], 'bool');
      test_util.expectArraysClose(math.logicalNot(a), [0, boolNaN, 1, boolNaN]);
    });

    it('Tensor3D', math => {
      let a =
          Tensor3D.new([2, 3, 1], [[[1], [0], [1]], [[0], [0], [0]]], 'bool');
      test_util.expectArraysClose(math.logicalNot(a), [0, 1, 0, 1, 1, 1]);

      a = Tensor3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'bool');
      test_util.expectArraysClose(math.logicalNot(a), [1, 1, 1, 0, 0, 0]);
    });
    it('NaNs in Tensor3D', math => {
      const a =
          Tensor3D.new([2, 3, 1], [[[1], [NaN], [1]], [[0], [0], [0]]], 'bool');
      test_util.expectArraysClose(math.logicalNot(a), [0, boolNaN, 0, 1, 1, 1]);
    });

    it('Tensor4D', math => {
      let a = Tensor4D.new([2, 2, 1, 1], [1, 0, 1, 0], 'bool');
      test_util.expectArraysClose(math.logicalNot(a), [0, 1, 0, 1]);

      a = Tensor4D.new([2, 2, 1, 1], [0, 0, 0, 0], 'bool');
      test_util.expectArraysClose(math.logicalNot(a), [1, 1, 1, 1]);

      a = Tensor4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'bool');
      test_util.expectArraysClose(math.logicalNot(a), [0, 0, 0, 0]);
    });
    it('NaNs in Tensor4D', math => {
      const a = Tensor4D.new([2, 2, 1, 1], [1, NaN, 1, 0], 'bool');
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
      let a = Tensor1D.new([1, 0, 0], 'bool');
      let b = Tensor1D.new([0, 1, 0], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 0]);

      a = Tensor1D.new([0, 0, 0], 'bool');
      b = Tensor1D.new([0, 0, 0], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 0]);

      a = Tensor1D.new([1, 1], 'bool');
      b = Tensor1D.new([1, 1], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [1, 1]);
    });
    it('mismatched Tensor1D shapes', math => {
      const a = Tensor1D.new([1, 0], 'bool');
      const b = Tensor1D.new([0, 1, 0], 'bool');
      const f = () => {
        math.logicalAnd(a, b);
      };
      expect(f).toThrowError();
    });
    it('NaNs in Tensor1D', math => {
      const a = Tensor1D.new([1, NaN, 0], 'bool');
      const b = Tensor1D.new([0, 0, NaN], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [0, boolNaN, boolNaN]);
    });

    it('Tensor2D', math => {
      let a = Tensor2D.new([2, 3], [[1, 0, 1], [0, 0, 0]], 'bool');
      let b = Tensor2D.new([2, 3], [[0, 0, 0], [0, 1, 0]], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 0, 0, 0, 0]);

      a = Tensor2D.new([2, 3], [[0, 0, 0], [1, 1, 1]], 'bool');
      b = Tensor2D.new([2, 3], [[0, 0, 0], [1, 1, 1]], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 0, 1, 1, 1]);
    });
    it('broadcasting Tensor2D shapes', math => {
      const a = Tensor2D.new([2, 1], [[1], [0]], 'bool');
      const b = Tensor2D.new([2, 3], [[0, 1, 0], [0, 1, 0]], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [0, 1, 0, 0, 0, 0]);
    });
    it('NaNs in Tensor2D', math => {
      const a = Tensor2D.new([2, 2], [[1, NaN], [0, NaN]], 'bool');
      const b = Tensor2D.new([2, 2], [[0, NaN], [1, NaN]], 'bool');
      test_util.expectArraysClose(
          math.logicalAnd(a, b), [0, boolNaN, 0, boolNaN]);
    });

    it('Tensor3D', math => {
      let a =
          Tensor3D.new([2, 3, 1], [[[1], [0], [1]], [[0], [0], [1]]], 'bool');
      let b =
          Tensor3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [0]]], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 1, 0, 0, 0]);

      a = Tensor3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'bool');
      b = Tensor3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 0, 1, 1, 1]);
    });
    it('broadcasting Tensor3D shapes', math => {
      const a = Tensor3D.new(
          [2, 3, 2], [[[1, 0], [0, 0], [1, 1]], [[0, 0], [0, 1], [0, 0]]],
          'bool');
      const b =
          Tensor3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [0]]], 'bool');
      test_util.expectArraysClose(
          math.logicalAnd(a, b), [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]);
    });
    it('NaNs in Tensor3D', math => {
      const a =
          Tensor3D.new([2, 3, 1], [[[1], [NaN], [1]], [[0], [0], [0]]], 'bool');
      const b =
          Tensor3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [NaN]]], 'bool');
      test_util.expectArraysClose(
          math.logicalAnd(a, b), [0, boolNaN, 1, 0, 0, boolNaN]);
    });

    it('Tensor4D', math => {
      let a = Tensor4D.new([2, 2, 1, 1], [1, 0, 1, 0], 'bool');
      let b = Tensor4D.new([2, 2, 1, 1], [0, 1, 1, 0], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 1, 0]);

      a = Tensor4D.new([2, 2, 1, 1], [0, 0, 0, 0], 'bool');
      b = Tensor4D.new([2, 2, 1, 1], [0, 0, 0, 0], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 0, 0]);

      a = Tensor4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'bool');
      b = Tensor4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [1, 1, 1, 1]);
    });
    it('broadcasting Tensor4D shapes', math => {
      const a = Tensor4D.new([2, 2, 1, 1], [1, 0, 1, 0], 'bool');
      const b = Tensor4D.new(
          [2, 2, 1, 2], [[[[1, 0]], [[0, 0]]], [[[0, 0]], [[1, 1]]]], 'bool');
      test_util.expectArraysClose(
          math.logicalAnd(a, b), [1, 0, 0, 0, 0, 0, 0, 0]);
    });
    it('NaNs in Tensor4D', math => {
      const a = Tensor4D.new([2, 2, 1, 1], [1, NaN, 1, 0], 'bool');
      const b = Tensor4D.new([2, 2, 1, 1], [0, 1, 0, NaN], 'bool');
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
      let a = Tensor1D.new([1, 0, 0], 'bool');
      let b = Tensor1D.new([0, 1, 0], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [1, 1, 0]);

      a = Tensor1D.new([0, 0, 0], 'bool');
      b = Tensor1D.new([0, 0, 0], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [0, 0, 0]);

      a = Tensor1D.new([1, 1], 'bool');
      b = Tensor1D.new([1, 1], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [1, 1]);
    });
    it('mismatched Tensor1D shapes', math => {
      const a = Tensor1D.new([1, 0], 'bool');
      const b = Tensor1D.new([0, 1, 0], 'bool');
      const f = () => {
        math.logicalOr(a, b);
      };
      expect(f).toThrowError();
    });
    it('NaNs in Tensor1D', math => {
      const a = Tensor1D.new([1, NaN, 0], 'bool');
      const b = Tensor1D.new([0, 0, NaN], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [1, boolNaN, boolNaN]);
    });

    it('Tensor2D', math => {
      let a = Tensor2D.new([2, 3], [[1, 0, 1], [0, 0, 0]], 'bool');
      let b = Tensor2D.new([2, 3], [[0, 0, 0], [0, 1, 0]], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [1, 0, 1, 0, 1, 0]);

      a = Tensor2D.new([2, 3], [[0, 0, 0], [1, 1, 1]], 'bool');
      b = Tensor2D.new([2, 3], [[0, 0, 0], [1, 1, 1]], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [0, 0, 0, 1, 1, 1]);
    });
    it('broadcasting Tensor2D shapes', math => {
      const a = Tensor2D.new([2, 1], [[1], [0]], 'bool');
      const b = Tensor2D.new([2, 3], [[0, 0, 0], [0, 1, 0]], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [1, 1, 1, 0, 1, 0]);
    });
    it('NaNs in Tensor2D', math => {
      const a = Tensor2D.new([2, 2], [[1, NaN], [0, NaN]], 'bool');
      const b = Tensor2D.new([2, 2], [[0, NaN], [1, NaN]], 'bool');
      test_util.expectArraysClose(
          math.logicalOr(a, b), [1, boolNaN, 1, boolNaN]);
    });

    it('Tensor3D', math => {
      let a =
          Tensor3D.new([2, 3, 1], [[[1], [0], [1]], [[0], [0], [0]]], 'bool');
      let b =
          Tensor3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [0]]], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [1, 0, 1, 1, 0, 0]);

      a = Tensor3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'bool');
      b = Tensor3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [0, 0, 0, 1, 1, 1]);
    });
    it('broadcasting Tensor3D shapes', math => {
      const a = Tensor3D.new(
          [2, 3, 2], [[[1, 0], [0, 0], [1, 1]], [[0, 0], [0, 1], [0, 0]]],
          'bool');
      const b =
          Tensor3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [0]]], 'bool');
      test_util.expectArraysClose(
          math.logicalOr(a, b), [1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0]);
    });
    it('NaNs in Tensor3D', math => {
      const a =
          Tensor3D.new([2, 3, 1], [[[1], [NaN], [1]], [[0], [0], [0]]], 'bool');
      const b =
          Tensor3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [NaN]]], 'bool');
      test_util.expectArraysClose(
          math.logicalOr(a, b), [1, boolNaN, 1, 1, 0, boolNaN]);
    });

    it('Tensor4D', math => {
      let a = Tensor4D.new([2, 2, 1, 1], [1, 0, 1, 0], 'bool');
      let b = Tensor4D.new([2, 2, 1, 1], [0, 1, 0, 0], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [1, 1, 1, 0]);

      a = Tensor4D.new([2, 2, 1, 1], [0, 0, 0, 0], 'bool');
      b = Tensor4D.new([2, 2, 1, 1], [0, 0, 0, 0], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [0, 0, 0, 0]);

      a = Tensor4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'bool');
      b = Tensor4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [1, 1, 1, 1]);
    });
    it('broadcasting Tensor4D shapes', math => {
      const a = Tensor4D.new([2, 2, 1, 1], [1, 0, 1, 0], 'bool');
      const b = Tensor4D.new(
          [2, 2, 1, 2], [[[[1, 0]], [[0, 0]]], [[[0, 0]], [[1, 1]]]], 'bool');
      test_util.expectArraysClose(
          math.logicalOr(a, b), [1, 1, 0, 0, 1, 1, 1, 1]);
    });
    it('NaNs in Tensor4D', math => {
      const a = Tensor4D.new([2, 2, 1, 1], [1, NaN, 1, 0], 'bool');
      const b = Tensor4D.new([2, 2, 1, 1], [0, 1, 0, NaN], 'bool');
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
      let a = Tensor1D.new([1, 0, 0], 'bool');
      let b = Tensor1D.new([0, 1, 0], 'bool');
      test_util.expectArraysClose(math.logicalXor(a, b), [1, 1, 0]);

      a = Tensor1D.new([0, 0, 0], 'bool');
      b = Tensor1D.new([0, 0, 0], 'bool');
      test_util.expectArraysClose(math.logicalXor(a, b), [0, 0, 0]);

      a = Tensor1D.new([1, 1], 'bool');
      b = Tensor1D.new([1, 1], 'bool');
      test_util.expectArraysClose(math.logicalXor(a, b), [0, 0]);
    });
    it('mismatched Tensor1D shapes', math => {
      const a = Tensor1D.new([1, 0], 'bool');
      const b = Tensor1D.new([0, 1, 0], 'bool');
      const f = () => {
        math.logicalXor(a, b);
      };
      expect(f).toThrowError();
    });
    it('NaNs in Tensor1D', math => {
      const a = Tensor1D.new([1, NaN, 0], 'bool');
      const b = Tensor1D.new([0, 0, NaN], 'bool');
      test_util.expectArraysClose(math.logicalXor(a, b), [1, boolNaN, boolNaN]);
    });

    // Tensor2D:
    it('Tensor2D', math => {
      let a = Tensor2D.new([2, 3], [[1, 0, 1], [0, 0, 0]], 'bool');
      let b = Tensor2D.new([2, 3], [[0, 0, 0], [0, 1, 0]], 'bool');
      test_util.expectArraysClose(math.logicalXor(a, b), [1, 0, 1, 0, 1, 0]);

      a = Tensor2D.new([2, 3], [[0, 0, 0], [1, 1, 1]], 'bool');
      b = Tensor2D.new([2, 3], [[0, 0, 0], [1, 1, 1]], 'bool');
      test_util.expectArraysClose(math.logicalXor(a, b), [0, 0, 0, 0, 0, 0]);
    });
    it('broadcasting Tensor2D shapes', math => {
      const a = Tensor2D.new([2, 1], [[1], [0]], 'bool');
      const b = Tensor2D.new([2, 3], [[0, 0, 0], [0, 1, 0]], 'bool');
      test_util.expectArraysClose(math.logicalXor(a, b), [1, 1, 1, 0, 1, 0]);
    });
    it('NaNs in Tensor2D', math => {
      const a = Tensor2D.new([2, 2], [[1, NaN], [0, NaN]], 'bool');
      const b = Tensor2D.new([2, 2], [[0, NaN], [1, NaN]], 'bool');
      test_util.expectArraysClose(
          math.logicalXor(a, b), [1, boolNaN, 1, boolNaN]);
    });

    // Tensor3D:
    it('Tensor3D', math => {
      let a =
          Tensor3D.new([2, 3, 1], [[[1], [0], [1]], [[0], [0], [0]]], 'bool');
      let b =
          Tensor3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [0]]], 'bool');
      test_util.expectArraysClose(math.logicalXor(a, b), [1, 0, 0, 1, 0, 0]);

      a = Tensor3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'bool');
      b = Tensor3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'bool');
      test_util.expectArraysClose(math.logicalXor(a, b), [0, 0, 0, 0, 0, 0]);
    });
    it('broadcasting Tensor3D shapes', math => {
      const a = Tensor3D.new(
          [2, 3, 2], [[[1, 0], [0, 0], [1, 1]], [[0, 0], [0, 1], [0, 0]]],
          'bool');
      const b =
          Tensor3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [0]]], 'bool');
      test_util.expectArraysClose(
          math.logicalXor(a, b), [1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]);
    });
    it('NaNs in Tensor3D', math => {
      const a =
          Tensor3D.new([2, 3, 1], [[[1], [NaN], [1]], [[0], [0], [0]]], 'bool');
      const b =
          Tensor3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [NaN]]], 'bool');
      test_util.expectArraysClose(
          math.logicalXor(a, b), [1, boolNaN, 0, 1, 0, boolNaN]);
    });

    // Tensor4D:
    it('Tensor4D', math => {
      let a = Tensor4D.new([2, 2, 1, 1], [1, 0, 1, 0], 'bool');
      let b = Tensor4D.new([2, 2, 1, 1], [0, 1, 1, 0], 'bool');
      test_util.expectArraysClose(math.logicalXor(a, b), [1, 1, 0, 0]);

      a = Tensor4D.new([2, 2, 1, 1], [0, 0, 0, 0], 'bool');
      b = Tensor4D.new([2, 2, 1, 1], [0, 0, 0, 0], 'bool');
      test_util.expectArraysClose(math.logicalXor(a, b), [0, 0, 0, 0]);

      a = Tensor4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'bool');
      b = Tensor4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'bool');
      test_util.expectArraysClose(math.logicalXor(a, b), [0, 0, 0, 0]);
    });
    it('broadcasting Tensor4D shapes', math => {
      const a = Tensor4D.new([2, 2, 1, 1], [1, 0, 1, 0], 'bool');
      const b = Tensor4D.new(
          [2, 2, 1, 2], [[[[1, 0]], [[0, 0]]], [[[0, 0]], [[1, 1]]]], 'bool');
      test_util.expectArraysClose(
          math.logicalXor(a, b), [0, 1, 0, 0, 1, 1, 1, 1]);
    });
    it('NaNs in Tensor4D', math => {
      const a = Tensor4D.new([2, 2, 1, 1], [1, NaN, 1, 0], 'bool');
      const b = Tensor4D.new([2, 2, 1, 1], [0, 1, 0, NaN], 'bool');
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
      const c = Tensor1D.new([1, 0, 1, 0], 'bool');
      const a = Tensor1D.new([10, 10, 10, 10]);
      const b = Tensor1D.new([20, 20, 20, 20]);
      test_util.expectArraysClose(math.where(c, a, b), [10, 20, 10, 20]);
    });

    it('Tensor1D different a/b shapes', math => {
      let c = Tensor1D.new([1, 0, 1, 0], 'bool');
      let a = Tensor1D.new([10, 10, 10]);
      let b = Tensor1D.new([20, 20, 20, 20]);
      let f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();

      c = Tensor1D.new([1, 0, 1, 0], 'bool');
      a = Tensor1D.new([10, 10, 10, 10]);
      b = Tensor1D.new([20, 20, 20]);
      f = () => {
        math.where(c, a, b);
      };
    });

    it('Tensor1D different condition/a shapes', math => {
      const c = Tensor1D.new([1, 0, 1, 0], 'bool');
      const a = Tensor1D.new([10, 10, 10]);
      const b = Tensor1D.new([20, 20, 20]);
      const f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();
    });

    it('Tensor2D', math => {
      const c = Tensor2D.new([2, 2], [[1, 0], [0, 1]], 'bool');
      const a = Tensor2D.new([2, 2], [[10, 10], [10, 10]]);
      const b = Tensor2D.new([2, 2], [[5, 5], [5, 5]]);
      test_util.expectArraysClose(math.where(c, a, b), [10, 5, 5, 10]);
    });

    it('Tensor2D different a/b shapes', math => {
      let c = Tensor2D.new([2, 2], [[1, 1], [0, 0]], 'bool');
      let a = Tensor2D.new([2, 3], [[5, 5, 5], [5, 5, 5]]);
      let b = Tensor2D.new([2, 2], [[4, 4], [4, 4]]);
      let f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();

      c = Tensor2D.new([2, 2], [[1, 1], [0, 0]], 'bool');
      a = Tensor2D.new([2, 2], [[5, 5], [5, 5]]);
      b = Tensor2D.new([2, 3], [[4, 4, 4], [4, 4, 4]]);
      f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();
    });

    it('Tensor2D different condition/a shapes', math => {
      const c = Tensor2D.new([2, 2], [[1, 0], [0, 1]], 'bool');
      const a = Tensor2D.new([2, 3], [[10, 10, 10], [10, 10, 10]]);
      const b = Tensor2D.new([2, 3], [[5, 5, 5], [5, 5, 5]]);
      const f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();
    });

    it('Tensor2D different `a` dimension w/ condition rank=1', math => {
      const c = Tensor1D.new([1, 0, 1, 0], 'bool');
      let a = Tensor2D.new([2, 2], [[10, 10], [10, 10]]);
      let b = Tensor2D.new([2, 2], [[5, 5], [5, 5]]);
      const f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();

      a = Tensor2D.new([4, 1], [[10], [10], [10], [10]]);
      b = Tensor2D.new([4, 1], [[5], [5], [5], [5]]);
      test_util.expectArraysClose(math.where(c, a, b), [10, 5, 10, 5]);

      a = Tensor2D.new([4, 2], [[10, 10], [10, 10], [10, 10], [10, 10]]);
      b = Tensor2D.new([4, 2], [[5, 5], [5, 5], [5, 5], [5, 5]]);
      test_util.expectArraysClose(
          math.where(c, a, b), [10, 10, 5, 5, 10, 10, 5, 5]);
    });

    it('Tensor3D', math => {
      const c =
          Tensor3D.new([2, 3, 1], [[[1], [0], [1]], [[0], [0], [0]]], 'bool');
      const a = Tensor3D.new([2, 3, 1], [[[5], [5], [5]], [[5], [5], [5]]]);
      const b = Tensor3D.new([2, 3, 1], [[[3], [3], [3]], [[3], [3], [3]]]);
      test_util.expectArraysClose(math.where(c, a, b), [5, 3, 5, 3, 3, 3]);
    });

    it('Tensor3D different a/b shapes', math => {
      const c =
          Tensor3D.new([2, 3, 1], [[[1], [0], [1]], [[0], [0], [0]]], 'bool');
      let a = Tensor3D.new([2, 2, 1], [[[5], [5]], [[5], [5]]]);
      let b = Tensor3D.new([2, 3, 1], [[[3], [3], [3]], [[3], [3], [3]]]);
      let f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();

      a = Tensor3D.new([2, 3, 1], [[[5], [5], [5]], [[5], [5], [5]]]);
      b = Tensor3D.new([2, 2, 1], [[[3], [3]], [[3], [3]]]);
      f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();
    });

    it('Tensor3D different condition/a shapes', math => {
      const c = Tensor3D.new([2, 2, 1], [[[1], [0]], [[0], [0]]], 'bool');
      const a = Tensor3D.new([2, 3, 1], [[[5], [5], [5]], [[5], [5], [5]]]);
      const b = Tensor3D.new([2, 3, 1], [[[3], [3], [3]], [[3], [3], [3]]]);
      const f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();
    });

    it('Tensor3D different `a` dimension w/ condition rank=1', math => {
      const c = Tensor1D.new([1, 0, 1, 0], 'bool');
      let a = Tensor3D.new([2, 2, 2], [[[9, 9], [9, 9]], [[9, 9], [9, 9]]]);
      let b = Tensor3D.new([2, 2, 2], [[[8, 8], [8, 8]], [[8, 8], [8, 8]]]);
      const f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();

      a = Tensor3D.new([4, 1, 1], [[[9]], [[9]], [[9]], [[9]]]);
      b = Tensor3D.new([4, 1, 1], [[[8]], [[8]], [[8]], [[8]]]);
      test_util.expectArraysClose(math.where(c, a, b), [9, 8, 9, 8]);

      a = Tensor3D.new(
          [4, 2, 1], [[[9], [9]], [[9], [9]], [[9], [9]], [[9], [9]]]);
      b = Tensor3D.new(
          [4, 2, 1], [[[8], [8]], [[8], [8]], [[8], [8]], [[8], [8]]]);
      test_util.expectArraysClose(
          math.where(c, a, b), [9, 9, 8, 8, 9, 9, 8, 8]);
    });

    it('Tensor4D', math => {
      const c = Tensor4D.new([2, 2, 1, 1], [1, 0, 1, 1], 'bool');
      const a = Tensor4D.new([2, 2, 1, 1], [7, 7, 7, 7]);
      const b = Tensor4D.new([2, 2, 1, 1], [3, 3, 3, 3]);
      test_util.expectArraysClose(math.where(c, a, b), [7, 3, 7, 7]);
    });

    it('Tensor4D different a/b shapes', math => {
      const c = Tensor4D.new([2, 2, 1, 1], [1, 0, 1, 1], 'bool');
      let a = Tensor4D.new([2, 2, 2, 1], [7, 7, 7, 7, 7, 7, 7, 7]);
      let b = Tensor4D.new([2, 2, 1, 1], [3, 3, 3, 3]);
      let f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();

      a = Tensor4D.new([2, 2, 1, 1], [7, 7, 7, 7]);
      b = Tensor4D.new([2, 2, 2, 1], [3, 3, 3, 3, 3, 3, 3, 3]);
      f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();
    });

    it('Tensor4D different condition/a shapes', math => {
      const c = Tensor4D.new([2, 2, 2, 1], [1, 0, 1, 1, 1, 0, 1, 1], 'bool');
      const a = Tensor4D.new([2, 2, 1, 1], [7, 7, 7, 7]);
      const b = Tensor4D.new([2, 2, 1, 1], [3, 3, 3, 3]);
      const f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();
    });

    it('Tensor4D different `a` dimension w/ condition rank=1', math => {
      const c = Tensor1D.new([1, 0, 1, 0], 'bool');
      let a = Tensor4D.new([2, 2, 2, 1], [7, 7, 7, 7, 7, 7, 7, 7]);
      let b = Tensor4D.new([2, 2, 2, 1], [3, 3, 3, 3, 3, 3, 3, 3]);
      const f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();

      a = Tensor4D.new([4, 1, 1, 1], [7, 7, 7, 7]);
      b = Tensor4D.new([4, 1, 1, 1], [3, 3, 3, 3]);
      test_util.expectArraysClose(math.where(c, a, b), [7, 3, 7, 3]);

      a = Tensor4D.new([4, 2, 1, 1], [7, 7, 7, 7, 7, 7, 7, 7]);
      b = Tensor4D.new([4, 2, 1, 1], [3, 3, 3, 3, 3, 3, 3, 3]);
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
