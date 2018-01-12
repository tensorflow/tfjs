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

import {Array1D, Array2D, Array3D, Array4D} from './ndarray';

// LogicalOr:
{
  const boolNaN = util.getNaN('bool');

  const tests: MathTests = it => {
    // Array1D:
    it('Array1D.', math => {
      let a = Array1D.new([1, 0, 0], 'bool');
      let b = Array1D.new([0, 1, 0], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [1, 1, 0]);

      a = Array1D.new([0, 0, 0], 'bool');
      b = Array1D.new([0, 0, 0], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [0, 0, 0]);

      a = Array1D.new([1, 1], 'bool');
      b = Array1D.new([1, 1], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [1, 1]);
    });
    it('mismatched Array1D shapes', math => {
      const a = Array1D.new([1, 0], 'bool');
      const b = Array1D.new([0, 1, 0], 'bool');
      const f = () => {
        math.logicalOr(a, b);
      };
      expect(f).toThrowError();
    });
    it('NaNs in Array1D', math => {
      const a = Array1D.new([1, NaN, 0], 'bool');
      const b = Array1D.new([0, 0, NaN], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [1, boolNaN, boolNaN]);
    });

    // Array2D:
    it('Array2D', math => {
      let a = Array2D.new([2, 3], [[1, 0, 1], [0, 0, 0]], 'bool');
      let b = Array2D.new([2, 3], [[0, 0, 0], [0, 1, 0]], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [1, 0, 1, 0, 1, 0]);

      a = Array2D.new([2, 3], [[0, 0, 0], [1, 1, 1]], 'bool');
      b = Array2D.new([2, 3], [[0, 0, 0], [1, 1, 1]], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [0, 0, 0, 1, 1, 1]);
    });
    it('broadcasting Array2D shapes', math => {
      const a = Array2D.new([2, 1], [[1], [0]], 'bool');
      const b = Array2D.new([2, 3], [[0, 0, 0], [0, 1, 0]], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [1, 1, 1, 0, 1, 0]);
    });
    it('NaNs in Array2D', math => {
      const a = Array2D.new([2, 2], [[1, NaN], [0, NaN]], 'bool');
      const b = Array2D.new([2, 2], [[0, NaN], [1, NaN]], 'bool');
      test_util.expectArraysClose(
          math.logicalOr(a, b), [1, boolNaN, 1, boolNaN]);
    });

    // Array3D:
    it('Array3D', math => {
      let a =
          Array3D.new([2, 3, 1], [[[1], [0], [1]], [[0], [0], [0]]], 'bool');
      let b =
          Array3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [0]]], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [1, 0, 1, 1, 0, 0]);

      a = Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'bool');
      b = Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [0, 0, 0, 1, 1, 1]);
    });
    it('broadcasting Array3D shapes', math => {
      const a = Array3D.new(
          [2, 3, 2], [[[1, 0], [0, 0], [1, 1]], [[0, 0], [0, 1], [0, 0]]],
          'bool');
      const b =
          Array3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [0]]], 'bool');
      test_util.expectArraysClose(
          math.logicalOr(a, b), [1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0]);
    });
    it('NaNs in Array3D', math => {
      const a =
          Array3D.new([2, 3, 1], [[[1], [NaN], [1]], [[0], [0], [0]]], 'bool');
      const b =
          Array3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [NaN]]], 'bool');
      test_util.expectArraysClose(
          math.logicalOr(a, b), [1, boolNaN, 1, 1, 0, boolNaN]);
    });

    // Array4D:
    it('Array4D', math => {
      let a = Array4D.new([2, 2, 1, 1], [1, 0, 1, 0], 'bool');
      let b = Array4D.new([2, 2, 1, 1], [0, 1, 0, 0], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [1, 1, 1, 0]);

      a = Array4D.new([2, 2, 1, 1], [0, 0, 0, 0], 'bool');
      b = Array4D.new([2, 2, 1, 1], [0, 0, 0, 0], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [0, 0, 0, 0]);

      a = Array4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'bool');
      b = Array4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'bool');
      test_util.expectArraysClose(math.logicalOr(a, b), [1, 1, 1, 1]);
    });
    it('broadcasting Array4D shapes', math => {
      const a = Array4D.new([2, 2, 1, 1], [1, 0, 1, 0], 'bool');
      const b = Array4D.new(
          [2, 2, 1, 2], [[[[1, 0]], [[0, 0]]], [[[0, 0]], [[1, 1]]]], 'bool');
      test_util.expectArraysClose(
          math.logicalOr(a, b), [1, 1, 0, 0, 1, 1, 1, 1]);
    });
    it('NaNs in Array4D', math => {
      const a = Array4D.new([2, 2, 1, 1], [1, NaN, 1, 0], 'bool');
      const b = Array4D.new([2, 2, 1, 1], [0, 1, 0, NaN], 'bool');
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
