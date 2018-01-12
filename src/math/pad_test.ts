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
import {Array1D, Array2D} from './ndarray';

// math.pad1D
{
  const tests: MathTests = it => {
    it('Should pad 1D arrays', math => {
      const a = Array1D.new([1, 2, 3, 4, 5, 6], 'int32');
      const b = math.pad1D(a, [2, 3]);
      test_util.expectArraysClose(b, [0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0]);
    });

    it('Should not pad 1D arrays with 0s', math => {
      const a = Array1D.new([1, 2, 3, 4], 'int32');
      const b = math.pad1D(a, [0, 0]);
      test_util.expectArraysClose(b, [1, 2, 3, 4]);
    });

    it('Should handle padding with custom value', math => {
      let a = Array1D.new([1, 2, 3, 4], 'int32');
      let b = math.pad1D(a, [2, 3], 9);
      test_util.expectArraysClose(b, [9, 9, 1, 2, 3, 4, 9, 9, 9]);

      a = Array1D.new([1, 2, 3, 4]);
      b = math.pad1D(a, [2, 1], 1.1);
      test_util.expectArraysClose(b, [1.1, 1.1, 1, 2, 3, 4, 1.1]);

      a = Array1D.new([1, 2, 3, 4]);
      b = math.pad1D(a, [2, 1], 1);
      test_util.expectArraysClose(b, [1, 1, 1, 2, 3, 4, 1]);
    });

    it('Should handle NaNs with 1D arrays', math => {
      const a = Array1D.new([1, NaN, 2, NaN]);
      const b = math.pad1D(a, [1, 1]);
      test_util.expectArraysClose(b, [0, 1, NaN, 2, NaN, 0]);
    });

    it('Should handle invalid paddings', math => {
      const a = Array1D.new([1, 2, 3, 4], 'int32');
      const f = () => {
        math.pad1D(a, [2, 2, 2]);
      };
      expect(f).toThrowError();
    });
  };

  test_util.describeMathCPU('pad1D', [tests]);
  test_util.describeMathGPU('pad1D', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.path2D
{
  const tests: MathTests = it => {
    it('Should pad 2D arrays', math => {
      let a = Array2D.new([2, 1], [[1], [2]], 'int32');
      let b = math.pad2D(a, [[1, 1], [1, 1]]);
      // 0, 0, 0
      // 0, 1, 0
      // 0, 2, 0
      // 0, 0, 0
      test_util.expectArraysClose(b, [0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0]);

      a = Array2D.new([2, 3], [[1, 2, 3], [4, 5, 6]], 'int32');
      b = math.pad2D(a, [[2, 2], [1, 1]]);
      // 0, 0, 0, 0, 0
      // 0, 0, 0, 0, 0
      // 0, 1, 2, 3, 0
      // 0, 4, 5, 6, 0
      // 0, 0, 0, 0, 0
      // 0, 0, 0, 0, 0
      test_util.expectArraysClose(b, [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0,
        0, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
      ]);
    });

    it('Should not pad 2D arrays with 0s', math => {
      const a = Array2D.new([2, 3], [[1, 2, 3], [4, 5, 6]], 'int32');
      const b = math.pad2D(a, [[0, 0], [0, 0]]);
      test_util.expectArraysClose(b, [1, 2, 3, 4, 5, 6]);
    });

    it('Should handle padding with custom value', math => {
      let a = Array2D.new([2, 3], [[1, 2, 3], [4, 5, 6]], 'int32');
      let b = math.pad2D(a, [[1, 1], [1, 1]], 10);
      test_util.expectArraysClose(b, [
        10, 10, 10, 10, 10, 10, 1,  2,  3,  10,
        10, 4,  5,  6,  10, 10, 10, 10, 10, 10
      ]);

      a = Array2D.new([2, 1], [[1], [1]]);
      b = math.pad2D(a, [[1, 1], [1, 1]], -2.1);
      test_util.expectArraysClose(
          b,
          [-2.1, -2.1, -2.1, -2.1, 1, -2.1, -2.1, 1, -2.1, -2.1, -2.1, -2.1]);

      a = Array2D.new([2, 1], [[1], [1]]);
      b = math.pad2D(a, [[1, 1], [1, 1]], -2);
      test_util.expectArraysClose(
          b, [-2, -2, -2, -2, 1, -2, -2, 1, -2, -2, -2, -2]);
    });

    it('Should handle NaNs with 2D arrays', math => {
      const a = Array2D.new([2, 2], [[1, NaN], [1, NaN]]);
      const b = math.pad2D(a, [[1, 1], [1, 1]]);
      // 0, 0, 0,   0
      // 0, 1, NaN, 0
      // 0, 1, NaN, 0
      // 0, 0, 0,   0
      test_util.expectArraysClose(
          b, [0, 0, 0, 0, 0, 1, NaN, 0, 0, 1, NaN, 0, 0, 0, 0, 0]);
    });

    it('Should handle invalid paddings', math => {
      const a = Array2D.new([2, 1], [[1], [2]], 'int32');
      const f = () => {
        math.pad2D(a, [[2, 2, 2], [1, 1, 1]]);
      };
      expect(f).toThrowError();
    });
  };

  test_util.describeMathCPU('pad2D', [tests]);
  test_util.describeMathGPU('pad2D', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
