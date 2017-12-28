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

import {Array3D, Array4D} from './ndarray';

// math.maxPoolBackprop
{
  const tests: MathTests = it => {
    it('x=3x3x1, f=2, s=1, no duplicate max value, test #1', math => {
      const dy = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
      const x = Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, 8, 9]);

      const dx = math.maxPoolBackprop(dy, x, 2, 1, 0);

      const expected = [0, 0, 0, 0, 1, 2, 0, 3, 4];
      test_util.expectArraysClose(dx, expected);
    });

    it('x=3x3x1, f=2, s=1, no duplicate max value, test #2', math => {
      const dy = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
      const x = Array3D.new([3, 3, 1], [9, 5, 6, 6, 8, 4, 9, 5, 10]);

      const dx = math.maxPoolBackprop(dy, x, 2, 1, 0);

      const expected = [1, 0, 0, 0, 2, 0, 3, 0, 4];
      test_util.expectArraysClose(dx, expected);
    });

    it('x=3x3x1, f=2, s=1 duplicate max value, test 1', math => {
      const dy = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
      const x = Array3D.new([3, 3, 1], [0, 0, 0, 0, 5, 0, 0, 0, 0]);

      const dx = math.maxPoolBackprop(dy, x, 2, 1, 0);

      const expected = [0, 0, 0, 0, 10, 0, 0, 0, 0];
      test_util.expectArraysClose(dx, expected);
    });

    it('x=3x3x1, f=2, s=1 duplicate max value, test 2', math => {
      const dy = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
      const x = Array3D.new([3, 3, 1], [1, 3, 2, 1, 2, 1, 1, 1, 5]);

      const dx = math.maxPoolBackprop(dy, x, 2, 1, 0);

      const expected = [0, 3, 0, 0, 3, 0, 0, 0, 4];
      test_util.expectArraysClose(dx, expected);
    });

    it('x=3x3x1, f=2, s=1, batch=2, duplicate max value in 2nd input', math => {
      const dy = Array4D.new([2, 2, 2, 1], [1, 2, 3, 4, 5, 6, 7, 8]);
      const x = Array4D.new(
          [2, 3, 3, 1], [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 9, 8]);

      const dx = math.maxPoolBackprop(dy, x, 2, 1, 0);

      const expected = new Float32Array(
          [0, 0, 0, 0, 1, 2, 0, 3, 4, 0, 0, 0, 0, 5, 6, 0, 15, 0]);
      test_util.expectArraysClose(dx, expected);
    });

    it('x=4x4x1, f=2, s=2, test #1', math => {
      const dy = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
      const x = Array3D.new(
          [4, 4, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);

      const dx = math.maxPoolBackprop(dy, x, 2, 2, 0);

      const expected = [0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 3, 0, 4];
      test_util.expectArraysClose(dx, expected);
    });

    it('x=4x4x1, f=2, s=2, test #2', math => {
      const dy = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
      const x = Array3D.new(
          [4, 4, 1], [1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1]);
      const expected = [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0];
      const dx = math.maxPoolBackprop(dy, x, 2, 2, 0);
      test_util.expectArraysClose(dx, expected);
    });

    it('x=5x5x1, f=3, s=2 no duplicate max value', math => {
      const dy = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
      const x = Array3D.new([5, 5, 1], [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
      ]);

      const dx = math.maxPoolBackprop(dy, x, 3, 2, 0);

      const expected = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 4
      ];
      test_util.expectArraysClose(dx, expected);
    });

    it('x=5x5x1, f=3, s=2 duplicate max value', math => {
      const dy = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
      const x = Array3D.new([5, 5, 1], [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 24,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 12
      ]);

      const dx = math.maxPoolBackprop(dy, x, 3, 2, 0);

      const expected = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
      ];
      test_util.expectArraysClose(dx, expected);
    });

    // Max pool backprop depth > 1.
    it('x=3x3x2, f=2, s=1, no duplicate max value', math => {
      // This test combines the first two 3x3x1 tests with no duplicates to
      // make depth=2,
      // dy is slightly modified to show the difference.
      const dy = Array3D.new([2, 2, 2], [1, 44, 2, 33, 3, 22, 4, 11]);
      const x = Array3D.new(
          [3, 3, 2],
          [1, 99, 2, 55, 3, 66, 4, 66, 5, 88, 6, 44, 7, 99, 8, 55, 9, 100]);

      const dx = math.maxPoolBackprop(dy, x, 2, 1, 0);

      const expected =
          [0, 44, 0, 0, 0, 0, 0, 0, 1, 33, 2, 0, 0, 22, 3, 0, 4, 11];
      test_util.expectArraysClose(dx, expected);
    });

    it('x=3x3x2, f=2, s=1, duplicate max value', math => {
      // This test combines the first two 3x3x1 tests with duplicates to
      // make depth=2,
      // dy is slightly modified to show the difference.
      const dy = Array3D.new([2, 2, 2], [1, 44, 2, 33, 3, 22, 4, 11]);
      const x = Array3D.new(
          [3, 3, 2], [0, 1, 0, 3, 0, 2, 0, 1, 5, 2, 0, 1, 0, 1, 0, 1, 0, 5]);

      const dx = math.maxPoolBackprop(dy, x, 2, 1, 0);

      const expected = new Float32Array(
          [0, 0, 0, 77, 0, 0, 0, 0, 10, 22, 0, 0, 0, 0, 0, 0, 0, 11]);
      test_util.expectArraysClose(dx, expected);
    });

    it('x=4x4x2, f=2, s=1', math => {
      // This test combines the first two 4x4x1 tests with duplicates to make
      // depth=2,
      // dy is slightly modified to show the difference.
      const dy = Array3D.new([2, 2, 2], [1, 11, 2, 22, 3, 33, 4, 44]);
      const x = Array3D.new([4, 4, 2], [
        0, 1, 1, 2, 2,  2, 3,  1, 4,  1, 5,  1, 6,  1, 7,  1,
        8, 1, 9, 1, 10, 1, 11, 1, 12, 1, 13, 2, 14, 2, 15, 1
      ]);

      const dx = math.maxPoolBackprop(dy, x, 2, 2, 0);

      const expected = [
        0, 0, 0, 11, 0, 22, 0, 0, 0, 0, 1, 0,  0, 0,  2, 0,
        0, 0, 0, 0,  0, 0,  0, 0, 0, 0, 3, 33, 0, 44, 4, 0
      ];
      test_util.expectArraysClose(dx, expected);
    });

    it('x=5x5x2, f=3, s=2 no duplicate max value', math => {
      // This test combines the first two 5x5x1 tests with duplicates to make
      // depth=2,
      // dy is slightly modified to show the difference.
      const dy = Array3D.new([2, 2, 2], [1, 11, 2, 22, 3, 33, 4, 44]);
      const x = Array3D.new([5, 5, 2], [
        0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,
        8,  9,  9,  10, 10, 11, 11, 12, 24, 13, 13, 14, 14, 15, 15, 16, 16,
        17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 12
      ]);

      const dx = math.maxPoolBackprop(dy, x, 3, 2, 0);

      const expected = [
        0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 110, 0, 0, 2, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 3, 0, 0, 0, 4, 0
      ];
      test_util.expectArraysClose(dx, expected);
    });
  };

  test_util.describeMathCPU('maxPoolBackprop', [tests]);
  test_util.describeMathGPU('maxPoolBackprop', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
