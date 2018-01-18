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

// LogicalAnd:
{
  const boolNaN = util.getNaN('bool');

  const tests: MathTests = it => {
    // Array1D:
    it('Array1D.', math => {
      let a = Array1D.new([1, 0, 0], 'bool');
      let b = Array1D.new([0, 1, 0], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 0]);

      a = Array1D.new([0, 0, 0], 'bool');
      b = Array1D.new([0, 0, 0], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 0]);

      a = Array1D.new([1, 1], 'bool');
      b = Array1D.new([1, 1], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [1, 1]);
    });
    it('mismatched Array1D shapes', math => {
      const a = Array1D.new([1, 0], 'bool');
      const b = Array1D.new([0, 1, 0], 'bool');
      const f = () => {
        math.logicalAnd(a, b);
      };
      expect(f).toThrowError();
    });
    it('NaNs in Array1D', math => {
      const a = Array1D.new([1, NaN, 0], 'bool');
      const b = Array1D.new([0, 0, NaN], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [0, boolNaN, boolNaN]);
    });

    // Array2D:
    it('Array2D', math => {
      let a = Array2D.new([2, 3], [[1, 0, 1], [0, 0, 0]], 'bool');
      let b = Array2D.new([2, 3], [[0, 0, 0], [0, 1, 0]], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 0, 0, 0, 0]);

      a = Array2D.new([2, 3], [[0, 0, 0], [1, 1, 1]], 'bool');
      b = Array2D.new([2, 3], [[0, 0, 0], [1, 1, 1]], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 0, 1, 1, 1]);
    });
    it('broadcasting Array2D shapes', math => {
      const a = Array2D.new([2, 1], [[1], [0]], 'bool');
      const b = Array2D.new([2, 3], [[0, 1, 0], [0, 1, 0]], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [0, 1, 0, 0, 0, 0]);
    });
    it('NaNs in Array2D', math => {
      const a = Array2D.new([2, 2], [[1, NaN], [0, NaN]], 'bool');
      const b = Array2D.new([2, 2], [[0, NaN], [1, NaN]], 'bool');
      test_util.expectArraysClose(
          math.logicalAnd(a, b), [0, boolNaN, 0, boolNaN]);
    });

    // Array3D:
    it('Array3D', math => {
      let a =
          Array3D.new([2, 3, 1], [[[1], [0], [1]], [[0], [0], [1]]], 'bool');
      let b =
          Array3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [0]]], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 1, 0, 0, 0]);

      a = Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'bool');
      b = Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 0, 1, 1, 1]);
    });
    it('broadcasting Array3D shapes', math => {
      const a = Array3D.new(
          [2, 3, 2], [[[1, 0], [0, 0], [1, 1]], [[0, 0], [0, 1], [0, 0]]],
          'bool');
      const b =
          Array3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [0]]], 'bool');
      test_util.expectArraysClose(
          math.logicalAnd(a, b), [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]);
    });
    it('NaNs in Array3D', math => {
      const a =
          Array3D.new([2, 3, 1], [[[1], [NaN], [1]], [[0], [0], [0]]], 'bool');
      const b =
          Array3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [NaN]]], 'bool');
      test_util.expectArraysClose(
          math.logicalAnd(a, b), [0, boolNaN, 1, 0, 0, boolNaN]);
    });

    // Array4D:
    it('Array4D', math => {
      let a = Array4D.new([2, 2, 1, 1], [1, 0, 1, 0], 'bool');
      let b = Array4D.new([2, 2, 1, 1], [0, 1, 1, 0], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 1, 0]);

      a = Array4D.new([2, 2, 1, 1], [0, 0, 0, 0], 'bool');
      b = Array4D.new([2, 2, 1, 1], [0, 0, 0, 0], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 0, 0]);

      a = Array4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'bool');
      b = Array4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'bool');
      test_util.expectArraysClose(math.logicalAnd(a, b), [1, 1, 1, 1]);
    });
    it('broadcasting Array4D shapes', math => {
      const a = Array4D.new([2, 2, 1, 1], [1, 0, 1, 0], 'bool');
      const b = Array4D.new(
          [2, 2, 1, 2], [[[[1, 0]], [[0, 0]]], [[[0, 0]], [[1, 1]]]], 'bool');
      test_util.expectArraysClose(
          math.logicalAnd(a, b), [1, 0, 0, 0, 0, 0, 0, 0]);
    });
    it('NaNs in Array4D', math => {
      const a = Array4D.new([2, 2, 1, 1], [1, NaN, 1, 0], 'bool');
      const b = Array4D.new([2, 2, 1, 1], [0, 1, 0, NaN], 'bool');
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

// Where
{
  const tests: MathTests = it => {
    it('Array1D', math => {
      const c = Array1D.new([1, 0, 1, 0], 'bool');
      const a = Array1D.new([10, 10, 10, 10]);
      const b = Array1D.new([20, 20, 20, 20]);
      test_util.expectArraysClose(math.where(c, a, b), [10, 20, 10, 20]);
    });

    it('Array1D different a/b shapes', math => {
      let c = Array1D.new([1, 0, 1, 0], 'bool');
      let a = Array1D.new([10, 10, 10]);
      let b = Array1D.new([20, 20, 20, 20]);
      let f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();

      c = Array1D.new([1, 0, 1, 0], 'bool');
      a = Array1D.new([10, 10, 10, 10]);
      b = Array1D.new([20, 20, 20]);
      f = () => {
        math.where(c, a, b);
      };
    });

    it('Array1D different condition/a shapes', math => {
      const c = Array1D.new([1, 0, 1, 0], 'bool');
      const a = Array1D.new([10, 10, 10]);
      const b = Array1D.new([20, 20, 20]);
      const f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();
    });

    it('Array2D', math => {
      const c = Array2D.new([2, 2], [[1, 0], [0, 1]], 'bool');
      const a = Array2D.new([2, 2], [[10, 10], [10, 10]]);
      const b = Array2D.new([2, 2], [[5, 5], [5, 5]]);
      test_util.expectArraysClose(math.where(c, a, b), [10, 5, 5, 10]);
    });

    it('Array2D different a/b shapes', math => {
      let c = Array2D.new([2, 2], [[1, 1], [0, 0]], 'bool');
      let a = Array2D.new([2, 3], [[5, 5, 5], [5, 5, 5]]);
      let b = Array2D.new([2, 2], [[4, 4], [4, 4]]);
      let f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();

      c = Array2D.new([2, 2], [[1, 1], [0, 0]], 'bool');
      a = Array2D.new([2, 2], [[5, 5], [5, 5]]);
      b = Array2D.new([2, 3], [[4, 4, 4], [4, 4, 4]]);
      f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();
    });

    it('Array2D different condition/a shapes', math => {
      const c = Array2D.new([2, 2], [[1, 0], [0, 1]], 'bool');
      const a = Array2D.new([2, 3], [[10, 10, 10], [10, 10, 10]]);
      const b = Array2D.new([2, 3], [[5, 5, 5], [5, 5, 5]]);
      const f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();
    });

    it('Array2D different `a` dimension w/ condition rank=1', math => {
      const c = Array1D.new([1, 0, 1, 0], 'bool');
      let a = Array2D.new([2, 2], [[10, 10], [10, 10]]);
      let b = Array2D.new([2, 2], [[5, 5], [5, 5]]);
      const f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();

      a = Array2D.new([4, 1], [[10], [10], [10], [10]]);
      b = Array2D.new([4, 1], [[5], [5], [5], [5]]);
      test_util.expectArraysClose(math.where(c, a, b), [10, 5, 10, 5]);

      a = Array2D.new([4, 2], [[10, 10], [10, 10], [10, 10], [10, 10]]);
      b = Array2D.new([4, 2], [[5, 5], [5, 5], [5, 5], [5, 5]]);
      test_util.expectArraysClose(
          math.where(c, a, b), [10, 10, 5, 5, 10, 10, 5, 5]);
    });

    it('Array3D', math => {
      const c =
          Array3D.new([2, 3, 1], [[[1], [0], [1]], [[0], [0], [0]]], 'bool');
      const a = Array3D.new([2, 3, 1], [[[5], [5], [5]], [[5], [5], [5]]]);
      const b = Array3D.new([2, 3, 1], [[[3], [3], [3]], [[3], [3], [3]]]);
      test_util.expectArraysClose(math.where(c, a, b), [5, 3, 5, 3, 3, 3]);
    });

    it('Array3D different a/b shapes', math => {
      const c =
          Array3D.new([2, 3, 1], [[[1], [0], [1]], [[0], [0], [0]]], 'bool');
      let a = Array3D.new([2, 2, 1], [[[5], [5]], [[5], [5]]]);
      let b = Array3D.new([2, 3, 1], [[[3], [3], [3]], [[3], [3], [3]]]);
      let f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();

      a = Array3D.new([2, 3, 1], [[[5], [5], [5]], [[5], [5], [5]]]);
      b = Array3D.new([2, 2, 1], [[[3], [3]], [[3], [3]]]);
      f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();
    });

    it('Array3D different condition/a shapes', math => {
      const c = Array3D.new([2, 2, 1], [[[1], [0]], [[0], [0]]], 'bool');
      const a = Array3D.new([2, 3, 1], [[[5], [5], [5]], [[5], [5], [5]]]);
      const b = Array3D.new([2, 3, 1], [[[3], [3], [3]], [[3], [3], [3]]]);
      const f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();
    });

    it('Array3D different `a` dimension w/ condition rank=1', math => {
      const c = Array1D.new([1, 0, 1, 0], 'bool');
      let a = Array3D.new([2, 2, 2], [[[9, 9], [9, 9]], [[9, 9], [9, 9]]]);
      let b = Array3D.new([2, 2, 2], [[[8, 8], [8, 8]], [[8, 8], [8, 8]]]);
      const f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();

      a = Array3D.new([4, 1, 1], [[[9]], [[9]], [[9]], [[9]]]);
      b = Array3D.new([4, 1, 1], [[[8]], [[8]], [[8]], [[8]]]);
      test_util.expectArraysClose(math.where(c, a, b), [9, 8, 9, 8]);

      a = Array3D.new(
          [4, 2, 1], [[[9], [9]], [[9], [9]], [[9], [9]], [[9], [9]]]);
      b = Array3D.new(
          [4, 2, 1], [[[8], [8]], [[8], [8]], [[8], [8]], [[8], [8]]]);
      test_util.expectArraysClose(
          math.where(c, a, b), [9, 9, 8, 8, 9, 9, 8, 8]);
    });

    it('Array4D', math => {
      const c = Array4D.new([2, 2, 1, 1], [1, 0, 1, 1], 'bool');
      const a = Array4D.new([2, 2, 1, 1], [7, 7, 7, 7]);
      const b = Array4D.new([2, 2, 1, 1], [3, 3, 3, 3]);
      test_util.expectArraysClose(math.where(c, a, b), [7, 3, 7, 7]);
    });

    it('Array4D different a/b shapes', math => {
      const c = Array4D.new([2, 2, 1, 1], [1, 0, 1, 1], 'bool');
      let a = Array4D.new([2, 2, 2, 1], [7, 7, 7, 7, 7, 7, 7, 7]);
      let b = Array4D.new([2, 2, 1, 1], [3, 3, 3, 3]);
      let f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();

      a = Array4D.new([2, 2, 1, 1], [7, 7, 7, 7]);
      b = Array4D.new([2, 2, 2, 1], [3, 3, 3, 3, 3, 3, 3, 3]);
      f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();
    });

    it('Array4D different condition/a shapes', math => {
      const c = Array4D.new([2, 2, 2, 1], [1, 0, 1, 1, 1, 0, 1, 1], 'bool');
      const a = Array4D.new([2, 2, 1, 1], [7, 7, 7, 7]);
      const b = Array4D.new([2, 2, 1, 1], [3, 3, 3, 3]);
      const f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();
    });

    it('Array4D different `a` dimension w/ condition rank=1', math => {
      const c = Array1D.new([1, 0, 1, 0], 'bool');
      let a = Array4D.new([2, 2, 2, 1], [7, 7, 7, 7, 7, 7, 7, 7]);
      let b = Array4D.new([2, 2, 2, 1], [3, 3, 3, 3, 3, 3, 3, 3]);
      const f = () => {
        math.where(c, a, b);
      };
      expect(f).toThrowError();

      a = Array4D.new([4, 1, 1, 1], [7, 7, 7, 7]);
      b = Array4D.new([4, 1, 1, 1], [3, 3, 3, 3]);
      test_util.expectArraysClose(math.where(c, a, b), [7, 3, 7, 3]);

      a = Array4D.new([4, 2, 1, 1], [7, 7, 7, 7, 7, 7, 7, 7]);
      b = Array4D.new([4, 2, 1, 1], [3, 3, 3, 3, 3, 3, 3, 3]);
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
