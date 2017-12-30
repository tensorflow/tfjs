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
import {Array1D, Array2D, Scalar} from './ndarray';

// math.prelu
{
  const tests: MathTests = it => {
    it('basic', math => {
      const x = Array1D.new([0, 1, -2, -4]);
      const a = Array1D.new([0.15, 0.2, 0.25, 0.15]);
      const result = math.prelu(x, a);

      expect(result.shape).toEqual(x.shape);
      test_util.expectArraysClose(result, [0, 1, -0.5, -0.6]);
    });

    it('propagates NaN', math => {
      const x = Array1D.new([0, 1, NaN]);
      const a = Array1D.new([0.15, 0.2, 0.25]);
      const result = math.prelu(x, a);

      expect(result.shape).toEqual(x.shape);
      test_util.expectArraysClose(result, [0, 1, NaN]);
    });
  };

  test_util.describeMathCPU('prelu', [tests]);
  test_util.describeMathGPU('prelu', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.preluDer
{
  const tests: MathTests = it => {
    it('basic', math => {
      const x = Array1D.new([0.5, 3, -0.1, -4]);
      const a = Array1D.new([0.2, 0.4, 0.25, 0.15]);
      const result = math.preluDer(x, a);

      expect(result.shape).toEqual(x.shape);
      test_util.expectArraysClose(result, [1, 1, 0.25, 0.15]);
    });

    it('propagates NaN', math => {
      const x = Array1D.new([0.5, -0.1, NaN]);
      const a = Array1D.new([0.2, 0.3, 0.25]);
      const result = math.preluDer(x, a);

      expect(result.shape).toEqual(x.shape);
      test_util.expectArraysClose(result, [1, 0.3, NaN]);
    });
  };

  test_util.describeMathCPU('preluDer', [tests]);
  test_util.describeMathGPU('preluDer', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.maximum
{
  const tests: MathTests = it => {
    it('float32 and float32', math => {
      const a = Array1D.new([0.5, 3, -0.1, -4]);
      const b = Array1D.new([0.2, 0.4, 0.25, 0.15]);
      const result = math.maximum(a, b);

      expect(result.shape).toEqual(a.shape);
      test_util.expectArraysClose(result, [0.5, 3, 0.25, 0.15]);
    });

    it('int32 and int32', math => {
      const a = Array1D.new([1, 5, 2, 3], 'int32');
      const b = Array1D.new([2, 3, 1, 4], 'int32');
      const result = math.maximum(a, b);

      expect(result.shape).toEqual(a.shape);
      expect(result.dtype).toBe('int32');
      test_util.expectArraysEqual(result, [2, 5, 2, 4]);
    });

    it('bool and bool', math => {
      const a = Array1D.new([true, false, false, true], 'bool');
      const b = Array1D.new([false, false, true, true], 'bool');
      const result = math.maximum(a, b);

      expect(result.shape).toEqual(a.shape);
      expect(result.dtype).toBe('bool');
      test_util.expectArraysEqual(result, [true, false, true, true]);
    });

    it('different dtypes throws error', math => {
      const a = Array1D.new([true, false, false, true], 'float32');
      const b = Array1D.new([false, false, true, true], 'int32');
      // tslint:disable-next-line:no-any
      expect(() => math.maximum(a, b as any)).toThrowError();
    });

    it('propagates NaN', math => {
      const a = Array1D.new([0.5, -0.1, NaN]);
      const b = Array1D.new([0.2, 0.3, 0.25]);
      const result = math.maximum(a, b);

      expect(result.shape).toEqual(a.shape);
      test_util.expectArraysClose(result, [0.5, 0.3, NaN]);
    });

    it('broadcasts array1d and scalar', math => {
      const a = Array1D.new([0.5, 3, -0.1, -4]);
      const b = Scalar.new(0.6);
      const result = math.maximum(a, b);

      expect(result.shape).toEqual(a.shape);
      test_util.expectArraysClose(result, [0.6, 3, 0.6, 0.6]);
    });

    it('broadcasts scalar and array1d', math => {
      const a = Scalar.new(0.6);
      const b = Array1D.new([0.5, 3, -0.1, -4]);
      const result = math.maximum(a, b);

      expect(result.shape).toEqual(b.shape);
      test_util.expectArraysClose(result, [0.6, 3, 0.6, 0.6]);
    });

    it('broadcasts array1d and array2d', math => {
      const a = Array1D.new([0.5, 0.3]);
      const b = Array2D.new([2, 2], [0.2, 0.4, 0.6, 0.15]);
      const result = math.maximum(a, b);

      expect(result.shape).toEqual(b.shape);
      test_util.expectArraysClose(result, [0.5, 0.4, 0.6, 0.3]);
    });

    it('broadcasts 2x1 array2d and 2x2 array2d', math => {
      const a = Array2D.new([2, 1], [0.5, 0.3]);
      const b = Array2D.new([2, 2], [0.2, 0.4, 0.6, 0.15]);
      const result = math.maximum(a, b);

      expect(result.shape).toEqual(b.shape);
      test_util.expectArraysClose(result, [0.5, 0.5, 0.6, 0.3]);
    });
  };

  test_util.describeMathCPU('maximum', [tests]);
  test_util.describeMathGPU('maximum', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.minimum
{
  const tests: MathTests = it => {
    it('float32 and float32', math => {
      const a = Array1D.new([0.5, 3, -0.1, -4]);
      const b = Array1D.new([0.2, 0.4, 0.25, 0.15]);
      const result = math.minimum(a, b);

      expect(result.shape).toEqual(a.shape);
      test_util.expectArraysClose(result, [0.2, 0.4, -0.1, -4]);
    });

    it('int32 and int32', math => {
      const a = Array1D.new([1, 5, 2, 3], 'int32');
      const b = Array1D.new([2, 3, 1, 4], 'int32');
      const result = math.minimum(a, b);

      expect(result.shape).toEqual(a.shape);
      expect(result.dtype).toBe('int32');
      test_util.expectArraysEqual(result, [1, 3, 1, 3]);
    });

    it('bool and bool', math => {
      const a = Array1D.new([true, false, false, true], 'bool');
      const b = Array1D.new([false, false, true, true], 'bool');
      const result = math.minimum(a, b);

      expect(result.shape).toEqual(a.shape);
      expect(result.dtype).toBe('bool');
      test_util.expectArraysEqual(result, [false, false, false, true]);
    });

    it('different dtypes throws error', math => {
      const a = Array1D.new([true, false, false, true], 'float32');
      const b = Array1D.new([false, false, true, true], 'int32');
      // tslint:disable-next-line:no-any
      expect(() => math.minimum(a, b as any)).toThrowError();
    });

    it('propagates NaN', math => {
      const a = Array1D.new([0.5, -0.1, NaN]);
      const b = Array1D.new([0.2, 0.3, 0.25]);
      const result = math.minimum(a, b);

      expect(result.shape).toEqual(a.shape);
      test_util.expectArraysClose(result, [0.2, -0.1, NaN]);
    });

    it('broadcasts array1d and scalar', math => {
      const a = Array1D.new([0.5, 3, -0.1, -4]);
      const b = Scalar.new(0.6);
      const result = math.minimum(a, b);

      expect(result.shape).toEqual(a.shape);
      test_util.expectArraysClose(result, [0.5, 0.6, -0.1, -4]);
    });

    it('broadcasts scalar and array1d', math => {
      const a = Scalar.new(0.6);
      const b = Array1D.new([0.5, 3, -0.1, -4]);
      const result = math.minimum(a, b);

      expect(result.shape).toEqual(b.shape);
      test_util.expectArraysClose(result, [0.5, 0.6, -0.1, -4]);
    });

    it('broadcasts array1d and array2d', math => {
      const a = Array1D.new([0.5, 0.3]);
      const b = Array2D.new([2, 2], [0.2, 0.4, 0.6, 0.15]);
      const result = math.minimum(a, b);

      expect(result.shape).toEqual(b.shape);
      test_util.expectArraysClose(result, [0.2, 0.3, 0.5, 0.15]);
    });

    it('broadcasts 2x1 array2d and 2x2 array2d', math => {
      const a = Array2D.new([2, 1], [0.5, 0.3]);
      const b = Array2D.new([2, 2], [0.2, 0.4, 0.6, 0.15]);
      const result = math.minimum(a, b);

      expect(result.shape).toEqual(b.shape);
      test_util.expectArraysClose(result, [0.2, 0.4, 0.3, 0.15]);
    });
  };

  test_util.describeMathCPU('minimum', [tests]);
  test_util.describeMathGPU('minimum', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
