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
import * as util from '../util';

import {Array1D, Array2D, Scalar} from './ndarray';

// math.relu
{
  const tests: MathTests = it => {
    it('basic', math => {
      const a = Array1D.new([1, -2, 0, 3, -0.1]);
      const result = math.relu(a);
      test_util.expectArraysClose(result, [1, 0, 0, 3, 0]);
    });

    it('does nothing to positive values', math => {
      const a = Scalar.new(1);
      const result = math.relu(a);
      test_util.expectNumbersClose(result.get(), 1);
    });

    it('sets negative values to 0', math => {
      const a = Scalar.new(-1);
      const result = math.relu(a);
      test_util.expectNumbersClose(result.get(), 0);
    });

    it('preserves zero values', math => {
      const a = Scalar.new(0);
      const result = math.relu(a);
      test_util.expectNumbersClose(result.get(), 0);
    });

    it('propagates NaNs, float32', math => {
      const a = Array1D.new([1, -2, 0, 3, -0.1, NaN]);
      const result = math.relu(a);
      expect(result.dtype).toBe('float32');
      test_util.expectArraysClose(result, [1, 0, 0, 3, 0, NaN]);
    });

    it('propagates NaNs, int32', math => {
      const a = Array1D.new([1, -2, 0, 3, -1, util.NAN_INT32], 'int32');
      const result = math.relu(a);
      expect(result.dtype).toBe('int32');
      test_util.expectArraysClose(result, [1, 0, 0, 3, 0, util.NAN_INT32]);
    });

    it('propagates NaNs, bool', math => {
      const a = Array1D.new([1, 0, 0, 1, 0, util.NAN_BOOL], 'bool');
      const result = math.relu(a);
      expect(result.dtype).toBe('bool');
      test_util.expectArraysClose(result, [1, 0, 0, 1, 0, util.NAN_BOOL]);
    });

    it('gradients: positive scalar', math => {
      const a = Scalar.new(3);
      const dy = Scalar.new(5);

      const gradients = math.vjp(() => math.relu(a), a, dy);

      expect(gradients.shape).toEqual(a.shape);
      expect(gradients.dtype).toEqual('float32');
      test_util.expectArraysClose(gradients, [5]);
    });

    it('gradients: negative scalar', math => {
      const a = Scalar.new(-3);
      const dy = Scalar.new(5);

      const gradients = math.vjp(() => math.relu(a), a, dy);

      expect(gradients.shape).toEqual(a.shape);
      expect(gradients.dtype).toEqual('float32');
      test_util.expectArraysClose(gradients, [0]);
    });

    it('gradients: array', math => {
      // TODO(nsthorat): Use 0 instead of -.001 when we fix the precision
      const a = Array2D.new([2, 2], [1, -1, -.001, .1]);
      const dy = Array2D.new([2, 2], [1, 2, 3, 4]);

      const gradients = math.vjp(() => math.relu(a), a, dy);

      expect(gradients.shape).toEqual(a.shape);
      expect(gradients.dtype).toEqual('float32');
      test_util.expectArraysClose(gradients, [1, 0, 0, 4]);
    });
  };

  test_util.describeMathCPU('relu', [tests]);
  test_util.describeMathGPU('relu', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.abs
{
  const tests: MathTests = it => {
    it('basic', math => {
      const a = Array1D.new([1, -2, 0, 3, -0.1]);
      const result = math.abs(a);
      test_util.expectArraysClose(result, [1, 2, 0, 3, 0.1]);
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([1, -2, 0, 3, -0.1, NaN]);
      const result = math.abs(a);
      test_util.expectArraysClose(result, [1, 2, 0, 3, 0.1, NaN]);
    });
  };

  test_util.describeMathCPU('abs', [tests]);
  test_util.describeMathGPU('abs', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.step
{
  const tests: MathTests = it => {
    it('with 1d ndarray', math => {
      const a = Array1D.new([1, -2, -.01, 3, -0.1]);
      const result = math.step(a);
      test_util.expectArraysClose(result, [1, 0, 0, 1, 0]);
    });

    it('with 2d ndarray', math => {
      const a = Array2D.new([2, 2], [1, -5, -3, 4]);
      const result = math.step(a);
      expect(result.shape).toEqual([2, 2]);
      test_util.expectArraysClose(result, [1, 0, 0, 1]);
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([1, -2, -.01, 3, NaN]);
      const result = math.step(a);
      test_util.expectArraysClose(result, [1, 0, 0, 1, NaN]);
    });
  };

  test_util.describeMathCPU('step', [tests]);
  test_util.describeMathGPU('step', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.neg
{
  const tests: MathTests = it => {
    it('basic', math => {
      const a = Array1D.new([1, -3, 2, 7, -4]);
      const result = math.neg(a);
      test_util.expectArraysClose(result, [-1, 3, -2, -7, 4]);
    });

    it('propagate NaNs', math => {
      const a = Array1D.new([1, -3, 2, 7, NaN]);
      const result = math.neg(a);
      const expected = [-1, 3, -2, -7, NaN];
      test_util.expectArraysClose(result, expected);
    });
  };

  test_util.describeMathCPU('neg', [tests]);
  test_util.describeMathGPU('neg', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.sigmoid
{
  const tests: MathTests = it => {
    it('basic', math => {
      const values = [1, -3, 2, 7, -4];
      const a = Array1D.new(values);

      const result = math.sigmoid(a);

      const expected = [];
      for (let i = 0; i < a.size; i++) {
        expected[i] = 1 / (1 + Math.exp(-values[i]));
      }
      test_util.expectArraysClose(result, expected);
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([3, NaN]);
      const res = math.sigmoid(a);
      test_util.expectArraysClose(res, [1 / (1 + Math.exp(-3)), NaN]);
    });
  };

  test_util.describeMathCPU('sigmoid', [tests]);
  test_util.describeMathGPU('sigmoid', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.sqrt
{
  const tests: MathTests = it => {
    it('sqrt', math => {
      const a = Array1D.new([2, 4]);
      const r = math.sqrt(a);
      test_util.expectNumbersClose(r.get(0), Math.sqrt(2));
      test_util.expectNumbersClose(r.get(1), Math.sqrt(4));
    });

    it('sqrt propagates NaNs', math => {
      const a = Array1D.new([1, NaN]);
      const r = math.sqrt(a);
      test_util.expectArraysClose(r, [Math.sqrt(1), NaN]);
    });
  };

  test_util.describeMathCPU('sqrt', [tests]);
  test_util.describeMathGPU('sqrt', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.square
{
  const tests: MathTests = it => {
    it('1D array', math => {
      const a = Array1D.new([2, 4, Math.sqrt(2)]);
      const r = math.square(a);
      test_util.expectArraysClose(r, [4, 16, 2]);
    });

    it('2D array', math => {
      const a = Array2D.new([2, 2], [1, 2, Math.sqrt(2), Math.sqrt(3)]);
      const r = math.square(a);
      expect(r.shape).toEqual([2, 2]);
      test_util.expectArraysClose(r, [1, 4, 2, 3]);
    });

    it('square propagates NaNs', math => {
      const a = Array1D.new([1.5, NaN]);
      const r = math.square(a);
      test_util.expectArraysClose(r, [2.25, NaN]);
    });

    it('gradients: Scalar', math => {
      const a = Scalar.new(5);
      const dy = Scalar.new(8);

      const gradients = math.vjp(() => math.square(a), a, dy);

      expect(gradients.shape).toEqual(a.shape);
      expect(gradients.dtype).toEqual('float32');
      test_util.expectArraysClose(gradients, [2 * 5 * 8], 1e-1);
    });

    it('gradients: Array1D', math => {
      const a = Array1D.new([-1, 2, 3, -5]);
      const dy = Array1D.new([1, 2, 3, 4]);

      const gradients = math.vjp(() => math.square(a), a, dy);

      expect(gradients.shape).toEqual(a.shape);
      expect(gradients.dtype).toEqual('float32');
      test_util.expectArraysClose(gradients, [-2, 4 * 2, 6 * 3, -10 * 4], 1e-1);
    });

    it('gradients: Array2D', math => {
      const a = Array2D.new([2, 2], [-3, 1, 2, 3]);
      const dy = Array2D.new([2, 2], [1, 2, 3, 4]);

      const gradients = math.vjp(() => math.square(a), a, dy);

      expect(gradients.shape).toEqual(a.shape);
      expect(gradients.dtype).toEqual('float32');
      test_util.expectArraysClose(
          gradients, [-6 * 1, 2 * 2, 4 * 3, 6 * 4], 1e-1);
    });
  };

  test_util.describeMathCPU('square', [tests]);
  test_util.describeMathGPU('square', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.log
{
  const tests: MathTests = it => {
    it('log', math => {
      const a = Array1D.new([1, 2]);
      const r = math.log(a);
      test_util.expectNumbersClose(r.get(0), Math.log(1));
      test_util.expectNumbersClose(r.get(1), Math.log(2));
    });

    it('log propagates NaNs', math => {
      const a = Array1D.new([1, NaN]);
      const r = math.log(a);
      test_util.expectArraysClose(r, [Math.log(1), NaN]);
    });
  };

  test_util.describeMathCPU('log', [tests]);
  test_util.describeMathGPU('log', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.ceil
{
  const tests: MathTests = it => {
    it('basic', math => {
      const a = Array1D.new([1.5, 2.1, -1.4]);
      const r = math.ceil(a);
      test_util.expectNumbersClose(r.get(0), 2);
      test_util.expectNumbersClose(r.get(1), 3);
      test_util.expectNumbersClose(r.get(2), -1);
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([1.5, NaN, -1.4]);
      const r = math.ceil(a);
      test_util.expectArraysClose(r, [2, NaN, -1]);
    });
  };

  test_util.describeMathCPU('ceil', [tests]);
  test_util.describeMathGPU('ceil', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.floor
{
  const tests: MathTests = it => {
    it('basic', math => {
      const a = Array1D.new([1.5, 2.1, -1.4]);
      const r = math.floor(a);

      test_util.expectNumbersClose(r.get(0), 1);
      test_util.expectNumbersClose(r.get(1), 2);
      test_util.expectNumbersClose(r.get(2), -2);
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([1.5, NaN, -1.4]);
      const r = math.floor(a);
      test_util.expectArraysClose(r, [1, NaN, -2]);
    });
  };

  test_util.describeMathCPU('floor', [tests]);
  test_util.describeMathGPU('floor', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.exp
{
  const tests: MathTests = it => {
    it('exp', math => {
      const a = Array1D.new([1, 2, 0]);
      const r = math.exp(a);

      test_util.expectNumbersClose(r.get(0), Math.exp(1));
      test_util.expectNumbersClose(r.get(1), Math.exp(2));
      test_util.expectNumbersClose(r.get(2), 1);
    });

    it('exp propagates NaNs', math => {
      const a = Array1D.new([1, NaN, 0]);
      const r = math.exp(a);
      test_util.expectArraysClose(r, [Math.exp(1), NaN, 1]);
    });
  };

  test_util.describeMathCPU('exp', [tests]);
  test_util.describeMathGPU('exp', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.sin
{
  const tests: MathTests = it => {
    it('basic', math => {
      const values = [1, -3, 2, 7, -4];
      const a = Array1D.new(values);
      const result = math.sin(a);

      const expected = [];
      for (let i = 0; i < a.size; i++) {
        expected[i] = Math.sin(values[i]);
      }
      test_util.expectArraysClose(result, expected);
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([4, NaN, 0]);
      const res = math.sin(a);
      test_util.expectArraysClose(res, [Math.sin(4), NaN, Math.sin(0)]);
    });
  };

  test_util.describeMathCPU('sin', [tests]);
  test_util.describeMathGPU('sin', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.cos
{
  const tests: MathTests = it => {
    it('basic', math => {
      const values = [1, -3, 2, 7, -4];
      const a = Array1D.new(values);
      const result = math.cos(a);

      const expected = [];
      for (let i = 0; i < a.size; i++) {
        expected[i] = Math.cos(values[i]);
      }
      test_util.expectArraysClose(result, expected);
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([4, NaN, 0]);
      const res = math.cos(a);
      test_util.expectArraysClose(res, [Math.cos(4), NaN, Math.cos(0)]);
    });
  };

  test_util.describeMathCPU('cos', [tests]);
  test_util.describeMathGPU('cos', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.tan
{
  const tests: MathTests = it => {
    it('basic', math => {
      const values = [1, -3, 2, 7, -4];
      const a = Array1D.new(values);
      const result = math.tan(a);

      const expected = [];
      for (let i = 0; i < a.size; i++) {
        expected[i] = Math.tan(values[i]);
      }
      test_util.expectArraysClose(result, expected, 1e-1);
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([4, NaN, 0]);
      const res = math.tan(a);
      test_util.expectArraysClose(res, [Math.tan(4), NaN, Math.tan(0)]);
    });
  };

  test_util.describeMathCPU('tan', [tests]);
  test_util.describeMathGPU('tan', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.asin
{
  const tests: MathTests = it => {
    it('basic', math => {
      const values = [.1, -3, 2, 7, -4];
      const a = Array1D.new(values);
      const result = math.asin(a);

      const expected = [];
      for (let i = 0; i < a.size; i++) {
        expected[i] = Math.asin(values[i]);
      }
      test_util.expectArraysClose(result, expected);
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([4, NaN, 0]);
      const res = math.asin(a);
      test_util.expectArraysClose(res, [Math.asin(4), NaN, Math.asin(0)]);
    });
  };

  test_util.describeMathCPU('asin', [tests]);
  test_util.describeMathGPU('asin', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.acos
{
  const tests: MathTests = it => {
    it('basic', math => {
      const values = [.1, -3, 2, 7, -4];
      const a = Array1D.new(values);
      const result = math.acos(a);

      const expected = [];
      for (let i = 0; i < a.size; i++) {
        expected[i] = Math.acos(values[i]);
      }
      // TODO(nsthorat): Fix the precision with byte textures here.
      test_util.expectArraysClose(result, expected, 1e-1);
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([4, NaN, 0]);
      const res = math.acos(a);
      test_util.expectArraysClose(res, [Math.acos(4), NaN, Math.acos(0)]);
    });
  };

  test_util.describeMathCPU('acos', [tests]);
  test_util.describeMathGPU('acos', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.atan
{
  const tests: MathTests = it => {
    it('basic', math => {
      const values = [1, -3, 2, 7, -4];
      const a = Array1D.new(values);
      const result = math.atan(a);

      const expected = [];
      for (let i = 0; i < a.size; i++) {
        expected[i] = Math.atan(values[i]);
      }
      test_util.expectArraysClose(result, expected, 1e-3);
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([4, NaN, 0]);
      const res = math.atan(a);
      test_util.expectArraysClose(res, [Math.atan(4), NaN, Math.atan(0)]);
    });
  };

  test_util.describeMathCPU('atan', [tests]);
  test_util.describeMathGPU('atan', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.sinh
{
  // TODO(nsthorat): Fix the precision problem here.
  const epsilon = 1e-1;

  const tests: MathTests = it => {
    it('basic', math => {
      const values = [1, -3, 2, 7, -4];
      const a = Array1D.new(values);
      const result = math.sinh(a);

      const expected = [];
      for (let i = 0; i < a.size; i++) {
        expected[i] = Math.sinh(values[i]);
      }
      test_util.expectArraysClose(result, expected, epsilon);
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([4, NaN, 0]);
      const res = math.sinh(a);
      test_util.expectArraysClose(
          res, [Math.sinh(4), NaN, Math.sinh(0)], epsilon);
    });
  };

  test_util.describeMathCPU('sinh', [tests]);
  test_util.describeMathGPU('sinh', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.cosh
{
  // TODO(nsthorat): Fix the precision problem here.
  const epsilon = 1e-1;

  const tests: MathTests = it => {
    it('basic', math => {
      const values = [1, -3, 2, -1, -4];
      const a = Array1D.new(values);
      const result = math.cosh(a);

      const expected = [];
      for (let i = 0; i < a.size; i++) {
        expected[i] = Math.cosh(values[i]);
      }

      // TODO(nsthorat): Fix the precision problem here.
      test_util.expectArraysClose(result, expected, epsilon);
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([4, NaN, 0]);
      const res = math.cosh(a);
      test_util.expectArraysClose(
          res, [Math.cosh(4), NaN, Math.cosh(0)], epsilon);
    });
  };

  test_util.describeMathCPU('cosh', [tests]);
  test_util.describeMathGPU('cosh', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.tanh
{
  const tests: MathTests = it => {
    it('basic', math => {
      const values = [1, -3, 2, 7, -4];
      const a = Array1D.new(values);
      const result = math.tanh(a);

      const expected = [];
      for (let i = 0; i < a.size; i++) {
        expected[i] = util.tanh(values[i]);
      }
      test_util.expectArraysClose(result, expected);
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([4, NaN, 0]);
      const res = math.tanh(a);
      test_util.expectArraysClose(res, [util.tanh(4), NaN, util.tanh(0)]);
    });
  };

  test_util.describeMathCPU('tanh', [tests]);
  test_util.describeMathGPU('tanh', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.leakyRelu
{
  const tests: MathTests = it => {
    it('basic', math => {
      const a = Array1D.new([0, 1, -2]);
      const result = math.leakyRelu(a);

      expect(result.shape).toEqual(a.shape);
      test_util.expectArraysClose(result, [0, 1, -0.4]);
    });

    it('propagates NaN', math => {
      const a = Array1D.new([0, 1, NaN]);
      const result = math.leakyRelu(a);

      expect(result.shape).toEqual(a.shape);
      test_util.expectArraysClose(result, [0, 1, NaN]);
    });
  };

  test_util.describeMathCPU('leakyRelu', [tests]);
  test_util.describeMathGPU('leakyRelu', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.elu
{
  const tests: MathTests = it => {
    it('calculate elu', math => {
      const a = Array1D.new([1, -1, 0]);
      const result = math.elu(a);

      expect(result.shape).toEqual(a.shape);
      test_util.expectArraysClose(result, [1, -0.6321, 0]);
    });

    it('elu propagates NaN', math => {
      const a = Array1D.new([1, NaN]);
      const result = math.elu(a);
      expect(result.shape).toEqual(a.shape);
      test_util.expectArraysClose(result, [1, NaN]);
    });
  };
  test_util.describeMathCPU('elu', [tests]);
  test_util.describeMathGPU('elu', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.selu
{
  const tests: MathTests = it => {
    it('calculate selu', math => {
      const a = Array1D.new([1, -1, 0]);
      const result = math.selu(a);

      expect(result.shape).toEqual(a.shape);
      test_util.expectArraysClose(result, [1.0507, -1.1113, 0]);
    });

    it('selu propagates NaN', math => {
      const a = Array1D.new([1, NaN]);
      const result = math.selu(a);
      expect(result.shape).toEqual(a.shape);
      test_util.expectArraysClose(result, [1.0507, NaN]);
    });
  };
  test_util.describeMathCPU('selu', [tests]);
  test_util.describeMathGPU('selu', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
