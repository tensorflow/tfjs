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

      test_util.expectArraysClose(
          result.getValues(), new Float32Array([1, 0, 0, 3, 0]));

      a.dispose();
    });

    it('does nothing to positive values', math => {
      const a = Scalar.new(1);

      const result = math.relu(a);

      test_util.expectNumbersClose(result.get(), 1);

      a.dispose();
    });

    it('sets negative values to 0', math => {
      const a = Scalar.new(-1);

      const result = math.relu(a);

      test_util.expectNumbersClose(result.get(), 0);

      a.dispose();
    });

    it('preserves zero values', math => {
      const a = Scalar.new(0);

      const result = math.relu(a);

      test_util.expectNumbersClose(result.get(), 0);

      a.dispose();
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([1, -2, 0, 3, -0.1, NaN]);

      const result = math.relu(a);

      test_util.expectArraysClose(
          result.getValues(), new Float32Array([1, 0, 0, 3, 0, NaN]));

      a.dispose();
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
      test_util.expectArraysClose(
          result.getValues(), new Float32Array([1, 2, 0, 3, 0.1]));

      a.dispose();
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([1, -2, 0, 3, -0.1, NaN]);
      const result = math.abs(a);
      test_util.expectArraysClose(
          result.getValues(), new Float32Array([1, 2, 0, 3, 0.1, NaN]));
      a.dispose();
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

      test_util.expectArraysClose(
          result.getValues(), new Float32Array([1, 0, 0, 1, 0]));

      a.dispose();
    });

    it('with 2d ndarray', math => {
      const a = Array2D.new([2, 2], [1, -5, -3, 4]);
      const result = math.step(a);

      expect(result.shape).toEqual([2, 2]);

      test_util.expectArraysClose(
          result.getValues(), new Float32Array([1, 0, 0, 1]));

      a.dispose();
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([1, -2, -.01, 3, NaN]);

      const result = math.step(a);

      test_util.expectArraysClose(
          result.getValues(), new Float32Array([1, 0, 0, 1, NaN]));

      a.dispose();
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

      test_util.expectArraysClose(
          result.getValues(), new Float32Array([-1, 3, -2, -7, 4]));

      a.dispose();
    });

    it('propagate NaNs', math => {
      const a = Array1D.new([1, -3, 2, 7, NaN]);

      const result = math.neg(a);

      const expected = [-1, 3, -2, -7, NaN];
      test_util.expectArraysClose(
          result.getValues(), new Float32Array(expected));

      a.dispose();
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

      const expected = new Float32Array(a.size);
      for (let i = 0; i < a.size; i++) {
        expected[i] = 1 / (1 + Math.exp(-values[i]));
      }
      test_util.expectArraysClose(result.getValues(), expected);

      a.dispose();
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([3, NaN]);

      const res = math.sigmoid(a).getValues();

      test_util.expectArraysClose(
          res, new Float32Array([1 / (1 + Math.exp(-3)), NaN]));

      a.dispose();
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

      expect(r.get(0)).toBeCloseTo(Math.sqrt(2));
      expect(r.get(1)).toBeCloseTo(Math.sqrt(4));

      a.dispose();
    });

    it('sqrt propagates NaNs', math => {
      const a = Array1D.new([1, NaN]);

      const r = math.sqrt(a).getValues();

      test_util.expectArraysClose(r, new Float32Array([Math.sqrt(1), NaN]));

      a.dispose();
    });
  };

  test_util.describeMathCPU('sqrt', [tests]);
  test_util.describeMathGPU('sqrt', [tests], [
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

      expect(r.get(0)).toBeCloseTo(Math.log(1));
      expect(r.get(1)).toBeCloseTo(Math.log(2));

      a.dispose();
    });

    it('log propagates NaNs', math => {
      const a = Array1D.new([1, NaN]);

      const r = math.log(a).getValues();

      test_util.expectArraysClose(r, new Float32Array([Math.log(1), NaN]));

      a.dispose();
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

      expect(r.get(0)).toBeCloseTo(2);
      expect(r.get(1)).toBeCloseTo(3);
      expect(r.get(2)).toBeCloseTo(-1);

      a.dispose();
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([1.5, NaN, -1.4]);

      const r = math.ceil(a).getValues();

      test_util.expectArraysClose(r, new Float32Array([2, NaN, -1]));

      a.dispose();
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

      expect(r.get(0)).toBeCloseTo(1);
      expect(r.get(1)).toBeCloseTo(2);
      expect(r.get(2)).toBeCloseTo(-2);

      a.dispose();
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([1.5, NaN, -1.4]);

      const r = math.floor(a).getValues();

      test_util.expectArraysClose(r, new Float32Array([1, NaN, -2]));

      a.dispose();
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

      expect(r.get(0)).toBeCloseTo(Math.exp(1));
      expect(r.get(1)).toBeCloseTo(Math.exp(2));
      expect(r.get(2)).toBeCloseTo(1);

      a.dispose();
    });

    it('exp propagates NaNs', math => {
      const a = Array1D.new([1, NaN, 0]);

      const r = math.exp(a).getValues();

      test_util.expectArraysClose(r, new Float32Array([Math.exp(1), NaN, 1]));

      a.dispose();
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

      const expected = new Float32Array(a.size);
      for (let i = 0; i < a.size; i++) {
        expected[i] = Math.sin(values[i]);
      }
      test_util.expectArraysClose(result.getValues(), expected);

      a.dispose();
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([4, NaN, 0]);

      const res = math.sin(a).getValues();

      const expected = [Math.sin(4), NaN, Math.sin(0)];
      test_util.expectArraysClose(res, new Float32Array(expected));

      a.dispose();
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

      const expected = new Float32Array(a.size);
      for (let i = 0; i < a.size; i++) {
        expected[i] = Math.cos(values[i]);
      }
      test_util.expectArraysClose(result.getValues(), expected);

      a.dispose();
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([4, NaN, 0]);

      const res = math.cos(a).getValues();

      const expected = [Math.cos(4), NaN, Math.cos(0)];
      test_util.expectArraysClose(res, new Float32Array(expected));

      a.dispose();
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

      const expected = new Float32Array(a.size);
      for (let i = 0; i < a.size; i++) {
        expected[i] = Math.tan(values[i]);
      }
      test_util.expectArraysClose(result.getValues(), expected, 1e-1);

      a.dispose();
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([4, NaN, 0]);

      const res = math.tan(a).getValues();

      const expected = [Math.tan(4), NaN, Math.tan(0)];
      test_util.expectArraysClose(res, new Float32Array(expected));

      a.dispose();
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

      const expected = new Float32Array(a.size);
      for (let i = 0; i < a.size; i++) {
        expected[i] = Math.asin(values[i]);
      }
      test_util.expectArraysClose(result.getValues(), expected, 1e-3);

      a.dispose();
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([4, NaN, 0]);

      const res = math.asin(a).getValues();

      const expected = [Math.asin(4), NaN, Math.asin(0)];
      test_util.expectArraysClose(res, new Float32Array(expected));

      a.dispose();
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

      const expected = new Float32Array(a.size);

      for (let i = 0; i < a.size; i++) {
        expected[i] = Math.acos(values[i]);
      }
      // TODO(nsthorat): Fix the precision with byte textures here.
      test_util.expectArraysClose(result.getValues(), expected, 1e-1);

      a.dispose();
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([4, NaN, 0]);
      const res = math.acos(a).getValues();
      const expected = [Math.acos(4), NaN, Math.acos(0)];
      test_util.expectArraysClose(res, new Float32Array(expected));
      a.dispose();
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

      const expected = new Float32Array(a.size);
      for (let i = 0; i < a.size; i++) {
        expected[i] = Math.atan(values[i]);
      }
      test_util.expectArraysClose(result.getValues(), expected, 1e-3);

      a.dispose();
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([4, NaN, 0]);

      const res = math.atan(a).getValues();

      const expected = [Math.atan(4), NaN, Math.atan(0)];
      test_util.expectArraysClose(res, new Float32Array(expected));

      a.dispose();
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

      const expected = new Float32Array(a.size);
      for (let i = 0; i < a.size; i++) {
        expected[i] = Math.sinh(values[i]);
      }

      test_util.expectArraysClose(result.getValues(), expected, epsilon);

      a.dispose();
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([4, NaN, 0]);

      const res = math.sinh(a).getValues();

      const expected = [Math.sinh(4), NaN, Math.sinh(0)];
      test_util.expectArraysClose(res, new Float32Array(expected), epsilon);

      a.dispose();
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

      const expected = new Float32Array(a.size);
      for (let i = 0; i < a.size; i++) {
        expected[i] = Math.cosh(values[i]);
      }

      // TODO(nsthorat): Fix the precision problem here.
      test_util.expectArraysClose(result.getValues(), expected, epsilon);

      a.dispose();
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([4, NaN, 0]);

      const res = math.cosh(a).getValues();

      const expected = [Math.cosh(4), NaN, Math.cosh(0)];
      test_util.expectArraysClose(res, new Float32Array(expected), epsilon);

      a.dispose();
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
      const expected = new Float32Array(a.size);
      for (let i = 0; i < a.size; i++) {
        expected[i] = util.tanh(values[i]);
      }
      test_util.expectArraysClose(result.getValues(), expected);

      a.dispose();
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([4, NaN, 0]);
      const res = math.tanh(a).getValues();
      const expected = [util.tanh(4), NaN, util.tanh(0)];
      test_util.expectArraysClose(res, new Float32Array(expected));
      a.dispose();
    });
  };

  test_util.describeMathCPU('tanh', [tests]);
  test_util.describeMathGPU('tanh', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
