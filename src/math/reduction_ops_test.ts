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

import {Array1D, Array2D, Array3D, Array4D, Scalar} from './ndarray';
import * as reduce_util from './reduce_util';

// math.min
{
  const tests: MathTests = it => {
    it('Array1D', math => {
      const a = Array1D.new([3, -1, 0, 100, -7, 2]);
      test_util.expectNumbersClose(math.min(a).get(), -7);
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([3, NaN, 2]);
      expect(math.min(a).get()).toEqual(NaN);
    });

    it('2D', math => {
      const a = Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
      test_util.expectNumbersClose(math.min(a).get(), -7);
    });

    it('2D axis=[0,1]', math => {
      const a = Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
      test_util.expectNumbersClose(math.min(a, [0, 1]).get(), -7);
    });

    it('2D, axis=0', math => {
      const a = Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
      const r = math.min(a, 0);

      expect(r.shape).toEqual([3]);
      test_util.expectArraysClose(r, [3, -7, 0]);
    });

    it('2D, axis=0, keepDims', math => {
      const a = Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
      const r = math.min(a, 0, true /* keepDims */);

      expect(r.shape).toEqual([1, 3]);
      test_util.expectArraysClose(r, [3, -7, 0]);
    });

    it('2D, axis=1 provided as a number', math => {
      const a = Array2D.new([2, 3], [3, 2, 5, 100, -7, 2]);
      const r = math.min(a, 1);
      test_util.expectArraysClose(r, [2, -7]);
    });

    it('2D, axis = -1 provided as a number', math => {
      const a = Array2D.new([2, 3], [3, 2, 5, 100, -7, 2]);
      const r = math.min(a, -1);
      test_util.expectArraysClose(r, [2, -7]);
    });

    it('2D, axis=[1]', math => {
      const a = Array2D.new([2, 3], [3, 2, 5, 100, -7, 2]);
      const r = math.min(a, [1]);
      test_util.expectArraysClose(r, [2, -7]);
    });
  };

  test_util.describeMathCPU('min', [tests]);
  test_util.describeMathGPU('min', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.max
{
  const tests: MathTests = it => {
    it('with one element dominating', math => {
      const a = Array1D.new([3, -1, 0, 100, -7, 2]);
      const r = math.max(a);
      test_util.expectNumbersClose(r.get(), 100);
    });

    it('with all elements being the same', math => {
      const a = Array1D.new([3, 3, 3]);
      const r = math.max(a);
      test_util.expectNumbersClose(r.get(), 3);
    });

    it('propagates NaNs', math => {
      expect(math.max(Array1D.new([3, NaN, 2])).get()).toEqual(NaN);
    });

    it('2D', math => {
      const a = Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
      test_util.expectNumbersClose(math.max(a).get(), 100);
    });

    it('2D axis=[0,1]', math => {
      const a = Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
      test_util.expectNumbersClose(math.max(a, [0, 1]).get(), 100);
    });

    it('2D, axis=0', math => {
      const a = Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
      const r = math.max(a, [0]);

      expect(r.shape).toEqual([3]);
      test_util.expectArraysClose(r, [100, -1, 2]);
    });

    it('2D, axis=0, keepDims', math => {
      const a = Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
      const r = math.max(a, [0], true /* keepDims */);

      expect(r.shape).toEqual([1, 3]);
      test_util.expectArraysClose(r, [100, -1, 2]);
    });

    it('2D, axis=1 provided as a number', math => {
      const a = Array2D.new([2, 3], [3, 2, 5, 100, -7, 2]);
      const r = math.max(a, 1);
      test_util.expectArraysClose(r, [5, 100]);
    });

    it('2D, axis = -1 provided as a number', math => {
      const a = Array2D.new([2, 3], [3, 2, 5, 100, -7, 2]);
      const r = math.max(a, -1);
      test_util.expectArraysClose(r, [5, 100]);
    });

    it('2D, axis=[1]', math => {
      const a = Array2D.new([2, 3], [3, 2, 5, 100, -7, 2]);
      const r = math.max(a, [1]);
      test_util.expectArraysClose(r, [5, 100]);
    });
  };

  test_util.describeMathCPU('max', [tests]);
  test_util.describeMathGPU('max', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.argmax
{
  const tests: MathTests = it => {
    it('Array1D', math => {
      const a = Array1D.new([1, 0, 3, 2]);
      const result = math.argMax(a);
      expect(result.dtype).toBe('int32');
      expect(result.get()).toBe(2);
    });

    it('one value', math => {
      const a = Array1D.new([10]);
      const result = math.argMax(a);
      expect(result.dtype).toBe('int32');
      expect(result.get()).toBe(0);
    });

    it('N > than parallelization threshold', math => {
      const n = reduce_util.PARALLELIZE_THRESHOLD * 2;
      const values = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        values[i] = i;
      }
      const a = Array1D.new(values);
      const result = math.argMax(a);
      expect(result.dtype).toBe('int32');
      expect(result.get()).toBe(n - 1);
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([5, 0, 3, NaN, 3]);
      const res = math.argMax(a);
      expect(res.dtype).toBe('int32');
      test_util.assertIsNan(res.get(), res.dtype);
    });

    it('2D, no axis specified', math => {
      const a = Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
      expect(math.argMax(a).get()).toBe(3);
    });

    it('2D, axis=0', math => {
      const a = Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
      const r = math.argMax(a, 0);

      expect(r.shape).toEqual([3]);
      expect(r.dtype).toBe('int32');
      test_util.expectArraysEqual(r, [1, 0, 1]);
    });

    it('2D, axis=1', math => {
      const a = Array2D.new([2, 3], [3, 2, 5, 100, -7, 2]);
      const r = math.argMax(a, 1);
      expect(r.dtype).toBe('int32');
      test_util.expectArraysEqual(r, [2, 0]);
    });

    it('2D, axis = -1', math => {
      const a = Array2D.new([2, 3], [3, 2, 5, 100, -7, 2]);
      const r = math.argMax(a, -1);
      expect(r.dtype).toBe('int32');
      test_util.expectArraysEqual(r, [2, 0]);
    });
  };

  test_util.describeMathCPU('argmax', [tests]);
  test_util.describeMathGPU('argmax', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.argmin
{
  const tests: MathTests = it => {
    it('Array1D', math => {
      const a = Array1D.new([1, 0, 3, 2]);
      const result = math.argMin(a);
      expect(result.get()).toBe(1);
    });

    it('one value', math => {
      const a = Array1D.new([10]);
      const result = math.argMin(a);
      expect(result.get()).toBe(0);
    });

    it('N > than parallelization threshold', math => {
      const n = reduce_util.PARALLELIZE_THRESHOLD * 2;
      const values = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        values[i] = n - i;
      }
      const a = Array1D.new(values);
      const result = math.argMin(a);
      expect(result.dtype).toBe('int32');
      expect(result.get()).toBe(n - 1);
    });

    it('Arg min propagates NaNs', math => {
      const a = Array1D.new([5, 0, NaN, 7, 3]);
      const res = math.argMin(a);
      test_util.assertIsNan(res.get(), res.dtype);
    });

    it('2D, no axis specified', math => {
      const a = Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
      expect(math.argMin(a).get()).toBe(4);
    });

    it('2D, axis=0', math => {
      const a = Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
      const r = math.argMin(a, 0);

      expect(r.shape).toEqual([3]);
      expect(r.dtype).toBe('int32');
      test_util.expectArraysEqual(r, [0, 1, 0]);
    });

    it('2D, axis=1', math => {
      const a = Array2D.new([2, 3], [3, 2, 5, 100, -7, -8]);
      const r = math.argMin(a, 1);
      test_util.expectArraysEqual(r, [1, 2]);
    });

    it('2D, axis = -1', math => {
      const a = Array2D.new([2, 3], [3, 2, 5, 100, -7, -8]);
      const r = math.argMin(a, -1);
      test_util.expectArraysEqual(r, [1, 2]);
    });
  };

  test_util.describeMathCPU('argmin', [tests]);
  test_util.describeMathGPU('argmin', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.argMaxEquals
{
  const tests: MathTests = it => {
    it('equals', math => {
      const a = Array1D.new([5, 0, 3, 7, 3]);
      const b = Array1D.new([-100.3, -20.0, -10.0, -5, -100]);
      const result = math.argMaxEquals(a, b);
      expect(result.get()).toBe(1);
    });

    it('not equals', math => {
      const a = Array1D.new([5, 0, 3, 1, 3]);
      const b = Array1D.new([-100.3, -20.0, -10.0, -5, 0]);
      const result = math.argMaxEquals(a, b);
      expect(result.get()).toBe(0);
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([0, 3, 1, 3]);
      const b = Array1D.new([NaN, -20.0, -10.0, -5]);
      const result = math.argMaxEquals(a, b);
      test_util.assertIsNan(result.get(), result.dtype);
    });

    it('throws when given arrays of different shape', math => {
      const a = Array1D.new([5, 0, 3, 7, 3, 10]);
      const b = Array1D.new([-100.3, -20.0, -10.0, -5, -100]);
      expect(() => math.argMaxEquals(a, b)).toThrowError();
    });
  };

  test_util.describeMathCPU('argMaxEquals', [tests]);
  test_util.describeMathGPU('argMaxEquals', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.logSumExp
{
  const tests: MathTests = it => {
    it('0', math => {
      const a = Scalar.new(0);
      const result = math.logSumExp(a);
      test_util.expectNumbersClose(result.get(), 0);
    });

    it('basic', math => {
      const a = Array1D.new([1, 2, -3]);
      const result = math.logSumExp(a);

      test_util.expectNumbersClose(
          result.get(), Math.log(Math.exp(1) + Math.exp(2) + Math.exp(-3)));
    });

    it('propagates NaNs', math => {
      const a = Array1D.new([1, 2, NaN]);
      const result = math.logSumExp(a);
      expect(result.get()).toEqual(NaN);
    });

    it('axes=0 in 2D array', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const r = math.logSumExp(a, [0]);

      expect(r.shape).toEqual([2]);
      const expected = [
        Math.log(Math.exp(1) + Math.exp(3) + Math.exp(0)),
        Math.log(Math.exp(2) + Math.exp(0) + Math.exp(1))
      ];
      test_util.expectArraysClose(r, expected);
    });

    it('axes=0 in 2D array, keepDims', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const r = math.logSumExp(a, [0], true /* keepDims */);

      expect(r.shape).toEqual([1, 2]);
      const expected = [
        Math.log(Math.exp(1) + Math.exp(3) + Math.exp(0)),
        Math.log(Math.exp(2) + Math.exp(0) + Math.exp(1))
      ];
      test_util.expectArraysClose(r, expected);
    });

    it('axes=1 in 2D array', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const res = math.logSumExp(a, [1]);

      expect(res.shape).toEqual([3]);
      const expected = [
        Math.log(Math.exp(1) + Math.exp(2)),
        Math.log(Math.exp(3) + Math.exp(0)),
        Math.log(Math.exp(0) + Math.exp(1)),
      ];
      test_util.expectArraysClose(res, expected);
    });

    it('axes = -1 in 2D array', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const res = math.logSumExp(a, -1);

      expect(res.shape).toEqual([3]);
      const expected = [
        Math.log(Math.exp(1) + Math.exp(2)),
        Math.log(Math.exp(3) + Math.exp(0)),
        Math.log(Math.exp(0) + Math.exp(1)),
      ];
      test_util.expectArraysClose(res, expected);
    });

    it('2D, axes=1 provided as a single digit', math => {
      const a = Array2D.new([2, 3], [1, 2, 3, 0, 0, 1]);
      const res = math.logSumExp(a, 1);

      expect(res.shape).toEqual([2]);
      const expected = [
        Math.log(Math.exp(1) + Math.exp(2) + Math.exp(3)),
        Math.log(Math.exp(0) + Math.exp(0) + Math.exp(1))
      ];
      test_util.expectArraysClose(res, expected);
    });

    it('axes=0,1 in 2D array', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const res = math.logSumExp(a, [0, 1]);

      expect(res.shape).toEqual([]);
      const expected = [Math.log(
          Math.exp(1) + Math.exp(2) + Math.exp(3) + Math.exp(0) + Math.exp(0) +
          Math.exp(1))];
      test_util.expectArraysClose(res, expected);
    });
  };

  test_util.describeMathCPU('logSumExp', [tests]);
  test_util.describeMathGPU('logSumExp', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.sum
{
  const tests: MathTests = it => {
    it('basic', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const result = math.sum(a);
      test_util.expectNumbersClose(result.get(), 7);
    });

    it('propagates NaNs', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, NaN, 0, 1]);
      expect(math.sum(a).get()).toEqual(NaN);
    });

    it('sum over dtype int32', math => {
      const a = Array1D.new([1, 5, 7, 3], 'int32');
      const sum = math.sum(a);
      expect(sum.get()).toBe(16);
    });

    it('sum over dtype bool', math => {
      const a = Array1D.new([true, false, false, true, true], 'bool');
      const sum = math.sum(a);
      expect(sum.get()).toBe(3);
    });

    it('sums all values in 2D array with keep dim', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const res = math.sum(a, null, true /* keepDims */);

      expect(res.shape).toEqual([1, 1]);
      test_util.expectArraysClose(res, [7]);
    });

    it('sums across axis=0 in 2D array', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const res = math.sum(a, [0]);

      expect(res.shape).toEqual([2]);
      test_util.expectArraysClose(res, [4, 3]);
    });

    it('sums across axis=0 in 2D array, keepDims', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const res = math.sum(a, [0], true /* keepDims */);

      expect(res.shape).toEqual([1, 2]);
      test_util.expectArraysClose(res, [4, 3]);
    });

    it('sums across axis=1 in 2D array', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const res = math.sum(a, [1]);

      expect(res.shape).toEqual([3]);
      test_util.expectArraysClose(res, [3, 3, 1]);
    });

    it('2D, axis=1 provided as number', math => {
      const a = Array2D.new([2, 3], [1, 2, 3, 0, 0, 1]);
      const res = math.sum(a, 1);

      expect(res.shape).toEqual([2]);
      test_util.expectArraysClose(res, [6, 1]);
    });

    it('2D, axis = -1 provided as number', math => {
      const a = Array2D.new([2, 3], [1, 2, 3, 0, 0, 1]);
      const res = math.sum(a, -1);

      expect(res.shape).toEqual([2]);
      test_util.expectArraysClose(res, [6, 1]);
    });

    it('sums across axis=0,1 in 2D array', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const res = math.sum(a, [0, 1]);

      expect(res.shape).toEqual([]);
      test_util.expectArraysClose(res, [7]);
    });

    it('2D, axis=[-1,-2] in 2D array', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const res = math.sum(a, [-1, -2]);

      expect(res.shape).toEqual([]);
      test_util.expectArraysClose(res, [7]);
    });

    it('gradients: sum(2d)', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const dy = Scalar.new(10);

      const gradients = math.vjp(() => math.sum(a), a, dy);

      expect(gradients.shape).toEqual(a.shape);
      expect(gradients.dtype).toEqual('float32');
      test_util.expectArraysClose(gradients, [10, 10, 10, 10, 10, 10], 1e-1);
    });

    it('gradients: sum(2d, axis=0)', math => {
      const a = Array2D.new([3, 2], [[1, 2], [3, 0], [0, 1]]);
      const dy = Array1D.new([10, 20]);
      const axis = 0;

      const gradients = math.vjp(() => math.sum(a, axis), a, dy);

      expect(gradients.shape).toEqual(a.shape);
      expect(gradients.dtype).toEqual('float32');
      test_util.expectArraysClose(gradients, [10, 20, 10, 20, 10, 20], 1e-1);
    });

    it('gradients: sum(2d, axis=1)', math => {
      const a = Array2D.new([3, 2], [[1, 2], [3, 0], [0, 1]]);
      const dy = Array1D.new([10, 20, 30]);
      const axis = 1;

      const gradients = math.vjp(() => math.sum(a, axis), a, dy);

      expect(gradients.shape).toEqual(a.shape);
      expect(gradients.dtype).toEqual('float32');
      test_util.expectArraysClose(gradients, [10, 10, 20, 20, 30, 30], 1e-1);
    });
  };

  test_util.describeMathCPU('sum', [tests]);
  test_util.describeMathGPU('sum', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.mean
{
  const tests: MathTests = it => {
    it('basic', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const r = math.mean(a);

      expect(r.dtype).toBe('float32');
      test_util.expectNumbersClose(r.get(), 7 / 6);
    });

    it('propagates NaNs', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, NaN, 0, 1]);
      const r = math.mean(a);

      expect(r.dtype).toBe('float32');
      expect(r.get()).toEqual(NaN);
    });

    it('mean(int32) => float32', math => {
      const a = Array1D.new([1, 5, 7, 3], 'int32');
      const r = math.mean(a);

      expect(r.dtype).toBe('float32');
      test_util.expectNumbersClose(r.get(), 4);
    });

    it('mean(bool) => float32', math => {
      const a = Array1D.new([true, false, false, true, true], 'bool');
      const r = math.mean(a);

      expect(r.dtype).toBe('float32');
      test_util.expectNumbersClose(r.get(), 3 / 5);
    });

    it('2D array with keep dim', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const res = math.mean(a, null, true /* keepDims */);

      expect(res.shape).toEqual([1, 1]);
      expect(res.dtype).toBe('float32');
      test_util.expectArraysClose(res, [7 / 6]);
    });

    it('axis=0 in 2D array', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const res = math.mean(a, [0]);

      expect(res.shape).toEqual([2]);
      expect(res.dtype).toBe('float32');
      test_util.expectArraysClose(res, [4 / 3, 1]);
    });

    it('axis=0 in 2D array, keepDims', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const res = math.mean(a, [0], true /* keepDims */);

      expect(res.shape).toEqual([1, 2]);
      expect(res.dtype).toBe('float32');
      test_util.expectArraysClose(res, [4 / 3, 1]);
    });

    it('axis=1 in 2D array', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const res = math.mean(a, [1]);

      expect(res.dtype).toBe('float32');
      expect(res.shape).toEqual([3]);
      test_util.expectArraysClose(res, [1.5, 1.5, 0.5]);
    });

    it('axis = -1 in 2D array', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const res = math.mean(a, [-1]);

      expect(res.dtype).toBe('float32');
      expect(res.shape).toEqual([3]);
      test_util.expectArraysClose(res, [1.5, 1.5, 0.5]);
    });

    it('2D, axis=1 provided as number', math => {
      const a = Array2D.new([2, 3], [1, 2, 3, 0, 0, 1]);
      const res = math.mean(a, 1);

      expect(res.shape).toEqual([2]);
      expect(res.dtype).toBe('float32');
      test_util.expectArraysClose(res, [2, 1 / 3]);
    });

    it('axis=0,1 in 2D array', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const res = math.mean(a, [0, 1]);

      expect(res.shape).toEqual([]);
      expect(res.dtype).toBe('float32');
      test_util.expectArraysClose(res, [7 / 6]);
    });

    it('gradients', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const dy = Scalar.new(1.5);

      const vjp = math.vjp(() => math.mean(a), a, dy);

      expect(vjp.shape).toEqual(a.shape);
      test_util.expectArraysClose(vjp, [
        dy.get() / a.size, dy.get() / a.size, dy.get() / a.size,
        dy.get() / a.size, dy.get() / a.size, dy.get() / a.size
      ]);
    });

    it('gradients throws for defined axis', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const dy = Scalar.new(1.5);

      expect(() => math.vjp(() => math.mean(a, 1), a, dy)).toThrowError();
    });
  };

  test_util.describeMathCPU('mean', [tests]);
  test_util.describeMathGPU('mean', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.moments
{
  const tests: MathTests = it => {
    it('basic', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const {mean, variance} = math.moments(a);

      expect(mean.dtype).toBe('float32');
      expect(variance.dtype).toBe('float32');
      test_util.expectNumbersClose(mean.get(), 7 / 6);
      test_util.expectNumbersClose(variance.get(), 1.1389);
    });

    it('propagates NaNs', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, NaN, 0, 1]);
      const {mean, variance} = math.moments(a);

      expect(mean.dtype).toBe('float32');
      expect(variance.dtype).toBe('float32');
      expect(mean.get()).toEqual(NaN);
      expect(variance.get()).toEqual(NaN);
    });

    it('moments(int32) => float32', math => {
      const a = Array1D.new([1, 5, 7, 3], 'int32');
      const {mean, variance} = math.moments(a);

      expect(mean.dtype).toBe('float32');
      expect(variance.dtype).toBe('float32');
      test_util.expectNumbersClose(mean.get(), 4);
      test_util.expectNumbersClose(variance.get(), 5);
    });

    it('moments(bool) => float32', math => {
      const a = Array1D.new([true, false, false, true, true], 'bool');
      const {mean, variance} = math.moments(a);

      expect(mean.dtype).toBe('float32');
      expect(variance.dtype).toBe('float32');
      test_util.expectNumbersClose(mean.get(), 3 / 5);
      test_util.expectNumbersClose(variance.get(), 0.23999998);
    });

    it('2D array with keep dim', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const {mean, variance} = math.moments(a, null, true /* keepDims */);

      expect(mean.shape).toEqual([1, 1]);
      expect(mean.dtype).toBe('float32');
      expect(variance.shape).toEqual([1, 1]);
      expect(variance.dtype).toBe('float32');
      test_util.expectArraysClose(mean, [7 / 6]);
      test_util.expectArraysClose(variance, [1.138889]);
    });

    it('axis=0 in 2D array', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const {mean, variance} = math.moments(a, [0]);

      expect(mean.shape).toEqual([2]);
      expect(mean.dtype).toBe('float32');
      expect(variance.shape).toEqual([2]);
      expect(variance.dtype).toBe('float32');
      test_util.expectArraysClose(mean, [4 / 3, 1]);
      test_util.expectArraysClose(variance, [1.556, 2 / 3]);
    });

    it('axis=1 in 2D array', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const {mean, variance} = math.moments(a, [1]);

      expect(mean.dtype).toBe('float32');
      expect(mean.shape).toEqual([3]);
      expect(variance.dtype).toBe('float32');
      expect(variance.shape).toEqual([3]);
      test_util.expectArraysClose(mean, [1.5, 1.5, 0.5]);
      test_util.expectArraysClose(variance, [0.25, 2.25, 0.25]);
    });

    it('2D, axis=1 provided as number', math => {
      const a = Array2D.new([2, 3], [1, 2, 3, 0, 0, 1]);
      const {mean, variance} = math.moments(a, 1);

      expect(mean.shape).toEqual([2]);
      expect(mean.dtype).toBe('float32');
      expect(variance.shape).toEqual([2]);
      expect(variance.dtype).toBe('float32');
      test_util.expectArraysClose(mean, [2, 1 / 3]);
      test_util.expectArraysClose(variance, [2 / 3, 0.222]);
    });

    it('2D, axis=-1 provided as number', math => {
      const a = Array2D.new([2, 3], [1, 2, 3, 0, 0, 1]);
      const {mean, variance} = math.moments(a, -1);

      expect(mean.shape).toEqual([2]);
      expect(mean.dtype).toBe('float32');
      expect(variance.shape).toEqual([2]);
      expect(variance.dtype).toBe('float32');
      test_util.expectArraysClose(mean, [2, 1 / 3]);
      test_util.expectArraysClose(variance, [2 / 3, 0.222]);
    });

    it('axis=0,1 in 2D array', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const {mean, variance} = math.moments(a, [0, 1]);

      expect(mean.shape).toEqual([]);
      expect(mean.dtype).toBe('float32');
      expect(variance.shape).toEqual([]);
      expect(variance.dtype).toBe('float32');
      test_util.expectArraysClose(mean, [7 / 6]);
      test_util.expectArraysClose(variance, [1.1389]);
    });
  };

  test_util.describeMathCPU('moments', [tests]);
  test_util.describeMathGPU('moments', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.norm
{
  const tests: MathTests = it => {
    it('scalar norm', math => {
      const a = Scalar.new(-22.0);
      const norm = math.norm(a);

      expect(norm.dtype).toBe('float32');
      test_util.expectNumbersClose(norm.get(), 22);
    });

    it('vector inf norm', math => {
      const a = Array1D.new([1, -2, 3, -4]);
      const norm = math.norm(a, Infinity);

      expect(norm.dtype).toBe('float32');
      test_util.expectNumbersClose(norm.get(), 4);
    });

    it('vector -inf norm', math => {
      const a = Array1D.new([1, -2, 3, -4]);
      const norm = math.norm(a, -Infinity);

      expect(norm.dtype).toBe('float32');
      test_util.expectNumbersClose(norm.get(), 1);
    });

    it('vector 1 norm', math => {
      const a = Array1D.new([1, -2, 3, -4]);
      const norm = math.norm(a, 1);

      expect(norm.dtype).toBe('float32');
      test_util.expectNumbersClose(norm.get(), 10);
    });

    it('vector euclidean norm', math => {
      const a = Array1D.new([1, -2, 3, -4]);
      const norm = math.norm(a, 'euclidean');

      expect(norm.dtype).toBe('float32');
      test_util.expectNumbersClose(norm.get(), 5.4772);
    });

    it('vector 2-norm', math => {
      const a = Array1D.new([1, -2, 3, -4]);
      const norm = math.norm(a, 2);

      expect(norm.dtype).toBe('float32');
      test_util.expectNumbersClose(norm.get(), 5.4772);
    });

    it('vector >2-norm to throw error', math => {
      const a = Array1D.new([1, -2, 3, -4]);
      expect(() => math.norm(a, 3)).toThrowError();
    });

    it('matrix inf norm', math => {
      const a = Array2D.new([3, 2], [1, 2, -3, 1, 0, 1]);
      const norm = math.norm(a, Infinity, [0, 1]);

      expect(norm.dtype).toBe('float32');
      test_util.expectNumbersClose(norm.get(), 4);
    });

    it('matrix -inf norm', math => {
      const a = Array2D.new([3, 2], [1, 2, -3, 1, 0, 1]);
      const norm = math.norm(a, -Infinity, [0, 1]);

      expect(norm.dtype).toBe('float32');
      test_util.expectNumbersClose(norm.get(), 1);
    });

    it('matrix 1 norm', math => {
      const a = Array2D.new([3, 2], [1, 2, -3, 1, 1, 1]);
      const norm = math.norm(a, 1, [0, 1]);

      expect(norm.dtype).toBe('float32');
      test_util.expectNumbersClose(norm.get(), 5);
    });

    it('matrix euclidean norm', math => {
      const a = Array2D.new([3, 2], [1, 2, -3, 1, 1, 1]);
      const norm = math.norm(a, 'euclidean', [0, 1]);

      expect(norm.dtype).toBe('float32');
      test_util.expectNumbersClose(norm.get(), 4.123);
    });

    it('matrix fro norm', math => {
      const a = Array2D.new([3, 2], [1, 2, -3, 1, 1, 1]);
      const norm = math.norm(a, 'fro', [0, 1]);

      expect(norm.dtype).toBe('float32');
      test_util.expectNumbersClose(norm.get(), 4.123);
    });

    it('matrix other norm to throw error', math => {
      const a = Array2D.new([3, 2], [1, 2, -3, 1, 1, 1]);
      expect(() => math.norm(a, 2, [0, 1])).toThrowError();
    });

    it('propagates NaNs for norm', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, NaN, 0, 1]);
      const norm = math.norm(a, Infinity, [0, 1]);

      expect(norm.dtype).toBe('float32');
      expect(norm.get()).toEqual(NaN);
    });

    it('axis=null in 2D array norm', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const norm = math.norm(a, Infinity);

      expect(norm.shape).toEqual([]);
      expect(norm.dtype).toBe('float32');
      test_util.expectArraysClose(norm, [3]);
    });

    it('2D array norm with keep dim', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const norm = math.norm(a, Infinity, null, true /* keepDims */);

      expect(norm.shape).toEqual([1, 1]);
      expect(norm.dtype).toBe('float32');
      test_util.expectArraysClose(norm, [3]);
    });

    it('axis=0 in 2D array norm', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const norm = math.norm(a, Infinity, [0]);

      expect(norm.shape).toEqual([2]);
      expect(norm.dtype).toBe('float32');
      test_util.expectArraysClose(norm, [3, 2]);
    });

    it('axis=1 in 2D array norm', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const norm = math.norm(a, Infinity, [1]);

      expect(norm.dtype).toBe('float32');
      expect(norm.shape).toEqual([3]);
      test_util.expectArraysClose(norm, [2, 3, 1]);
    });

    it('axis=1 keepDims in 2D array norm', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const norm = math.norm(a, Infinity, [1], true);

      expect(norm.dtype).toBe('float32');
      expect(norm.shape).toEqual([3, 1]);
      test_util.expectArraysClose(norm, [2, 3, 1]);
    });

    it('2D norm with axis=1 provided as number', math => {
      const a = Array2D.new([2, 3], [1, 2, 3, 0, 0, 1]);
      const norm = math.norm(a, Infinity, 1);

      expect(norm.shape).toEqual([2]);
      expect(norm.dtype).toBe('float32');
      test_util.expectArraysClose(norm, [3, 1]);
    });

    it('axis=0,1 in 2D array norm', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const norm = math.norm(a, Infinity, [0, 1]);

      expect(norm.shape).toEqual([]);
      expect(norm.dtype).toBe('float32');
      test_util.expectArraysClose(norm, [3]);
    });

    it('axis=0,1 keepDims in 2D array norm', math => {
      const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
      const norm = math.norm(a, Infinity, [0, 1], true);

      expect(norm.shape).toEqual([1, 1]);
      expect(norm.dtype).toBe('float32');
      test_util.expectArraysClose(norm, [3]);
    });

    it('3D norm axis=0,1, matrix inf norm', math => {
      const a = Array3D.new([3, 2, 1], [1, 2, -3, 1, 0, 1]);
      const norm = math.norm(a, Infinity, [0, 1]);

      expect(norm.shape).toEqual([1]);
      expect(norm.dtype).toBe('float32');
      test_util.expectArraysClose(norm, [4]);
    });

    it('axis=0,1 keepDims in 3D array norm', math => {
      const a = Array3D.new([3, 2, 1], [1, 2, 3, 0, 0, 1]);
      const norm = math.norm(a, Infinity, [0, 1], true);

      expect(norm.shape).toEqual([1, 1, 1]);
      expect(norm.dtype).toBe('float32');
      test_util.expectArraysClose(norm, [3]);
    });

    it('axis=0,1 keepDims in 3D array norm', math => {
      const a = Array3D.new([3, 2, 2], [1, 2, 3, 0, 0, 1, 1, 2, 3, 0, 0, 1]);
      const norm = math.norm(a, Infinity, [0, 1], true);

      expect(norm.shape).toEqual([1, 1, 2]);
      expect(norm.dtype).toBe('float32');
      test_util.expectArraysClose(norm, [4, 3]);
    });

    it('axis=null in 3D array norm', math => {
      const a = Array3D.new([3, 2, 1], [1, 2, 3, 0, 0, 1]);
      const norm = math.norm(a, Infinity);

      expect(norm.shape).toEqual([]);
      expect(norm.dtype).toBe('float32');
      test_util.expectArraysClose(norm, [3]);
    });

    it('axis=null in 4D array norm', math => {
      const a = Array4D.new([3, 2, 1, 1], [1, 2, 3, 0, 0, 1]);
      const norm = math.norm(a, Infinity);

      expect(norm.shape).toEqual([]);
      expect(norm.dtype).toBe('float32');
      test_util.expectArraysClose(norm, [3]);
    });

    it('axis=0,1 in 4D array norm', math => {
      const a = Array4D.new([3, 2, 2, 2], [
        1, 2, 3, 0, 0, 1, 1, 2, 3, 0, 0, 1,
        1, 2, 3, 0, 0, 1, 1, 2, 3, 0, 0, 1
      ]);
      const norm = math.norm(a, Infinity, [0, 1]);

      expect(norm.shape).toEqual([2, 2]);
      expect(norm.dtype).toBe('float32');
      test_util.expectArraysClose(norm, [4, 3, 4, 3]);
    });

    it('axis=0,1 in 4D array norm', math => {
      const a = Array4D.new([3, 2, 2, 2], [
        1, 2, 3, 0, 0, 1, 1, 2, 3, 0, 0, 1,
        1, 2, 3, 0, 0, 1, 1, 2, 3, 0, 0, 1
      ]);
      const norm = math.norm(a, Infinity, [0, 1], true);

      expect(norm.shape).toEqual([1, 1, 2, 2]);
      expect(norm.dtype).toBe('float32');
      test_util.expectArraysClose(norm, [4, 3, 4, 3]);
    });
  };

  test_util.describeMathCPU('norm', [tests]);
  test_util.describeMathGPU('norm', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
