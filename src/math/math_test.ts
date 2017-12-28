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

import {MatrixOrientation} from './backends/types/matmul';
import {Array1D, Array2D, Array3D, Scalar} from './ndarray';

// math.scope
{
  const gpuTests: MathTests = it => {
    it('scope returns NDArray', async math => {
      await math.scope(async () => {
        const a = Array1D.new([1, 2, 3]);
        let b = Array1D.new([0, 0, 0]);

        expect(math.getNumArrays()).toBe(2);
        await math.scope(async () => {
          const result = math.scope(() => {
            b = math.add(a, b) as Array1D;
            b = math.add(a, b) as Array1D;
            b = math.add(a, b) as Array1D;
            return math.add(a, b);
          });

          // result is new. All intermediates should be disposed.
          expect(math.getNumArrays()).toBe(2 + 1);
          test_util.expectArraysClose(result, [4, 8, 12]);
        });

        // a, b are still here, result should be disposed.
        expect(math.getNumArrays()).toBe(2);
      });

      expect(math.getNumArrays()).toBe(0);
    });

    it('multiple disposes does not affect num arrays', math => {
      expect(math.getNumArrays()).toBe(0);
      const a = Array1D.new([1, 2, 3]);
      const b = Array1D.new([1, 2, 3]);
      expect(math.getNumArrays()).toBe(2);
      a.dispose();
      a.dispose();
      expect(math.getNumArrays()).toBe(1);
      b.dispose();
      expect(math.getNumArrays()).toBe(0);
    });

    it('scope returns NDArray[]', async math => {
      const a = Array1D.new([1, 2, 3]);
      const b = Array1D.new([0, -1, 1]);
      expect(math.getNumArrays()).toBe(2);

      await math.scope(async () => {
        const result = math.scope(() => {
          math.add(a, b);
          return [math.add(a, b), math.subtract(a, b)];
        });

        // the 2 results are new. All intermediates should be disposed.
        expect(math.getNumArrays()).toBe(4);
        test_util.expectArraysClose(result[0], [1, 1, 4]);
        test_util.expectArraysClose(result[1], [1, 3, 2]);
        expect(math.getNumArrays()).toBe(4);
      });

      // the 2 results should be disposed.
      expect(math.getNumArrays()).toBe(2);
      a.dispose();
      b.dispose();
      expect(math.getNumArrays()).toBe(0);
    });

    it('basic scope usage without return', math => {
      const a = Array1D.new([1, 2, 3]);
      let b = Array1D.new([0, 0, 0]);

      expect(math.getNumArrays()).toBe(2);

      math.scope(() => {
        b = math.add(a, b) as Array1D;
        b = math.add(a, b) as Array1D;
        b = math.add(a, b) as Array1D;
        math.add(a, b);
      });

      // all intermediates should be disposed.
      expect(math.getNumArrays()).toBe(2);
    });

    it('scope returns Promise<NDArray>', async math => {
      const a = Array1D.new([1, 2, 3]);
      const b = Array1D.new([0, 0, 0]);

      expect(math.getNumArrays()).toBe(2);

      await math.scope(async () => {
        const result = math.scope(() => {
          let c = math.add(a, b) as Array1D;
          c = math.add(a, c) as Array1D;
          c = math.add(a, c) as Array1D;
          return math.add(a, c);
        });

        // result is new. All intermediates should be disposed.
        expect(math.getNumArrays()).toBe(3);
        test_util.expectArraysClose(result, [4, 8, 12]);
      });

      // result should be disposed. a and b are still allocated.
      expect(math.getNumArrays()).toBe(2);
      a.dispose();
      b.dispose();
      expect(math.getNumArrays()).toBe(0);
    });

    it('nested scope usage', async math => {
      const a = Array1D.new([1, 2, 3]);
      let b = Array1D.new([0, 0, 0]);

      expect(math.getNumArrays()).toBe(2);

      await math.scope(async () => {
        const result = math.scope(() => {
          b = math.add(a, b) as Array1D;
          b = math.scope(() => {
            b = math.scope(() => {
              return math.add(a, b) as Array1D;
            });
            // original a, b, and two intermediates.
            expect(math.getNumArrays()).toBe(4);

            math.scope(() => {
              math.add(a, b);
            });
            // All the intermediates should be cleaned up.
            expect(math.getNumArrays()).toBe(4);

            return math.add(a, b) as Array1D;
          });
          expect(math.getNumArrays()).toBe(4);

          return math.add(a, b) as Array1D;
        });

        expect(math.getNumArrays()).toBe(3);
        test_util.expectArraysClose(result, [4, 8, 12]);
      });
      expect(math.getNumArrays()).toBe(2);
    });
  };

  test_util.describeMathGPU('scope', [gpuTests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// debug mode
{
  const gpuTests: MathTests = it => {
    it('debug mode does not error when no nans', math => {
      math.enableDebugMode();
      const a = Array1D.new([2, -1, 0, 3]);
      const res = math.relu(a);
      test_util.expectArraysClose(res, [2, 0, 0, 3]);
    });

    it('debug mode errors when there are nans, float32', math => {
      math.enableDebugMode();
      const a = Array1D.new([2, NaN]);
      const f = () => math.relu(a);
      expect(f).toThrowError();
    });

    it('debug mode errors when there are nans, int32', math => {
      math.enableDebugMode();
      const a = Array1D.new([2, util.NAN_INT32], 'int32');
      const f = () => math.relu(a);
      expect(f).toThrowError();
    });

    it('debug mode errors when there are nans, bool', math => {
      math.enableDebugMode();
      const a = Array1D.new([1, util.NAN_BOOL], 'bool');
      const f = () => math.relu(a);
      expect(f).toThrowError();
    });

    it('no errors where there are nans, and debug mode is disabled', math => {
      const a = Array1D.new([2, NaN]);
      const res = math.relu(a);
      test_util.expectArraysClose(res, [2, NaN]);
    });
  };

  test_util.describeMathCPU('debug mode', [gpuTests]);
  test_util.describeMathGPU('debug mode', [gpuTests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// fromPixels & math
{
  const tests: MathTests = it => {
    it('debug mode does not error when no nans', math => {
      const pixels = new ImageData(2, 2);
      for (let i = 0; i < 8; i++) {
        pixels.data[i] = 100;
      }
      for (let i = 8; i < 16; i++) {
        pixels.data[i] = 250;
      }

      const a = Array3D.fromPixels(pixels, 4);
      const b = Scalar.new(20, 'int32');

      const res = math.add(a, b);

      test_util.expectArraysEqual(res, [
        120, 120, 120, 120, 120, 120, 120, 120, 270, 270, 270, 270, 270, 270,
        270, 270
      ]);
    });
  };

  test_util.describeMathGPU('fromPixels + math', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// gradients integration tests
{
  const tests: MathTests = it => {
    it('matmul + relu', math => {
      const a = Array2D.new([2, 3], [-1, 2, -3, 10, -20, 30]);
      const b = Array2D.new([3, 2], [2, -3, 4, -1, 2, -3]);

      const gradients = math.gradients(() => {
        // m = dot(a, b)
        // y = relu(m)
        // e = sum(y)
        const m = math.matMul(a, b);
        const y = math.relu(m);
        return math.sum(y);
      }, {a, b});

      // de/dy = 1
      // dy/dm = step(m)
      // de/dm = de/dy * dy/dm = step(m)
      const dedm = math.step(math.matMul(a, b));

      // de/da = dot(de/dy, bT)
      expect(gradients.a.shape).toEqual(a.shape);
      test_util.expectArraysClose(
          gradients.a,
          math.matMul(
              dedm, b, MatrixOrientation.REGULAR,
              MatrixOrientation.TRANSPOSED));

      // de/db = dot(aT, de/dy)
      expect(gradients.b.shape).toEqual(b.shape);
      test_util.expectArraysClose(
          gradients.b,
          math.matMul(
              a, dedm, MatrixOrientation.TRANSPOSED,
              MatrixOrientation.REGULAR));
    });

    it('Throws if y is not a scalar', math => {
      const a = Array2D.new([2, 3], [-1, 2, -3, 10, -20, 30]);
      const b = Array2D.new([3, 2], [2, -3, 4, -1, 2, -3]);

      expect(
          // tslint:disable-next-line:no-any
          () => math.valueAndGradients(() => math.matMul(a, b) as any, {a, b}))
          .toThrowError();
    });

    test_util.describeMathCPU('gradients', [tests]);
    test_util.describeMathGPU('gradients', [tests], [
      {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
      {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
      {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
    ]);
  };
}

// valueAndGradients integration tests
{
  const tests: MathTests = it => {
    it('matmul + relu', math => {
      const a = Array2D.new([2, 3], [-1, 2, -3, 10, -20, 30]);
      const b = Array2D.new([3, 2], [2, -3, 4, -1, 2, -3]);

      const {value, gradients} = math.valueAndGradients(() => {
        // m = dot(a, b)
        // y = relu(m)
        // e = sum(y)
        const m = math.matMul(a, b);
        const y = math.relu(m);
        return math.sum(y);
      }, {a, b});

      test_util.expectNumbersClose(value.get(), 10, 1e-1);

      // de/dy = 1
      // dy/dm = step(m)
      // de/dm = de/dy * dy/dm = step(m)
      const dedm = math.step(math.matMul(a, b));

      // de/da = dot(de/dy, bT)
      test_util.expectArraysClose(
          gradients.a,
          math.matMul(
              dedm, b, MatrixOrientation.REGULAR,
              MatrixOrientation.TRANSPOSED));

      // de/db = dot(aT, de/dy)
      test_util.expectArraysClose(
          gradients.b,
          math.matMul(
              a, dedm, MatrixOrientation.TRANSPOSED,
              MatrixOrientation.REGULAR));
    });

    it('Throws is y is not a scalar', math => {
      const a = Array2D.new([2, 3], [-1, 2, -3, 10, -20, 30]);
      const b = Array2D.new([3, 2], [2, -3, 4, -1, 2, -3]);

      expect(
          // tslint:disable-next-line:no-any
          () => math.valueAndGradients(() => math.matMul(a, b) as any, {a, b}))
          .toThrowError();
    });

    it('matmul + relu + inner scope', math => {
      const a = Array2D.new([2, 3], [-1, 2, -3, 10, -20, 30]);
      const b = Array2D.new([3, 2], [2, -3, 4, -1, 2, -3]);

      const {value, gradients} = math.valueAndGradients(() => {
        // m = dot(a, b)
        // y = relu(m)
        // e = sum(y)
        const m = math.matMul(a, b);
        return math.scope(() => {
          const y = math.relu(m);
          return math.sum(y);
        });
      }, {a, b});

      test_util.expectNumbersClose(value.get(), 10, 1e-1);

      // de/dy = 1
      // dy/dm = step(m)
      // de/dm = de/dy * dy/dm = step(m)
      const dedm = math.step(math.matMul(a, b));

      // de/da = dot(de/dy, bT)
      test_util.expectArraysClose(
          gradients.a,
          math.matMul(
              dedm, b, MatrixOrientation.REGULAR,
              MatrixOrientation.TRANSPOSED));

      // de/db = dot(aT, de/dy)
      test_util.expectArraysClose(
          gradients.b,
          math.matMul(
              a, dedm, MatrixOrientation.TRANSPOSED,
              MatrixOrientation.REGULAR));
    });

    it('second order nested gradient', math => {
      const a = Scalar.new(2);
      const gradients = math.gradients(() => {
        return math.gradients(() => {
          return math.pow(a, Scalar.new(3, 'int32'));
        }, a);
      }, a);

      expect(gradients.shape).toEqual(a.shape);
      test_util.expectNumbersClose(gradients.get(), 6 * a.get());
    });

    it('second order with gradientsScope', math => {
      const a = Scalar.new(2);

      const gradients = math.gradientsScope(() => {
        const der = math.gradients(() => {
          return math.pow(a, Scalar.new(3, 'int32'));
        }, a);

        // TODO(nsthorat|smilkov): Test that the intermittent gradient arrays
        // have not been cleaned up.

        return math.gradients(() => der, a);
      });

      expect(gradients.shape).toEqual(a.shape);
      test_util.expectNumbersClose(gradients.get(), 6 * a.get());
    });
  };

  test_util.describeMathCPU('valueAndGradients', [tests]);
  test_util.describeMathGPU('valueAndGradients', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
