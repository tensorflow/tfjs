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

import * as dl from '../index';
import * as test_util from '../test_util';
import {Gradients} from './backends/gradients';
import {MatrixOrientation} from './backends/types/matmul';
import {Scalar, Tensor} from './tensor';

const gradientsScope = Gradients.gradientsScope;

// dl.tidy
{
  const gpuTests = () => {
    it('returns Tensor', () => {
      dl.tidy(() => {
        const a = dl.tensor1d([1, 2, 3]);
        let b = dl.tensor1d([0, 0, 0]);

        expect(dl.memory().numTensors).toBe(2);
        dl.tidy(() => {
          const result = dl.tidy(() => {
            b = dl.addStrict(a, b);
            b = dl.addStrict(a, b);
            b = dl.addStrict(a, b);
            return dl.add(a, b);
          });

          // result is new. All intermediates should be disposed.
          expect(dl.memory().numTensors).toBe(2 + 1);
          test_util.expectArraysClose(result, [4, 8, 12]);
        });

        // a, b are still here, result should be disposed.
        expect(dl.memory().numTensors).toBe(2);
      });

      expect(dl.memory().numTensors).toBe(0);
    });

    it('multiple disposes does not affect num arrays', () => {
      expect(dl.memory().numTensors).toBe(0);
      const a = dl.tensor1d([1, 2, 3]);
      const b = dl.tensor1d([1, 2, 3]);
      expect(dl.memory().numTensors).toBe(2);
      a.dispose();
      a.dispose();
      expect(dl.memory().numTensors).toBe(1);
      b.dispose();
      expect(dl.memory().numTensors).toBe(0);
    });

    it('returns Tensor[]', () => {
      const a = dl.tensor1d([1, 2, 3]);
      const b = dl.tensor1d([0, -1, 1]);
      expect(dl.memory().numTensors).toBe(2);

      dl.tidy(() => {
        const result = dl.tidy(() => {
          dl.add(a, b);
          return [dl.add(a, b), dl.sub(a, b)];
        });

        // the 2 results are new. All intermediates should be disposed.
        expect(dl.memory().numTensors).toBe(4);
        test_util.expectArraysClose(result[0], [1, 1, 4]);
        test_util.expectArraysClose(result[1], [1, 3, 2]);
        expect(dl.memory().numTensors).toBe(4);
      });

      // the 2 results should be disposed.
      expect(dl.memory().numTensors).toBe(2);
      a.dispose();
      b.dispose();
      expect(dl.memory().numTensors).toBe(0);
    });

    it('basic usage without return', () => {
      const a = dl.tensor1d([1, 2, 3]);
      let b = dl.tensor1d([0, 0, 0]);

      expect(dl.memory().numTensors).toBe(2);

      dl.tidy(() => {
        b = dl.addStrict(a, b);
        b = dl.addStrict(a, b);
        b = dl.addStrict(a, b);
        dl.add(a, b);
      });

      // all intermediates should be disposed.
      expect(dl.memory().numTensors).toBe(2);
    });

    it('nested usage', () => {
      const a = dl.tensor1d([1, 2, 3]);
      let b = dl.tensor1d([0, 0, 0]);

      expect(dl.memory().numTensors).toBe(2);

      dl.tidy(() => {
        const result = dl.tidy(() => {
          b = dl.addStrict(a, b);
          b = dl.tidy(() => {
            b = dl.tidy(() => {
              return dl.addStrict(a, b);
            });
            // original a, b, and two intermediates.
            expect(dl.memory().numTensors).toBe(4);

            dl.tidy(() => {
              dl.addStrict(a, b);
            });
            // All the intermediates should be cleaned up.
            expect(dl.memory().numTensors).toBe(4);

            return dl.addStrict(a, b);
          });
          expect(dl.memory().numTensors).toBe(4);

          return dl.addStrict(a, b);
        });

        expect(dl.memory().numTensors).toBe(3);
        test_util.expectArraysClose(result, [4, 8, 12]);
      });
      expect(dl.memory().numTensors).toBe(2);
    });

    it('single argument', () => {
      let hasRan = false;
      dl.tidy(() => {
        hasRan = true;
      });
      expect(hasRan).toBe(true);
    });

    it('single argument, but not a function throws error', () => {
      expect(() => {
        dl.tidy('asdf');
      }).toThrowError();
    });

    it('2 arguments, first is string', () => {
      let hasRan = false;
      dl.tidy('name', () => {
        hasRan = true;
      });
      expect(hasRan).toBe(true);
    });

    it('2 arguments, but first is not string throws error', () => {
      expect(() => {
        // tslint:disable-next-line:no-any
        dl.tidy(4 as any, () => {});
      }).toThrowError();
    });

    it('2 arguments, but second is not a function throws error', () => {
      expect(() => {
        // tslint:disable-next-line:no-any
        dl.tidy('name', 'another name' as any);
      }).toThrowError();
    });
  };

  test_util.describeMathGPU('tidy', [gpuTests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// fromPixels & math
{
  const tests = () => {
    it('debug mode does not error when no nans', () => {
      const pixels = new ImageData(2, 2);
      for (let i = 0; i < 8; i++) {
        pixels.data[i] = 100;
      }
      for (let i = 8; i < 16; i++) {
        pixels.data[i] = 250;
      }

      const a = Tensor.fromPixels(pixels, 4);
      const b = dl.scalar(20, 'int32');

      const res = dl.add(a, b);

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

// vjp integration tests
{
  const tests = () => {
    it('matmul + relu', () => {
      const a = dl.tensor2d([-1, 2, -3, 10, -20, 30], [2, 3]);
      const b = dl.tensor2d([2, -3, 4, -1, 2, -3], [3, 2]);
      const dy = dl.tensor2d([1, 10, 20, 30], [2, 2]);

      const gradients = dl.vjp(() => {
        // m = dot(a, b)
        // y = relu(m)
        const m = dl.matMul(a, b);
        return dl.relu(m);
      }, {a, b}, dy);

      // dy/dm = step(m)
      // de/dm = de/dy * dy/dm = de/dy * step(m)
      const dedm = dl.mulStrict(dy, dl.step(dl.matMul(a, b)));

      // de/da = dot(de/dy, bT)
      expect(gradients.a.shape).toEqual(a.shape);
      test_util.expectArraysClose(
          gradients.a,
          dl.matMul(
              dedm, b, MatrixOrientation.REGULAR,
              MatrixOrientation.TRANSPOSED));

      // de/db = dot(aT, de/dy)
      expect(gradients.b.shape).toEqual(b.shape);
      test_util.expectArraysClose(
          gradients.b,
          dl.matMul(
              a, dedm, MatrixOrientation.TRANSPOSED,
              MatrixOrientation.REGULAR));
    });

    it('second order nested gradient vjp & gradients', () => {
      const a = dl.scalar(2);
      const b = dl.scalar(3, 'int32');

      const dy = dl.scalar(4);

      const gradients = dl.vjp(() => {
        return dl.gradients(() => dl.pow(a, b), a);
      }, a, dy);

      expect(gradients.shape).toEqual(a.shape);
      test_util.expectNumbersClose(
          gradients.get(),
          dy.get() * b.get() * (b.get() - 1) * Math.pow(a.get(), b.get() - 2),
          1e-1);
    });

    it('second order nested gradient', () => {
      const a = dl.scalar(2);
      const b = dl.scalar(3, 'int32');

      const dy1 = dl.scalar(3);
      const dy2 = dl.scalar(4);

      const gradients = dl.vjp(() => {
        return dl.vjp(() => dl.pow(a, b), a, dy1);
      }, a, dy2);

      expect(gradients.shape).toEqual(a.shape);
      test_util.expectNumbersClose(
          gradients.get(),
          dy2.get() * dy1.get() * b.get() * (b.get() - 1) *
              Math.pow(a.get(), b.get() - 2),
          1e-1);
    });
  };

  test_util.describeMathCPU('vjp', [tests]);
  test_util.describeMathGPU('vjp', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// gradients integration tests
{
  const tests = () => {
    it('matmul + relu', () => {
      const a = dl.tensor2d([-1, 2, -3, 10, -20, 30], [2, 3]);
      const b = dl.tensor2d([2, -3, 4, -1, 2, -3], [3, 2]);

      const gradients = dl.gradients(() => {
        // m = dot(a, b)
        // y = relu(m)
        // e = sum(y)
        const m = dl.matMul(a, b);
        const y = dl.relu(m);
        return dl.sum(y);
      }, {a, b});

      // de/dy = 1
      // dy/dm = step(m)
      // de/dm = de/dy * dy/dm = step(m)
      const dedm = dl.step(dl.matMul(a, b));

      // de/da = dot(de/dy, bT)
      expect(gradients.a.shape).toEqual(a.shape);
      test_util.expectArraysClose(
          gradients.a,
          dl.matMul(
              dedm, b, MatrixOrientation.REGULAR, MatrixOrientation.TRANSPOSED),
          1e-1);

      // de/db = dot(aT, de/dy)
      expect(gradients.b.shape).toEqual(b.shape);
      test_util.expectArraysClose(
          gradients.b,
          dl.matMul(
              a, dedm, MatrixOrientation.TRANSPOSED, MatrixOrientation.REGULAR),
          1e-1);
    });

    it('second order nested gradient', () => {
      const a = dl.scalar(2);
      const gradients = dl.gradients(() => {
        return dl.gradients(() => {
          return dl.pow(a, dl.scalar(3, 'int32'));
        }, a);
      }, a);

      expect(gradients.shape).toEqual(a.shape);
      test_util.expectNumbersClose(gradients.get(), 6 * a.get(), 1e-1);
    });

    it('works with reshape', () => {
      const a = dl.tensor2d([1, 2, 3, 4], [2, 2]);
      const exponent = dl.tensor1d([2, 2, 2, 2], 'int32');

      const gradients = dl.gradients(() => {
        const b = a.flatten();
        const m = dl.pow(b, exponent);
        return dl.sum(m);
      }, {a});

      expect(gradients.a.shape).toEqual([2, 2]);
      test_util.expectArraysClose(gradients.a, [2, 4, 6, 8]);
    });

    it('reshape outside dl.gradients() throws error', () => {
      const a = dl.tensor2d([1, 2, 3, 4], [2, 2]);
      const b = a.flatten();
      const exponent = dl.tensor1d([2, 2, 2, 2], 'int32');

      const f = () => {
        return dl.gradients(() => {
          const m = dl.pow(b, exponent);
          return dl.sum(m);
        }, {a, b});
      };
      expect(f).toThrowError();
    });

    it('works with asType', () => {
      const a = dl.tensor2d([1, 2, 3, 4], [2, 2], 'int32');
      const exponent = dl.tensor2d([2, 2, 2, 2], [2, 2], 'int32');

      const gradients = dl.gradients(() => {
        const b = a.toFloat();
        const m = dl.pow(b, exponent);
        return dl.sum(m);
      }, {a});

      expect(gradients.a.shape).toEqual([2, 2]);
      expect(gradients.a.dtype).toEqual('float32');
      test_util.expectArraysClose(gradients.a, [2, 4, 6, 8]);
    });

    it('asType outside of dl.gradients() throws error', () => {
      const a = dl.tensor2d([1, 2, 3, 4], [2, 2], 'int32');
      const b = a.toFloat();
      const exponent = dl.tensor2d([2, 2, 2, 2], [2, 2], 'int32');

      const f = () => {
        return dl.gradients(() => {
          const m = dl.pow(b, exponent);
          return dl.sum(m);
        }, {a});
      };
      expect(f).toThrowError();
    });
  };

  test_util.describeMathCPU('gradients', [tests]);
  test_util.describeMathGPU('gradients', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// valueAndGradients integration tests
{
  const tests = () => {
    it('matmul + relu', () => {
      const a = dl.tensor2d([-1, 2, -3, 10, -20, 30], [2, 3]);
      const b = dl.tensor2d([2, -3, 4, -1, 2, -3], [3, 2]);

      const {value, gradients} = dl.valueAndGradients(() => {
        // m = dot(a, b)
        // y = relu(m)
        // e = sum(y)
        const m = dl.matMul(a, b);
        const y = dl.relu(m);
        return dl.sum(y);
      }, {a, b});

      test_util.expectNumbersClose(value.get(), 10, 1e-1);

      // de/dy = 1
      // dy/dm = step(m)
      // de/dm = de/dy * dy/dm = step(m)
      const dedm = dl.step(dl.matMul(a, b));

      // de/da = dot(de/dy, bT)
      test_util.expectArraysClose(
          gradients.a,
          dl.matMul(
              dedm, b, MatrixOrientation.REGULAR, MatrixOrientation.TRANSPOSED),
          1e-1);

      // de/db = dot(aT, de/dy)
      test_util.expectArraysClose(
          gradients.b,
          dl.matMul(
              a, dedm, MatrixOrientation.TRANSPOSED, MatrixOrientation.REGULAR),
          1e-1);
    });

    it('matmul + relu + inner tidy', () => {
      const a = dl.tensor2d([-1, 2, -3, 10, -20, 30], [2, 3]);
      const b = dl.tensor2d([2, -3, 4, -1, 2, -3], [3, 2]);

      const {value, gradients} = dl.valueAndGradients(() => {
        // m = dot(a, b)
        // y = relu(m)
        // e = sum(y)
        const m = dl.matMul(a, b);
        return dl.tidy(() => {
          const y = dl.relu(m);
          return dl.sum(y);
        });
      }, {a, b});

      test_util.expectNumbersClose(value.get(), 10, 1e-1);

      // de/dy = 1
      // dy/dm = step(m)
      // de/dm = de/dy * dy/dm = step(m)
      const dedm = dl.step(dl.matMul(a, b));

      // de/da = dot(de/dy, bT)
      test_util.expectArraysClose(
          gradients.a,
          dl.matMul(
              dedm, b, MatrixOrientation.REGULAR, MatrixOrientation.TRANSPOSED),
          1e-1);

      // de/db = dot(aT, de/dy)
      test_util.expectArraysClose(
          gradients.b,
          dl.matMul(
              a, dedm, MatrixOrientation.TRANSPOSED, MatrixOrientation.REGULAR),
          1e-1);
    });
  };

  test_util.describeMathCPU('valueAndGradients', [tests]);
  test_util.describeMathGPU('valueAndGradients', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

{
  const tests = () => {
    it('second order gradients with gradientsScope', () => {
      const a = dl.scalar(2);
      expect(dl.memory().numTensors).toBe(1);

      const gradients = gradientsScope(() => {
        const der = dl.gradients(() => {
          const result = dl.pow(a, dl.scalar(3, 'int32'));
          expect(dl.memory().numTensors).toBe(3);

          return result as Scalar;
        }, a);

        // Gradients shouldn't be disposed.
        const numArrays = dl.memory().numTensors;
        expect(numArrays).toBeGreaterThan(3);

        const result = dl.gradients(() => der, a);

        // New gradients shouldn't be disposed.
        expect(dl.memory().numTensors).toBeGreaterThan(numArrays + 1);
        return result;
      });

      // a and gradients are the only remaining arrays.
      expect(dl.memory().numTensors).toBe(2);

      expect(gradients.shape).toEqual(a.shape);
      test_util.expectArraysClose(gradients, [2 * 3 * a.get()], 1e-1);
    });
  };

  test_util.describeMathCPU('gradientsScope', [tests]);
  test_util.describeMathGPU('gradientsScope', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// customGradients
{
  const tests = () => {
    it('basic', () => {
      const a = dl.scalar(3);
      const b = dl.scalar(2, 'int32');
      const dy = dl.scalar(4);

      const vjp = dl.vjp(() => {
        return dl.customGradient('test', () => {
          const value = dl.pow(a, b);

          const gradients = (dy: Tensor, y: Tensor) => {
            return {a: () => dl.mul(dy, dl.scalar(3))};
          };

          return {value, gradients};
        }, {a});
      }, a, dy);

      expect(vjp.shape).toEqual(a.shape);
      test_util.expectArraysClose(vjp, [dy.get() * 3]);
    });

    it('second order derivative through customGradient', () => {
      const a = dl.scalar(3);
      const b = dl.scalar(2, 'int32');

      const dy1 = dl.scalar(5);
      const dy2 = dl.scalar(4);

      const vjp = dl.vjp(() => {
        return dl.vjp(() => {
          return dl.customGradient('test', () => {
            const value = dl.pow(a, b);
            const gradients = (dy: Tensor, y: Tensor) => {
              return {a: () => dl.mul(dy, a)};
            };

            return {value, gradients};
          }, {a});
        }, a, dy1);
      }, a, dy2);

      expect(vjp.shape).toEqual(a.shape);

      // First order: dy1 * a
      // Second order: dy2 * dy1
      test_util.expectArraysClose(vjp, [dy1.get() * dy2.get()]);
    });
  };

  test_util.describeMathCPU('customGradient', [tests]);
  test_util.describeMathGPU('customGradient', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// dl.memory
{
  const tests = () => {
    it('Sum(float)', () => {
      expect(dl.memory().numTensors).toBe(0);
      expect(dl.memory().numBytes).toBe(0);
      const sum = dl.tidy(() => {
        const a = dl.tensor1d([1, 2, 3, 4]);
        expect(dl.memory().numTensors).toBe(1);
        expect(dl.memory().numBytes).toBe(4 * 4);
        return a.sum();
      });
      expect(dl.memory().numTensors).toBe(1);
      expect(dl.memory().numBytes).toBe(4);
      test_util.expectArraysClose(sum, [1 + 2 + 3 + 4]);
    });

    it('Sum(bool)', () => {
      const sum = dl.tidy(() => {
        const a = dl.tensor1d([true, true, false, true], 'bool');
        expect(dl.memory().numTensors).toBe(1);
        expect(dl.memory().numBytes).toBe(4);
        return a.sum();
      });
      expect(dl.memory().numTensors).toBe(1);
      expect(dl.memory().numBytes).toBe(4);
      expect(sum.dtype).toBe('int32');
      test_util.expectArraysClose(sum, [1 + 1 + 0 + 1]);
    });

    it('Sum(int32)', () => {
      const sum = dl.tidy(() => {
        const a = dl.tensor1d([1, 1, 0, 1], 'int32');
        expect(dl.memory().numTensors).toBe(1);
        expect(dl.memory().numBytes).toBe(4 * 4);
        return a.sum();
      });
      expect(dl.memory().numTensors).toBe(1);
      expect(dl.memory().numBytes).toBe(4);
      expect(sum.dtype).toBe('int32');
      test_util.expectArraysClose(sum, [1 + 1 + 0 + 1]);
    });
  };
  test_util.describeMathCPU('memory', [tests]);
  test_util.describeMathGPU('memory', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
