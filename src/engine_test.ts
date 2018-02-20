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

import * as dl from './index';
import {Tensor} from './tensor';
// tslint:disable-next-line:max-line-length
import {ALL_ENVS, describeWithFlags, expectArraysClose, expectArraysEqual, expectNumbersClose} from './test_util';
import {ENV} from './environment';

describeWithFlags('tidy', ALL_ENVS, () => {
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
        expectArraysClose(result, [4, 8, 12]);
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
      expectArraysClose(result[0], [1, 1, 4]);
      expectArraysClose(result[1], [1, 3, 2]);
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
      expectArraysClose(result, [4, 8, 12]);
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
});

describeWithFlags('fromPixels + regular math op', ALL_ENVS, () => {
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

    expectArraysEqual(res, [
      120, 120, 120, 120, 120, 120, 120, 120, 270, 270, 270, 270, 270, 270, 270,
      270
    ]);
  });
});

describeWithFlags('gradients', ALL_ENVS, () => {
  it('matmul + relu', () => {
    const a = dl.tensor2d([-1, 2, -3, 10, -20, 30], [2, 3]);
    const b = dl.tensor2d([2, -3, 4, -1, 2, -3], [3, 2]);

    const [da, db] = dl.grads((a: dl.Tensor2D, b: dl.Tensor2D) => {
      // m = dot(a, b)
      // y = relu(m)
      // e = sum(y)
      const m = dl.matMul(a, b);
      const y = dl.relu(m);
      return dl.sum(y);
    })([a, b]);

    // de/dy = 1
    // dy/dm = step(m)
    // de/dm = de/dy * dy/dm = step(m)
    const dedm = dl.step(dl.matMul(a, b));

    // de/da = dot(de/dy, bT)
    expect(da.shape).toEqual(a.shape);
    let transposeA = false;
    let transposeB = true;
    expectArraysClose(da, dl.matMul(dedm, b, transposeA, transposeB));

    // de/db = dot(aT, de/dy)
    expect(db.shape).toEqual(b.shape);
    transposeA = true;
    transposeB = false;
    expectArraysClose(db, dl.matMul(a, dedm, transposeA, transposeB));
  });

  it('grad(f)', () => {
    const grad = dl.grad(x => x.square());
    const result = grad(dl.tensor1d([.1, .2]));
    expectArraysClose(result, [.2, .4]);
  });

  it('calling grad(f) twice works', () => {
    const grad = dl.grad(x => x.square());

    const result = grad(dl.tensor1d([.1, .2]));
    const result2 = grad(dl.tensor1d([.1, .4]));
    expectArraysClose(result, [.2, .4]);
    expectArraysClose(result2, [.2, .8]);
  });

  it('grads(f)', () => {
    const grads = dl.grads(x => x.square());
    const result = grads([dl.tensor1d([.1, .2])]);
    expectArraysClose(result[0], [.2, .4]);
  });

  it('calling grads(f) twice works', () => {
    const grads = dl.grads(x => x.square());

    const result = grads([dl.tensor1d([.1, .2])]);
    const result2 = grads([dl.tensor1d([.1, .4])]);
    expectArraysClose(result[0], [.2, .4]);
    expectArraysClose(result2[0], [.2, .8]);
  });

  it('works with reshape', () => {
    const a = dl.tensor2d([1, 2, 3, 4], [2, 2]);
    const exponent = dl.tensor1d([2, 2, 2, 2], 'int32');

    const da = dl.grad(a => {
      const b = a.flatten();
      const m = dl.pow(b, exponent);
      return dl.sum(m);
    })(a);

    expect(da.shape).toEqual([2, 2]);
    expectArraysClose(da, [2, 4, 6, 8]);
  });

  it('reshape outside dl.grads() throws error', () => {
    const a = dl.tensor2d([1, 2, 3, 4], [2, 2]);
    const b = a.flatten();
    const exponent = dl.tensor1d([2, 2, 2, 2], 'int32');

    const f = () => {
      dl.grads((a, b) => {
        const m = dl.pow(b, exponent);
        return dl.sum(m);
      })([a, b]);
    };
    expect(f).toThrowError();
  });

  it('works with asType', () => {
    const a = dl.tensor2d([1, 2, 3, 4], [2, 2], 'int32');
    const exponent = dl.tensor2d([2, 2, 2, 2], [2, 2], 'int32');

    const da = dl.grad(a => {
      const b = a.toFloat();
      const m = dl.pow(b, exponent);
      return dl.sum(m);
    })(a);

    expect(da.shape).toEqual([2, 2]);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [2, 4, 6, 8]);
  });

  it('asType outside of dl.grads() throws error', () => {
    const a = dl.tensor2d([1, 2, 3, 4], [2, 2], 'int32');
    const b = a.toFloat();
    const exponent = dl.tensor2d([2, 2, 2, 2], [2, 2], 'int32');

    const f = () => {
      dl.grad(a => {
        const m = dl.pow(b, exponent);
        return dl.sum(m);
      })(a);
    };
    expect(f).toThrowError();
  });

  it('empty list of xs leads to Error', () => {
    expect(() => ENV.engine.gradients(() => dl.scalar(1), [], dl.scalar(2)))
        .toThrowError(/empty list of xs/);
  });
});

describeWithFlags('valueAndGradients', ALL_ENVS, () => {
  it('matmul + relu', () => {
    const a = dl.tensor2d([-1, 2, -3, 10, -20, 30], [2, 3]);
    const b = dl.tensor2d([2, -3, 4, -1, 2, -3], [3, 2]);

    const {value, grads} =
        dl.valueAndGrads((a: dl.Tensor2D, b: dl.Tensor2D) => {
          // m = dot(a, b)
          // y = relu(m)
          // e = sum(y)
          const m = dl.matMul(a, b);
          const y = dl.relu(m);
          return dl.sum(y);
        })([a, b]);

    expectNumbersClose(value.get(), 10);

    // de/dy = 1
    // dy/dm = step(m)
    // de/dm = de/dy * dy/dm = step(m)
    const dedm = dl.step(dl.matMul(a, b));

    const [da, db] = grads;
    // de/da = dot(de/dy, bT)
    let transposeA = false;
    let transposeB = true;
    expectArraysClose(da, dl.matMul(dedm, b, transposeA, transposeB));

    // de/db = dot(aT, de/dy)
    transposeA = true;
    transposeB = false;
    expectArraysClose(db, dl.matMul(a, dedm, transposeA, transposeB));
  });

  it('matmul + relu + inner tidy', () => {
    const a = dl.tensor2d([-1, 2, -3, 10, -20, 30], [2, 3]);
    const b = dl.tensor2d([2, -3, 4, -1, 2, -3], [3, 2]);

    const {value, grads} =
        dl.valueAndGrads((a: dl.Tensor2D, b: dl.Tensor2D) => {
          // m = dot(a, b)
          // y = relu(m)
          // e = sum(y)
          const m = dl.matMul(a, b);
          return dl.tidy(() => {
            const y = dl.relu(m);
            return dl.sum(y);
          });
        })([a, b]);

    expectNumbersClose(value.get(), 10);

    // de/dy = 1
    // dy/dm = step(m)
    // de/dm = de/dy * dy/dm = step(m)
    const dedm = dl.step(dl.matMul(a, b));

    const [da, db] = grads;
    // de/da = dot(de/dy, bT)
    let transposeA = false;
    let transposeB = true;
    expectArraysClose(da, dl.matMul(dedm, b, transposeA, transposeB));

    // de/db = dot(aT, de/dy)
    transposeA = true;
    transposeB = false;
    expectArraysClose(db, dl.matMul(a, dedm, transposeA, transposeB));
  });
});

describeWithFlags('higher-order gradients', ALL_ENVS, () => {
  it('grad(grad(f))', () => {
    const gradgrad = dl.grad(dl.grad(x => x.mul(x).mul(x)));
    const result = gradgrad(dl.tensor1d([.1, .2]));
    expectArraysClose(result, [.6, 1.2]);
  });

  it('grads(grads(f))', () => {
    const grads = dl.grads(x => x.mul(x).mul(x));
    const gradsgrads = dl.grads(x => grads([x])[0]);
    const result = gradsgrads([dl.tensor1d([.1, .2])]);
    expectArraysClose(result[0], [.6, 1.2]);
  });
});

describeWithFlags('customGradient', ALL_ENVS, () => {
  it('basic', () => {
    const a = dl.scalar(3);
    const b = dl.scalar(2, 'int32');
    const dy = dl.scalar(4);

    const customPow = dl.customGrad(a => {
      const value = dl.pow(a, b);
      const gradFunc = (dy: Tensor) => dy.mul(dl.scalar(0.1));
      return {value, gradFunc};
    });

    const {value, grad} = dl.valueAndGrad(a => customPow(a))(a, dy);
    expect(value.shape).toEqual(a.shape);
    expectArraysClose(value, [9]);
    expect(grad.shape).toEqual(a.shape);
    expectArraysClose(grad, [.4]);
  });

  it('second order derivative through customGradient', () => {
    const a = dl.scalar(3);
    const b = dl.scalar(2, 'int32');

    const dy = dl.scalar(5);

    const customPow = dl.customGrad(a => {
      const value = dl.pow(a, b);
      const gradFunc = (dy: Tensor) => dy.mul(a);
      return {value, gradFunc};
    });

    const dda = dl.grad(dl.grad(a => customPow(a)))(a, dy);
    expect(dda.shape).toEqual(a.shape);

    // First order: dy * a. Second order: dy.
    expectArraysClose(dda, dy);
  });

  it('calling gradient of custom op twice works', () => {
    const customOp = dl.customGrad(x => {
      // Override gradient of our custom x ^ 2 op to be dy * abs(x);
      return {value: x.square(), gradFunc: dy => dy.mul(x.abs())};
    });
    const x = dl.tensor1d([-1, -2, 3]);
    const grad = dl.grad(x => customOp(x));

    expectArraysClose(grad(x), [1, 2, 3]);
    expectArraysClose(grad(x), [1, 2, 3]);
  });
});

describeWithFlags('memory', ALL_ENVS, () => {
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
    expectArraysClose(sum, [1 + 2 + 3 + 4]);
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
    expectArraysClose(sum, [1 + 1 + 0 + 1]);
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
    expectArraysClose(sum, [1 + 1 + 0 + 1]);
  });
});
