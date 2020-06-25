
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {ENGINE} from './engine';
import * as tf from './index';
import {ALL_ENVS, describeWithFlags} from './jasmine_util';
import {Tensor} from './tensor';
import {expectArraysClose} from './test_util';

describeWithFlags('gradients', ALL_ENVS, () => {
  it('matmul + relu', async () => {
    const a = tf.tensor2d([-1, 2, -3, 10, -20, 30], [2, 3]);
    const b = tf.tensor2d([2, -3, 4, -1, 2, -3], [3, 2]);

    const [da, db] = tf.grads((a: tf.Tensor2D, b: tf.Tensor2D) => {
      // m = dot(a, b)
      // y = relu(m)
      // e = sum(y)
      const m = tf.matMul(a, b);
      const y = tf.relu(m);
      return tf.sum(y);
    })([a, b]);

    // de/dy = 1
    // dy/dm = step(m)
    // de/dm = de/dy * dy/dm = step(m)
    const dedm = tf.step(tf.matMul(a, b));

    // de/da = dot(de/dy, bT)
    expect(da.shape).toEqual(a.shape);
    let transposeA = false;
    let transposeB = true;
    expectArraysClose(
        await da.data(),
        await tf.matMul(dedm, b, transposeA, transposeB).data());

    // de/db = dot(aT, de/dy)
    expect(db.shape).toEqual(b.shape);
    transposeA = true;
    transposeB = false;
    expectArraysClose(
        await db.data(),
        await tf.matMul(a, dedm, transposeA, transposeB).data());
  });

  it('grad(f)', async () => {
    const grad = tf.grad(x => x.square());
    const result = grad(tf.tensor1d([.1, .2]));
    expectArraysClose(await result.data(), [.2, .4]);
  });

  it('calling grad(f) twice works', async () => {
    const grad = tf.grad(x => x.square());

    const result = grad(tf.tensor1d([.1, .2]));
    const result2 = grad(tf.tensor1d([.1, .4]));
    expectArraysClose(await result.data(), [.2, .4]);
    expectArraysClose(await result2.data(), [.2, .8]);
  });

  it('grad(f): throwing an error during forward pass', () => {
    const grad = tf.grad(x => {
      throw new Error('failed forward pass');
    });
    expect(() => grad(tf.zeros([]))).toThrowError();
    expect(ENGINE.isTapeOn()).toBe(false);
  });

  it('grad(f): throwing an error during backwards pass', () => {
    const customOp = tf.customGrad((x: tf.Tensor) => {
      return {
        value: x,
        gradFunc: () => {
          throw new Error('failed backward pass');
        }
      };
    });
    const grad = tf.grad(x => customOp(x));
    expect(() => grad(tf.zeros([]))).toThrowError();
    expect(ENGINE.isTapeOn()).toBe(false);
  });

  it('grads(f)', async () => {
    const grads = tf.grads(x => x.square());
    const result = grads([tf.tensor1d([.1, .2])]);
    expectArraysClose(await result[0].data(), [.2, .4]);
  });

  it('calling grads(f) twice works', async () => {
    const grads = tf.grads(x => x.square());

    const result = grads([tf.tensor1d([.1, .2])]);
    const result2 = grads([tf.tensor1d([.1, .4])]);
    expectArraysClose(await result[0].data(), [.2, .4]);
    expectArraysClose(await result2[0].data(), [.2, .8]);
  });

  it('works with reshape', async () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const exponent = tf.tensor1d([2, 2, 2, 2], 'int32');

    const da = tf.grad(a => {
      const b = a.flatten();
      const m = tf.pow(b, exponent);
      return tf.sum(m);
    })(a);

    expect(da.shape).toEqual([2, 2]);
    expectArraysClose(await da.data(), [2, 4, 6, 8]);
  });

  it('reshape outside tf.grads() throws error', () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const b = a.flatten();
    const exponent = tf.tensor1d([2, 2, 2, 2], 'int32');

    const f = () => {
      tf.grads((a, b) => {
        const m = tf.pow(b, exponent);
        return tf.sum(m);
      })([a, b]);
    };
    expect(f).toThrowError();
  });

  it('does not error if irrelevant (pruned) ops are missing grads',
     async () => {
       const a = tf.tensor1d([true, true], 'bool');
       const b = tf.tensor1d([false, true], 'bool');
       const da = tf.grad(a => {
         // Logical has no gradients, but it is irrelevant.
         a.logicalAnd(b);
         return a.sum();
       })(a);
       expectArraysClose(await da.data(), [1, 1]);
     });

  it('errors if relevant ops are missing grads', () => {
    const a = tf.tensor1d([true, true], 'bool');
    const b = tf.tensor1d([false, true], 'bool');
    const dfda = tf.grad(a => {
      // Logical has no gradients, but it's relevant to the output.
      return a.logicalAnd(b);
    });
    expect(() => dfda(a)).toThrowError();
  });

  it('works with asType', async () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2], 'int32');
    const exponent = tf.tensor2d([2, 2, 2, 2], [2, 2], 'int32');

    const da = tf.grad(a => {
      const b = a.toFloat();
      const m = tf.pow(b, exponent);
      return tf.sum(m);
    })(a);

    expect(da.shape).toEqual([2, 2]);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [2, 4, 6, 8]);
  });

  it('asType outside of tf.grads() throws error', () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2], 'int32');
    const b = a.toFloat();
    const exponent = tf.tensor2d([2, 2, 2, 2], [2, 2], 'int32');

    const f = () => {
      tf.grad(a => {
        const m = tf.pow(b, exponent);
        return tf.sum(m);
      })(a);
    };
    expect(f).toThrowError();
  });

  it('saves tensors from the forward pass as expected', () => {
    const x = tf.scalar(1).variable();
    const optimizer = tf.train.sgd(0.1);
    optimizer.minimize(() => {
      const y = x.square();
      const z = y.square();
      y.dispose();
      return z;
    });
  });

  it('custom ops do not leak', () => {
    const before = tf.memory().numTensors;
    const x = tf.softmax([1, 2, 3, 4]);
    x.dispose();
    const now = tf.memory().numTensors;
    expect(now).toBe(before);
  });
});

describeWithFlags('valueAndGradients', ALL_ENVS, () => {
  it('matmul + relu', async () => {
    const a = tf.tensor2d([-1, 2, -3, 10, -20, 30], [2, 3]);
    const b = tf.tensor2d([2, -3, 4, -1, 2, -3], [3, 2]);

    const {value, grads} =
        tf.valueAndGrads((a: tf.Tensor2D, b: tf.Tensor2D) => {
          // m = dot(a, b)
          // y = relu(m)
          // e = sum(y)
          const m = tf.matMul(a, b);
          const y = tf.relu(m);
          return tf.sum(y);
        })([a, b]);

    expectArraysClose(await value.data(), 10);

    // de/dy = 1
    // dy/dm = step(m)
    // de/dm = de/dy * dy/dm = step(m)
    const dedm = tf.step(tf.matMul(a, b));

    const [da, db] = grads;
    // de/da = dot(de/dy, bT)
    let transposeA = false;
    let transposeB = true;
    expectArraysClose(
        await da.data(),
        await tf.matMul(dedm, b, transposeA, transposeB).data());

    // de/db = dot(aT, de/dy)
    transposeA = true;
    transposeB = false;
    expectArraysClose(
        await db.data(),
        await tf.matMul(a, dedm, transposeA, transposeB).data());
  });

  it('matmul + relu + inner tidy', async () => {
    const a = tf.tensor2d([-1, 2, -3, 10, -20, 30], [2, 3]);
    const b = tf.tensor2d([2, -3, 4, -1, 2, -3], [3, 2]);

    const {value, grads} =
        tf.valueAndGrads((a: tf.Tensor2D, b: tf.Tensor2D) => {
          // m = dot(a, b)
          // y = relu(m)
          // e = sum(y)
          const m = tf.matMul(a, b);
          return tf.tidy(() => {
            const y = tf.relu(m);
            return tf.sum(y);
          });
        })([a, b]);

    expectArraysClose(await value.data(), 10);

    // de/dy = 1
    // dy/dm = step(m)
    // de/dm = de/dy * dy/dm = step(m)
    const dedm = tf.step(tf.matMul(a, b));

    const [da, db] = grads;
    // de/da = dot(de/dy, bT)
    let transposeA = false;
    let transposeB = true;
    expectArraysClose(
        await da.data(),
        await tf.matMul(dedm, b, transposeA, transposeB).data());

    // de/db = dot(aT, de/dy)
    transposeA = true;
    transposeB = false;
    expectArraysClose(
        await db.data(),
        await tf.matMul(a, dedm, transposeA, transposeB).data());
  });
});

describeWithFlags('higher-order gradients', ALL_ENVS, () => {
  it('grad(grad(f))', async () => {
    const x = tf.tensor1d([.1, .2]);
    const before = tf.memory().numTensors;
    const gradgrad = tf.grad(tf.grad(x => x.mul(x).mul(x)));
    const result = gradgrad(x);
    expect(tf.memory().numTensors).toBe(before + 1);
    expectArraysClose(await result.data(), [.6, 1.2]);
  });

  it('grad(grad(x^2))', async () => {
    const x = tf.scalar(3);
    const gradgrad = tf.grad(tf.grad(x => x.square()));
    const result = gradgrad(x);
    // grad(grad(x^2)) = grad(2x) = 2
    expectArraysClose(await result.data(), [2]);
  });

  it('grads(grads(f))', async () => {
    const grads = tf.grads(x => x.mul(x).mul(x));
    const gradsgrads = tf.grads(x => grads([x])[0]);
    const result = gradsgrads([tf.tensor1d([.1, .2])]);
    expectArraysClose(await result[0].data(), [.6, 1.2]);
  });
});

describeWithFlags('customGradient', ALL_ENVS, () => {
  it('basic', async () => {
    const a = tf.scalar(3);
    const b = tf.scalar(2, 'int32');
    const dy = tf.scalar(4);

    const customPow = tf.customGrad((a: tf.Tensor) => {
      const value = tf.pow(a, b);
      const gradFunc = (dy: tf.Tensor) => dy.mul(tf.scalar(0.1));
      return {value, gradFunc};
    });

    const {value, grad} = tf.valueAndGrad(a => customPow(a))(a, dy);
    expect(value.shape).toEqual(a.shape);
    expectArraysClose(await value.data(), [9]);
    expect(grad.shape).toEqual(a.shape);
    expectArraysClose(await grad.data(), [.4]);
  });

  it('second order derivative through customGradient', async () => {
    const a = tf.scalar(3);
    const b = tf.scalar(2, 'int32');

    const dy = tf.scalar(5);

    const customPow = tf.customGrad((a: tf.Tensor, save: tf.GradSaveFunc) => {
      const value = tf.pow(a, b);
      save([a]);
      const gradFunc = (dy: tf.Tensor, saved: Tensor[]) => {
        const [a] = saved;
        return dy.mul(a);
      };
      return {value, gradFunc};
    });

    const dda = tf.grad(tf.grad(a => customPow(a)))(a, dy);
    expect(dda.shape).toEqual(a.shape);

    // First order: dy * a. Second order: dy.
    expectArraysClose(await dda.data(), await dy.data());
  });

  it('calling gradient of custom op twice works', async () => {
    const customOp = tf.customGrad((x: tf.Tensor, save: tf.GradSaveFunc) => {
      // Override gradient of our custom x ^ 2 op to be dy * abs(x);
      save([x]);
      return {
        value: x.square(),
        gradFunc: (dy, saved: Tensor[]) => {
          const [x] = saved;
          return dy.mul(x.abs());
        }
      };
    });
    const x = tf.tensor1d([-1, -2, 3]);
    const grad = tf.grad(x => customOp(x));

    expectArraysClose(await grad(x).data(), [1, 2, 3]);
    expectArraysClose(await grad(x).data(), [1, 2, 3]);
  });
});
