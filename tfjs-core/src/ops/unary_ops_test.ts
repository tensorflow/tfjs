/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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

import * as tf from '../index';
import {ALL_ENVS, describeWithFlags} from '../jasmine_util';
import {expectArraysClose} from '../test_util';

describeWithFlags('step', ALL_ENVS, () => {
  it('with 1d tensor', async () => {
    const a = tf.tensor1d([1, -2, -.01, 3, -0.1]);
    const result = tf.step(a);
    expectArraysClose(await result.data(), [1, 0, 0, 1, 0]);
  });

  it('with 1d tensor and alpha', async () => {
    const a = tf.tensor1d([1, -2, -.01, 3, NaN]);
    const result = tf.step(a, 0.1);
    expectArraysClose(await result.data(), [1, 0.1, 0.1, 1, NaN]);
  });

  it('with 2d tensor', async () => {
    const a = tf.tensor2d([1, -5, -3, 4], [2, 2]);
    const result = tf.step(a);
    expect(result.shape).toEqual([2, 2]);
    expectArraysClose(await result.data(), [1, 0, 0, 1]);
  });

  it('propagates NaNs', async () => {
    const a = tf.tensor1d([1, -2, -.01, 3, NaN]);
    const result = tf.step(a);
    expectArraysClose(await result.data(), [1, 0, 0, 1, NaN]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(-4);
    const dy = tf.scalar(8);

    const gradients = tf.grad(a => tf.step(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [0]);
  });

  it('gradient with clones', async () => {
    const a = tf.scalar(-4);
    const dy = tf.scalar(8);

    const gradients = tf.grad(a => tf.step(a.clone()).clone())(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [0]);
  });

  it('gradients: Tensor1D', async () => {
    const a = tf.tensor1d([1, 2, -3, 5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const gradients = tf.grad(a => tf.step(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [0, 0, 0, 0]);
  });

  it('gradients: Tensor2D', async () => {
    const a = tf.tensor2d([3, -1, -2, 3], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = tf.grad(a => tf.step(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [0, 0, 0, 0]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.step({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'step' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const result = tf.step([1, -2, -.01, 3, -0.1]);
    expectArraysClose(await result.data(), [1, 0, 0, 1, 0]);
  });

  it('throws for string tensor', () => {
    expect(() => tf.step('q'))
        .toThrowError(/Argument 'x' passed to 'step' must be numeric/);
  });
});

describeWithFlags('sigmoid', ALL_ENVS, () => {
  it('basic', async () => {
    const values = [1, -3, 2, 7, -4];
    const a = tf.tensor1d(values);

    const result = tf.sigmoid(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = 1 / (1 + Math.exp(-values[i]));
    }
    expectArraysClose(await result.data(), expected);
  });

  it('6D', async () => {
    const a = tf.ones([2, 2, 2, 2, 2, 2]);
    const result = tf.sigmoid(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = 1 / (1 + Math.exp(-1.0));
    }
    expectArraysClose(await result.data(), expected);
  });

  it('propagates NaNs', async () => {
    const a = tf.tensor1d([3, NaN]);
    const res = tf.sigmoid(a);
    expectArraysClose(await res.data(), [1 / (1 + Math.exp(-3)), NaN]);
  });

  it('gradients: Tensor1D', async () => {
    const a = tf.tensor1d([1, 2, -3, 5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const da = tf.grad(a => tf.sigmoid(a))(a, dy);

    const aVals = await a.array();
    const dyVals = await dy.array();
    const expected = [];
    for (let i = 0; i < a.size; i++) {
      const y = 1 / (1 + Math.exp(-aVals[i]));
      expected[i] = dyVals[i] * y * (1 - y);
    }

    expectArraysClose(await da.data(), expected);
  });

  it('gradient with clones', async () => {
    const a = tf.tensor1d([1, 2, -3, 5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const da = tf.grad(a => tf.sigmoid(a.clone()).clone())(a, dy);

    const aVals = await a.array();
    const dyVals = await dy.array();
    const expected = [];
    for (let i = 0; i < a.size; i++) {
      const y = 1 / (1 + Math.exp(-aVals[i]));
      expected[i] = dyVals[i] * y * (1 - y);
    }

    expectArraysClose(await da.data(), expected);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.sigmoid({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'sigmoid' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const values = [1, -3, 2, 7, -4];
    const result = tf.sigmoid(values);

    const expected = [];
    for (let i = 0; i < values.length; i++) {
      expected[i] = 1 / (1 + Math.exp(-values[i]));
    }
    expectArraysClose(await result.data(), expected);
  });

  it('throws for string tensor', () => {
    expect(() => tf.sigmoid('q'))
        .toThrowError(/Argument 'x' passed to 'sigmoid' must be numeric/);
  });
});

describeWithFlags('softplus', ALL_ENVS, () => {
  it('basic', async () => {
    const values = [1, -3, 2, 7, -4];
    const a = tf.tensor1d(values);

    const result = tf.softplus(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.log((1 + Math.exp(values[i])));
    }
    expectArraysClose(await result.data(), expected);
  });

  it('scalar', async () => {
    const a = tf.scalar(-2);

    const result = tf.softplus(a);

    const expected = [Math.log((1 + Math.exp(-2)))];
    expectArraysClose(await result.data(), expected);
  });

  it('tensor2D', async () => {
    const values = [1, 2, -3, 5];
    const a = tf.tensor2d(values, [2, 2]);

    const result = tf.softplus(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.log((1 + Math.exp(values[i])));
    }
    expectArraysClose(await result.data(), expected);
  });

  it('larger magnitude negative inputs', async () => {
    const values = [-100, -200, -3000, -50000];
    const a = tf.tensor1d(values);

    const result = tf.softplus(a);

    const expected = [0, 0, 0, 0];

    expectArraysClose(await result.data(), expected);
  });

  it('larger magnitude positive inputs', async () => {
    const values = [100, 200, 3000];
    const a = tf.tensor1d(values);

    const result = tf.softplus(a);

    const expected = [100, 200, 3000];

    expectArraysClose(await result.data(), expected);
  });

  it('propagates NaNs', async () => {
    const a = tf.tensor1d([3, NaN]);
    const res = tf.softplus(a);
    expectArraysClose(await res.data(), [Math.log((1 + Math.exp(3))), NaN]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(3);
    const dy = tf.scalar(4);
    const aVal = await a.array();
    const dyVal = await dy.array();

    const da = tf.grad(a => tf.softplus(a))(a, dy);
    const y = 1 / (1 + Math.exp(-aVal));

    expectArraysClose(await da.data(), [dyVal * y]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(3);
    const dy = tf.scalar(4);
    const aVal = await a.array();
    const dyVal = await dy.array();

    const da = tf.grad(a => tf.softplus(a.clone()).clone())(a, dy);
    const y = 1 / (1 + Math.exp(-aVal));

    expectArraysClose(await da.data(), [dyVal * y]);
  });

  it('gradients: Tensor1D', async () => {
    const a = tf.tensor1d([1, 2, -3, 5]);
    const aVals = await a.array();
    const dy = tf.tensor1d([1, 2, 3, 4]);
    const dyVals = await dy.array();
    const da = tf.grad(a => tf.softplus(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      const y = 1 / (1 + Math.exp(-aVals[i]));
      expected[i] = dyVals[i] * y;
    }

    expectArraysClose(await da.data(), expected);
  });

  it('gradients: Tensor2D', async () => {
    const a = tf.tensor2d([1, 2, -3, 5], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const da = tf.grad(a => tf.softplus(a))(a, dy);

    const expected = [];
    const aVals = await a.data();
    const dyVals = await dy.data();

    for (let i = 0; i < a.size; i++) {
      const y = 1 / (1 + Math.exp(-aVals[i]));
      expected[i] = dyVals[i] * y;
    }

    expectArraysClose(await da.data(), expected);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.softplus({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'softplus' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const result = tf.softplus(-2);
    const expected = [Math.log((1 + Math.exp(-2)))];
    expectArraysClose(await result.data(), expected);
  });

  it('throws for string tensor', () => {
    expect(() => tf.softplus('q'))
        .toThrowError(/Argument 'x' passed to 'softplus' must be numeric/);
  });
});

describeWithFlags('sqrt', ALL_ENVS, () => {
  it('sqrt', async () => {
    const a = tf.tensor1d([2, 4]);
    const r = tf.sqrt(a);
    expectArraysClose(await r.data(), [Math.sqrt(2), Math.sqrt(4)]);
  });

  it('sqrt propagates NaNs', async () => {
    const a = tf.tensor1d([1, NaN]);
    const r = tf.sqrt(a);
    expectArraysClose(await r.data(), [Math.sqrt(1), NaN]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(4);
    const dy = tf.scalar(8);

    const da = tf.grad(a => tf.sqrt(a))(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [8 / (2 * Math.sqrt(4))]);
  });

  it('gradient with clones', async () => {
    const a = tf.scalar(4);
    const dy = tf.scalar(8);

    const da = tf.grad(a => tf.sqrt(a.clone()).clone())(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [8 / (2 * Math.sqrt(4))]);
  });

  it('gradients: Tensor1D', async () => {
    const a = tf.tensor1d([1, 2, 3, 5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const gradients = tf.grad(a => tf.sqrt(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        await gradients.data(),
        [
          1 / (2 * Math.sqrt(1)), 2 / (2 * Math.sqrt(2)),
          3 / (2 * Math.sqrt(3)), 4 / (2 * Math.sqrt(5))
        ],
        1e-1);
  });

  it('gradients: Tensor2D', async () => {
    const a = tf.tensor2d([3, 1, 2, 3], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = tf.grad(a => tf.sqrt(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        await gradients.data(),
        [
          1 / (2 * Math.sqrt(3)), 2 / (2 * Math.sqrt(1)),
          3 / (2 * Math.sqrt(2)), 4 / (2 * Math.sqrt(3))
        ],
        1e-1);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.sqrt({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'sqrt' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const r = tf.sqrt([2, 4]);
    expectArraysClose(await r.data(), [Math.sqrt(2), Math.sqrt(4)]);
  });

  it('throws for string tensor', () => {
    expect(() => tf.sqrt('q'))
        .toThrowError(/Argument 'x' passed to 'sqrt' must be numeric/);
  });
});

describeWithFlags('rsqrt', ALL_ENVS, () => {
  it('rsqrt', async () => {
    const a = tf.tensor1d([2, 4]);
    const r = tf.rsqrt(a);
    expectArraysClose(await r.data(), [1 / Math.sqrt(2), 1 / Math.sqrt(4)]);
  });

  it('rsqrt propagates NaNs', async () => {
    const a = tf.tensor1d([1, NaN]);
    const r = tf.rsqrt(a);
    expectArraysClose(await r.data(), [1 / Math.sqrt(1), NaN]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(4);
    const dy = tf.scalar(8);

    const da = tf.grad(a => tf.rsqrt(a))(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [(-1 * 8) / (2 * Math.pow(4, 1.5))]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(4);
    const dy = tf.scalar(8);

    const da = tf.grad(a => tf.rsqrt(a.clone()).clone())(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [(-1 * 8) / (2 * Math.pow(4, 1.5))]);
  });

  it('gradients: Tensor1D', async () => {
    const a = tf.tensor1d([1, 2, 3, 5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const gradients = tf.grad(a => tf.rsqrt(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        await gradients.data(),
        [
          -1 * 1 / (2 * Math.pow(1, 1.5)), -1 * 2 / (2 * Math.pow(2, 1.5)),
          -1 * 3 / (2 * Math.pow(3, 1.5)), -1 * 4 / (2 * Math.pow(5, 1.5))
        ],
        1e-1);
  });

  it('gradients: Tensor2D', async () => {
    const a = tf.tensor2d([3, 1, 2, 3], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = tf.grad(a => tf.rsqrt(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        await gradients.data(),
        [
          -1 * 1 / (2 * Math.pow(3, 1.5)), -1 * 2 / (2 * Math.pow(1, 1.5)),
          -1 * 3 / (2 * Math.pow(2, 1.5)), -1 * 4 / (2 * Math.pow(3, 1.5))
        ],
        1e-1);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.rsqrt({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'rsqrt' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const r = tf.rsqrt([2, 4]);
    expectArraysClose(await r.data(), [1 / Math.sqrt(2), 1 / Math.sqrt(4)]);
  });

  it('throws for string tensor', () => {
    expect(() => tf.rsqrt('q'))
        .toThrowError(/Argument 'x' passed to 'rsqrt' must be numeric/);
  });
});

describeWithFlags('square', ALL_ENVS, () => {
  it('1D array', async () => {
    const a = tf.tensor1d([2, 4, Math.sqrt(2)]);
    const r = tf.square(a);
    expectArraysClose(await r.data(), [4, 16, 2]);
  });

  it('2D array', async () => {
    const a = tf.tensor2d([1, 2, Math.sqrt(2), Math.sqrt(3)], [2, 2]);
    const r = tf.square(a);
    expect(r.shape).toEqual([2, 2]);
    expectArraysClose(await r.data(), [1, 4, 2, 3]);
  });

  it('5D array', async () => {
    const a = tf.tensor5d([1, 2, Math.sqrt(2), Math.sqrt(3)], [1, 1, 2, 2, 1]);
    const r = tf.square(a);
    expect(r.shape).toEqual([1, 1, 2, 2, 1]);
    expectArraysClose(await r.data(), [1, 4, 2, 3]);
  });

  it('6D array', async () => {
    const a = tf.tensor6d(
        [1, 2, Math.sqrt(2), Math.sqrt(3), 3, 4, Math.sqrt(7), Math.sqrt(13)],
        [1, 1, 2, 2, 2, 1]);
    const r = tf.square(a);
    expect(r.shape).toEqual(a.shape);
    expectArraysClose(await r.data(), [1, 4, 2, 3, 9, 16, 7, 13]);
  });

  it('square propagates NaNs', async () => {
    const a = tf.tensor1d([1.5, NaN]);
    const r = tf.square(a);
    expectArraysClose(await r.data(), [2.25, NaN]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(5);
    const dy = tf.scalar(8);

    const gradients = tf.grad(a => tf.square(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [2 * 5 * 8]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(5);
    const dy = tf.scalar(8);

    const gradients = tf.grad(a => tf.square(a.clone()).clone())(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [2 * 5 * 8]);
  });

  it('gradients: Tensor1D', async () => {
    const a = tf.tensor1d([-1, 2, 3, -5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const gradients = tf.grad(a => tf.square(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [-2, 4 * 2, 6 * 3, -10 * 4]);
  });

  it('gradients: Tensor2D', async () => {
    const a = tf.tensor2d([-3, 1, 2, 3], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = tf.grad(a => tf.square(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [-6 * 1, 2 * 2, 4 * 3, 6 * 4]);
  });

  it('gradients: Tensor5D', async () => {
    const a = tf.tensor5d([-3, 1, 2, 3], [1, 1, 1, 2, 2]);
    const dy = tf.tensor5d([1, 2, 3, 4], [1, 1, 1, 2, 2]);

    const gradients = tf.grad(a => tf.square(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [-6 * 1, 2 * 2, 4 * 3, 6 * 4]);
  });

  it('gradients: Tensor6D', async () => {
    const a = tf.tensor6d([-3, 1, 2, 3, -4, 5, 12, 3], [1, 1, 1, 2, 2, 2]);
    const dy = tf.tensor6d([1, 2, 3, 4, 5, 6, 7, 8], [1, 1, 1, 2, 2, 2]);

    const gradients = tf.grad(a => tf.square(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        await gradients.data(),
        [-6 * 1, 2 * 2, 4 * 3, 6 * 4, -8 * 5, 10 * 6, 24 * 7, 6 * 8]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.square({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'square' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const r = tf.square([2, 4, Math.sqrt(2)]);
    expectArraysClose(await r.data(), [4, 16, 2]);
  });

  it('throws for string tensor', () => {
    expect(() => tf.square('q'))
        .toThrowError(/Argument 'x' passed to 'square' must be numeric/);
  });
});

describeWithFlags('isNaN', ALL_ENVS, () => {
  it('basic', async () => {
    const a = tf.tensor1d([NaN, Infinity, -Infinity, 0, 1]);
    const r = tf.isNaN(a);
    expect(r.dtype).toEqual('bool');
    expectArraysClose(await r.data(), [1, 0, 0, 0, 0]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(NaN);
    const dy = tf.scalar(3);

    const gradients = tf.grad(a => tf.isNaN(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [0]);
  });

  it('gradients: Tensor1D', async () => {
    const a = tf.tensor1d([NaN, Infinity, -Infinity, 0, 1]);
    const dy = tf.tensor1d([1, 1, 1, 1, 1]);

    const gradients = tf.grad(a => tf.isNaN(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [0, 0, 0, 0, 0]);
  });

  it('gradients: Tensor2D', async () => {
    const a = tf.tensor2d([NaN, Infinity, -Infinity, 0], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = tf.grad(a => tf.isNaN(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [0, 0, 0, 0]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.isNaN({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'isNaN' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const r = tf.isNaN([NaN, Infinity, -Infinity, 0, 1]);
    expectArraysClose(await r.data(), [1, 0, 0, 0, 0]);
  });

  it('throws for string tensor', () => {
    expect(() => tf.isNaN('q'))
        .toThrowError(/Argument 'x' passed to 'isNaN' must be numeric/);
  });
});

describeWithFlags('isInf', ALL_ENVS, () => {
  it('basic', async () => {
    const a = tf.tensor1d([NaN, Infinity, -Infinity, 0, 1]);
    const r = tf.isInf(a);
    expect(r.dtype).toEqual('bool');
    expectArraysClose(await r.data(), [0, 1, 1, 0, 0]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(NaN);
    const dy = tf.scalar(3);

    const gradients = tf.grad(a => tf.isInf(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [0]);
  });

  it('gradients: Tensor1D', async () => {
    const a = tf.tensor1d([NaN, Infinity, -Infinity, 0, 1]);
    const dy = tf.tensor1d([1, 1, 1, 1, 1]);

    const gradients = tf.grad(a => tf.isInf(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [0, 0, 0, 0, 0]);
  });

  it('gradients: Tensor2D', async () => {
    const a = tf.tensor2d([NaN, Infinity, -Infinity, 0], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = tf.grad(a => tf.isInf(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [0, 0, 0, 0]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.isInf({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'isInf' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const r = tf.isInf([NaN, Infinity, -Infinity, 0, 1]);
    expectArraysClose(await r.data(), [0, 1, 1, 0, 0]);
  });

  it('throws for string tensor', () => {
    expect(() => tf.isInf('q'))
        .toThrowError(/Argument 'x' passed to 'isInf' must be numeric/);
  });
});

describeWithFlags('isFinite', ALL_ENVS, () => {
  it('basic', async () => {
    const a = tf.tensor1d([NaN, Infinity, -Infinity, 0, 1]);
    const r = tf.isFinite(a);
    expect(r.dtype).toEqual('bool');
    expectArraysClose(await r.data(), [0, 0, 0, 1, 1]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(NaN);
    const dy = tf.scalar(3);

    const gradients = tf.grad(a => tf.isFinite(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [0]);
  });

  it('gradients: Tensor1D', async () => {
    const a = tf.tensor1d([NaN, Infinity, -Infinity, 0, 1]);
    const dy = tf.tensor1d([1, 1, 1, 1, 1]);

    const gradients = tf.grad(a => tf.isFinite(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [0, 0, 0, 0, 0]);
  });

  it('gradients: Tensor2D', async () => {
    const a = tf.tensor2d([NaN, Infinity, -Infinity, 0], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = tf.grad(a => tf.isFinite(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [0, 0, 0, 0]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.isFinite({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'isFinite' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const r = tf.isFinite([NaN, Infinity, -Infinity, 0, 1]);
    expectArraysClose(await r.data(), [0, 0, 0, 1, 1]);
  });

  it('throws for string tensor', () => {
    expect(() => tf.isFinite('q'))
        .toThrowError(/Argument 'x' passed to 'isFinite' must be numeric/);
  });
});

describeWithFlags('round', ALL_ENVS, () => {
  it('basic', async () => {
    const a = tf.tensor1d([0.9, 2.5, 2.3, 1.5, -4.5]);
    const r = a.round();

    expectArraysClose(await r.data(), [1, 2, 2, 2, -4]);
  });

  it('propagates NaNs', async () => {
    const a = tf.tensor1d([1.5, NaN, -1.4]);
    const r = tf.round(a);
    expectArraysClose(await r.data(), [2, NaN, -1]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(5.2);
    const dy = tf.scalar(3);

    const gradients = tf.grad(a => tf.round(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [0]);
  });

  it('gradient with clones', async () => {
    const a = tf.scalar(5.2);
    const dy = tf.scalar(3);

    const gradients = tf.grad(a => tf.round(a.clone()).clone())(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [0]);
  });

  it('gradients: Tensor1D', async () => {
    const a = tf.tensor1d([-1.1, 2.6, 3, -5.9]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const gradients = tf.grad(a => tf.round(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [0, 0, 0, 0]);
  });

  it('gradients: Tensor2D', async () => {
    const a = tf.tensor2d([-3, 1, 2.2, 3], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = tf.grad(a => tf.round(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [0, 0, 0, 0]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.round({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'round' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const r = tf.round([0.9, 2.5, 2.3, 1.5, -4.5]);
    expectArraysClose(await r.data(), [1, 2, 2, 2, -4]);
  });

  it('throws for string tensor', () => {
    expect(() => tf.round('q'))
        .toThrowError(/Argument 'x' passed to 'round' must be numeric/);
  });
});
