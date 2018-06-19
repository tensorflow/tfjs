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

import * as tf from '../index';
import {describeWithFlags} from '../jasmine_util';
// tslint:disable-next-line:max-line-length
import {ALL_ENVS, expectArraysClose, expectNumbersClose} from '../test_util';
import * as util from '../util';

import * as selu_util from './selu_util';

describeWithFlags('relu', ALL_ENVS, () => {
  it('basic', () => {
    const a = tf.tensor1d([1, -2, 0, 3, -0.1]);
    const result = tf.relu(a);
    expectArraysClose(result, [1, 0, 0, 3, 0]);
  });

  it('5D', () => {
    const a = tf.tensor5d([1, -2, 5, -3], [1, 2, 2, 1, 1]);
    const result = tf.relu(a);
    expectArraysClose(result, [1, 0, 5, 0]);
  });

  it('6D', () => {
    const a = tf.tensor6d([1, -2, 5, -3, -1, 4, 7, 8], [1, 2, 2, 2, 1, 1]);
    const result = tf.relu(a);
    expectArraysClose(result, [1, 0, 5, 0, 0, 4, 7, 8]);
  });

  it('does nothing to positive values', () => {
    const a = tf.scalar(1);
    const result = tf.relu(a);
    expectNumbersClose(result.get(), 1);
  });

  it('sets negative values to 0', () => {
    const a = tf.scalar(-1);
    const result = tf.relu(a);
    expectNumbersClose(result.get(), 0);
  });

  it('preserves zero values', () => {
    const a = tf.scalar(0);
    const result = tf.relu(a);
    expectNumbersClose(result.get(), 0);
  });

  it('propagates NaNs, float32', () => {
    const a = tf.tensor1d([1, -2, 0, 3, -0.1, NaN]);
    const result = tf.relu(a);
    expect(result.dtype).toBe('float32');
    expectArraysClose(result, [1, 0, 0, 3, 0, NaN]);
  });

  it('gradients: positive scalar', () => {
    const a = tf.scalar(3);
    const dy = tf.scalar(5);

    const grad = tf.grad(a => tf.relu(a));
    const da = grad(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [5]);
  });

  it('gradients: negative scalar', () => {
    const a = tf.scalar(-3);
    const dy = tf.scalar(5);

    const grad = tf.grad(a => tf.relu(a));
    const da = grad(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [0]);
  });

  it('gradients: array', () => {
    const a = tf.tensor2d([1, -1, 0, .1], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const grad = tf.grad(a => tf.relu(a));
    const da = grad(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [1, 0, 0, 4]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.relu({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'relu' must be a Tensor/);
  });
});

describeWithFlags('abs', ALL_ENVS, () => {
  it('basic', () => {
    const a = tf.tensor1d([1, -2, 0, 3, -0.1]);
    const result = tf.abs(a);
    expectArraysClose(result, [1, 2, 0, 3, 0.1]);
  });

  it('5D', () => {
    const a = tf.tensor5d([1, -2, 0, -3], [1, 2, 2, 1, 1]);
    const result = tf.abs(a);
    expectArraysClose(result, [1, 2, 0, 3]);
  });

  it('6D', () => {
    const a = tf.tensor6d([1, -2, 5, -3, -1, 4, 7, 8], [1, 2, 2, 2, 1, 1]);
    const result = tf.abs(a);
    expectArraysClose(result, [1, 2, 5, 3, 1, 4, 7, 8]);
  });

  it('propagates NaNs', () => {
    const a = tf.tensor1d([1, -2, 0, 3, -0.1, NaN]);
    const result = tf.abs(a);
    expectArraysClose(result, [1, 2, 0, 3, 0.1, NaN]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(4);
    const dy = tf.scalar(8);

    const da = tf.grad(a => tf.abs(a))(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [8 * 1]);
  });

  it('gradients: Tensor1D', () => {
    const a = tf.tensor1d([1, 2, -3, 5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const da = tf.grad(a => tf.abs(a))(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [1 * 1, 2 * 1, 3 * -1, 4 * 1]);
  });

  it('gradients: Tensor2D', () => {
    const a = tf.tensor2d([3, -1, -2, 3], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const da = tf.grad(a => tf.abs(a))(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [1 * 1, 2 * -1, 3 * -1, 4 * 1]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.abs({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'abs' must be a Tensor/);
  });
});

describeWithFlags('step', ALL_ENVS, () => {
  it('with 1d tensor', () => {
    const a = tf.tensor1d([1, -2, -.01, 3, -0.1]);
    const result = tf.step(a);
    expectArraysClose(result, [1, 0, 0, 1, 0]);
  });

  it('with 1d tensor and alpha', () => {
    const a = tf.tensor1d([1, -2, -.01, 3, NaN]);
    const result = tf.step(a, 0.1);
    expectArraysClose(result, [1, 0.1, 0.1, 1, NaN]);
  });

  it('with 2d tensor', () => {
    const a = tf.tensor2d([1, -5, -3, 4], [2, 2]);
    const result = tf.step(a);
    expect(result.shape).toEqual([2, 2]);
    expectArraysClose(result, [1, 0, 0, 1]);
  });

  it('propagates NaNs', () => {
    const a = tf.tensor1d([1, -2, -.01, 3, NaN]);
    const result = tf.step(a);
    expectArraysClose(result, [1, 0, 0, 1, NaN]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(-4);
    const dy = tf.scalar(8);

    const gradients = tf.grad(a => tf.step(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [0]);
  });

  it('gradients: Tensor1D', () => {
    const a = tf.tensor1d([1, 2, -3, 5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const gradients = tf.grad(a => tf.step(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [0, 0, 0, 0]);
  });

  it('gradients: Tensor2D', () => {
    const a = tf.tensor2d([3, -1, -2, 3], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = tf.grad(a => tf.step(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [0, 0, 0, 0]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.step({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'step' must be a Tensor/);
  });
});

describeWithFlags('neg', ALL_ENVS, () => {
  it('basic', () => {
    const a = tf.tensor1d([1, -3, 2, 7, -4]);
    const result = tf.neg(a);
    expectArraysClose(result, [-1, 3, -2, -7, 4]);
  });

  it('propagate NaNs', () => {
    const a = tf.tensor1d([1, -3, 2, 7, NaN]);
    const result = tf.neg(a);
    const expected = [-1, 3, -2, -7, NaN];
    expectArraysClose(result, expected);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(4);
    const dy = tf.scalar(8);

    const da = tf.grad(a => tf.neg(a))(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [8 * -1]);
  });

  it('gradients: Tensor1D', () => {
    const a = tf.tensor1d([1, 2, -3, 5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const da = tf.grad(a => tf.neg(a))(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [1 * -1, 2 * -1, 3 * -1, 4 * -1]);
  });

  it('gradients: Tensor2D', () => {
    const a = tf.tensor2d([3, -1, -2, 3], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const da = tf.grad(a => tf.neg(a))(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [1 * -1, 2 * -1, 3 * -1, 4 * -1]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.neg({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'neg' must be a Tensor/);
  });
});

describeWithFlags('sigmoid', ALL_ENVS, () => {
  it('basic', () => {
    const values = [1, -3, 2, 7, -4];
    const a = tf.tensor1d(values);

    const result = tf.sigmoid(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = 1 / (1 + Math.exp(-values[i]));
    }
    expectArraysClose(result, expected);
  });

  it('6D', () => {
    const a = tf.ones([2, 2, 2, 2, 2, 2]);
    const result = tf.sigmoid(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = 1 / (1 + Math.exp(-1.0));
    }
    expectArraysClose(result, expected);
  });

  it('propagates NaNs', () => {
    const a = tf.tensor1d([3, NaN]);
    const res = tf.sigmoid(a);
    expectArraysClose(res, [1 / (1 + Math.exp(-3)), NaN]);
  });

  it('gradients: Tensor1D', () => {
    const a = tf.tensor1d([1, 2, -3, 5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const da = tf.grad(a => tf.sigmoid(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      const y = 1 / (1 + Math.exp(-a.get(i)));
      expected[i] = dy.get(i) * y * (1 - y);
    }

    expectArraysClose(da, expected);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.sigmoid({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'sigmoid' must be a Tensor/);
  });
});

describeWithFlags('logSigmoid', ALL_ENVS, () => {
  it('basic', () => {
    const values = [1, -3, 2, 7, -4];
    const a = tf.tensor1d(values);

    const result = tf.logSigmoid(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.log(1 / (1 + Math.exp(-values[i])));
    }
    expectArraysClose(result, expected);
  });

  it('scalar', () => {
    const a = tf.scalar(-2);

    const result = tf.logSigmoid(a);

    const expected = [Math.log(1 / (1 + Math.exp(2)))];
    expectArraysClose(result, expected);
  });

  it('tensor2D', () => {
    const values = [1, 2, -3, 5];
    const a = tf.tensor2d(values, [2, 2]);

    const result = tf.logSigmoid(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.log(1 / (1 + Math.exp(-values[i])));
    }
    expectArraysClose(result, expected);
  });

  it('larger magnitude negative inputs', () => {
    const values = [-100, -200, -3000];
    const a = tf.tensor1d(values);

    const result = tf.logSigmoid(a);

    const expected = [-100, -200, -3000];

    expectArraysClose(result, expected);
  });

  it('larger magnitude positive inputs', () => {
    const values = [100, 200, 3000, 50000];
    const a = tf.tensor1d(values);

    const result = tf.logSigmoid(a);

    const expected = [0, 0, 0, 0];

    expectArraysClose(result, expected);
  });

  it('propagates NaNs', () => {
    const a = tf.tensor1d([3, NaN]);
    const res = tf.logSigmoid(a);
    expectArraysClose(res, [Math.log(1 / (1 + Math.exp(-3))), NaN]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(3);
    const dy = tf.scalar(4);

    const da = tf.grad(a => tf.logSigmoid(a))(a, dy).get();
    const y = 1 / (1 + Math.exp(a.get()));
    expectNumbersClose(da, dy.get() * y);
  });

  it('gradients: Tensor1D', () => {
    const a = tf.tensor1d([1, 2, -3, 5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const da = tf.grad(a => tf.logSigmoid(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      const y = 1 / (1 + Math.exp(a.get(i)));
      expected[i] = dy.get(i) * y;
    }

    expectArraysClose(da, expected);
  });

  it('gradients: Tensor2D', () => {
    const a = tf.tensor2d([1, 2, -3, 5], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const da = tf.grad(a => tf.logSigmoid(a))(a, dy);

    const expected = [];
    const aVals = a.dataSync();
    const dyVals = dy.dataSync();
    for (let i = 0; i < a.size; i++) {
      const y = 1 / (1 + Math.exp(aVals[i]));
      expected[i] = dyVals[i] * y;
    }

    expectArraysClose(da, expected);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.logSigmoid({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'logSigmoid' must be a Tensor/);
  });
});

describeWithFlags('softplus', ALL_ENVS, () => {
  it('basic', () => {
    const values = [1, -3, 2, 7, -4];
    const a = tf.tensor1d(values);

    const result = tf.softplus(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.log((1 + Math.exp(values[i])));
    }
    expectArraysClose(result, expected);
  });

  it('scalar', () => {
    const a = tf.scalar(-2);

    const result = tf.softplus(a);

    const expected = [Math.log((1 + Math.exp(-2)))];
    expectArraysClose(result, expected);
  });

  it('tensor2D', () => {
    const values = [1, 2, -3, 5];
    const a = tf.tensor2d(values, [2, 2]);

    const result = tf.softplus(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.log((1 + Math.exp(values[i])));
    }
    expectArraysClose(result, expected);
  });

  it('larger magnitude negative inputs', () => {
    const values = [-100, -200, -3000, -50000];
    const a = tf.tensor1d(values);

    const result = tf.softplus(a);

    const expected = [0, 0, 0, 0];

    expectArraysClose(result, expected);
  });

  it('larger magnitude positive inputs', () => {
    const values = [100, 200, 3000];
    const a = tf.tensor1d(values);

    const result = tf.softplus(a);

    const expected = [100, 200, 3000];

    expectArraysClose(result, expected);
  });

  it('propagates NaNs', () => {
    const a = tf.tensor1d([3, NaN]);
    const res = tf.softplus(a);
    expectArraysClose(res, [Math.log((1 + Math.exp(3))), NaN]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(3);
    const dy = tf.scalar(4);

    const da = tf.grad(a => tf.softplus(a))(a, dy);
    const y = 1 / (1 + Math.exp(-a.get()));

    expectNumbersClose(da.get(), dy.get() * y);
  });

  it('gradients: Tensor1D', () => {
    const a = tf.tensor1d([1, 2, -3, 5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const da = tf.grad(a => tf.softplus(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      const y = 1 / (1 + Math.exp(-a.get(i)));
      expected[i] = dy.get(i) * y;
    }

    expectArraysClose(da, expected);
  });

  it('gradients: Tensor2D', () => {
    const a = tf.tensor2d([1, 2, -3, 5], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const da = tf.grad(a => tf.softplus(a))(a, dy);

    const expected = [];
    const aVals = a.dataSync();
    const dyVals = dy.dataSync();

    for (let i = 0; i < a.size; i++) {
      const y = 1 / (1 + Math.exp(-aVals[i]));
      expected[i] = dyVals[i] * y;
    }

    expectArraysClose(da, expected);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.softplus({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'softplus' must be a Tensor/);
  });
});

describeWithFlags('sqrt', ALL_ENVS, () => {
  it('sqrt', () => {
    const a = tf.tensor1d([2, 4]);
    const r = tf.sqrt(a);
    expectNumbersClose(r.get(0), Math.sqrt(2));
    expectNumbersClose(r.get(1), Math.sqrt(4));
  });

  it('sqrt propagates NaNs', () => {
    const a = tf.tensor1d([1, NaN]);
    const r = tf.sqrt(a);
    expectArraysClose(r, [Math.sqrt(1), NaN]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(4);
    const dy = tf.scalar(8);

    const da = tf.grad(a => tf.sqrt(a))(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [8 / (2 * Math.sqrt(4))]);
  });

  it('gradients: Tensor1D', () => {
    const a = tf.tensor1d([1, 2, 3, 5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const gradients = tf.grad(a => tf.sqrt(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        gradients,
        [
          1 / (2 * Math.sqrt(1)), 2 / (2 * Math.sqrt(2)),
          3 / (2 * Math.sqrt(3)), 4 / (2 * Math.sqrt(5))
        ],
        1e-1);
  });

  it('gradients: Tensor2D', () => {
    const a = tf.tensor2d([3, 1, 2, 3], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = tf.grad(a => tf.sqrt(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        gradients,
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
});

describeWithFlags('rsqrt', ALL_ENVS, () => {
  it('rsqrt', () => {
    const a = tf.tensor1d([2, 4]);
    const r = tf.rsqrt(a);
    expectNumbersClose(r.get(0), 1 / Math.sqrt(2));
    expectNumbersClose(r.get(1), 1 / Math.sqrt(4));
  });

  it('rsqrt propagates NaNs', () => {
    const a = tf.tensor1d([1, NaN]);
    const r = tf.rsqrt(a);
    expectArraysClose(r, [1 / Math.sqrt(1), NaN]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(4);
    const dy = tf.scalar(8);

    const da = tf.grad(a => tf.rsqrt(a))(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [(-1 * 8) / (2 * Math.pow(4, 1.5))]);
  });

  it('gradients: Tensor1D', () => {
    const a = tf.tensor1d([1, 2, 3, 5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const gradients = tf.grad(a => tf.rsqrt(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        gradients,
        [
          -1 * 1 / (2 * Math.pow(1, 1.5)), -1 * 2 / (2 * Math.pow(2, 1.5)),
          -1 * 3 / (2 * Math.pow(3, 1.5)), -1 * 4 / (2 * Math.pow(5, 1.5))
        ],
        1e-1);
  });

  it('gradients: Tensor2D', () => {
    const a = tf.tensor2d([3, 1, 2, 3], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = tf.grad(a => tf.rsqrt(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        gradients,
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
});

describeWithFlags('square', ALL_ENVS, () => {
  it('1D array', () => {
    const a = tf.tensor1d([2, 4, Math.sqrt(2)]);
    const r = tf.square(a);
    expectArraysClose(r, [4, 16, 2]);
  });

  it('2D array', () => {
    const a = tf.tensor2d([1, 2, Math.sqrt(2), Math.sqrt(3)], [2, 2]);
    const r = tf.square(a);
    expect(r.shape).toEqual([2, 2]);
    expectArraysClose(r, [1, 4, 2, 3]);
  });

  it('5D array', () => {
    const a = tf.tensor5d([1, 2, Math.sqrt(2), Math.sqrt(3)], [1, 1, 2, 2, 1]);
    const r = tf.square(a);
    expect(r.shape).toEqual([1, 1, 2, 2, 1]);
    expectArraysClose(r, [1, 4, 2, 3]);
  });

  it('6D array', () => {
    const a = tf.tensor6d(
        [1, 2, Math.sqrt(2), Math.sqrt(3), 3, 4, Math.sqrt(7), Math.sqrt(13)],
        [1, 1, 2, 2, 2, 1]);
    const r = tf.square(a);
    expect(r.shape).toEqual(a.shape);
    expectArraysClose(r, [1, 4, 2, 3, 9, 16, 7, 13]);
  });

  it('square propagates NaNs', () => {
    const a = tf.tensor1d([1.5, NaN]);
    const r = tf.square(a);
    expectArraysClose(r, [2.25, NaN]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(5);
    const dy = tf.scalar(8);

    const gradients = tf.grad(a => tf.square(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [2 * 5 * 8]);
  });

  it('gradients: Tensor1D', () => {
    const a = tf.tensor1d([-1, 2, 3, -5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const gradients = tf.grad(a => tf.square(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [-2, 4 * 2, 6 * 3, -10 * 4]);
  });

  it('gradients: Tensor2D', () => {
    const a = tf.tensor2d([-3, 1, 2, 3], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = tf.grad(a => tf.square(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [-6 * 1, 2 * 2, 4 * 3, 6 * 4]);
  });

  it('gradients: Tensor5D', () => {
    const a = tf.tensor5d([-3, 1, 2, 3], [1, 1, 1, 2, 2]);
    const dy = tf.tensor5d([1, 2, 3, 4], [1, 1, 1, 2, 2]);

    const gradients = tf.grad(a => tf.square(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [-6 * 1, 2 * 2, 4 * 3, 6 * 4]);
  });

  it('gradients: Tensor6D', () => {
    const a = tf.tensor6d([-3, 1, 2, 3, -4, 5, 12, 3], [1, 1, 1, 2, 2, 2]);
    const dy = tf.tensor6d([1, 2, 3, 4, 5, 6, 7, 8], [1, 1, 1, 2, 2, 2]);

    const gradients = tf.grad(a => tf.square(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        gradients,
        [-6 * 1, 2 * 2, 4 * 3, 6 * 4, -8 * 5, 10 * 6, 24 * 7, 6 * 8]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.square({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'square' must be a Tensor/);
  });
});

describeWithFlags('reciprocal', ALL_ENVS, () => {
  it('1D array', () => {
    const a = tf.tensor1d([2, 3, 0, NaN]);
    const r = tf.reciprocal(a);
    expectArraysClose(r, [1 / 2, 1 / 3, Infinity, NaN]);
  });

  it('2D array', () => {
    const a = tf.tensor2d([1, Infinity, 0, NaN], [2, 2]);
    const r = tf.reciprocal(a);
    expect(r.shape).toEqual([2, 2]);
    expectArraysClose(r, [1 / 1, 0, Infinity, NaN]);
  });

  it('reciprocal propagates NaNs', () => {
    const a = tf.tensor1d([1.5, NaN]);
    const r = tf.reciprocal(a);
    expectArraysClose(r, [1 / 1.5, NaN]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(5);
    const dy = tf.scalar(8);

    const gradients = tf.grad(a => tf.reciprocal(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [-1 * 8 * (1 / (5 * 5))]);
  });

  it('gradients: Tensor1D', () => {
    const a = tf.tensor1d([-1, 2, 3, -5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const gradients = tf.grad(a => tf.reciprocal(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [
      -1 * 1 * (1 / (-1 * -1)), -1 * 2 * (1 / (2 * 2)), -1 * 3 * (1 / (3 * 3)),
      -1 * 4 * (1 / (-5 * -5))
    ]);
  });

  it('gradients: Tensor2D', () => {
    const a = tf.tensor2d([-1, 2, 3, -5], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = tf.grad(a => tf.reciprocal(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [
      -1 * 1 * (1 / (-1 * -1)), -1 * 2 * (1 / (2 * 2)), -1 * 3 * (1 / (3 * 3)),
      -1 * 4 * (1 / (-5 * -5))
    ]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.reciprocal({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'reciprocal' must be a Tensor/);
  });
});

describeWithFlags('log', ALL_ENVS, () => {
  it('log', () => {
    const a = tf.tensor1d([1, 2]);
    const r = tf.log(a);
    expectNumbersClose(r.get(0), Math.log(1));
    expectNumbersClose(r.get(1), Math.log(2));
  });

  it('log 6D', () => {
    const a = tf.range(1, 65).reshape([2, 2, 2, 2, 2, 2]);
    const r = tf.log(a);

    const expectedResult = [];
    for (let i = 1; i < 65; i++) {
      expectedResult[i - 1] = Math.log(i);
    }

    expectArraysClose(r, expectedResult);
  });

  it('log propagates NaNs', () => {
    const a = tf.tensor1d([1, NaN]);
    const r = tf.log(a);
    expectArraysClose(r, [Math.log(1), NaN]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(5);
    const dy = tf.scalar(3);

    const gradients = tf.grad(a => tf.log(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [3 / 5]);
  });

  it('gradients: Tensor1D', () => {
    const a = tf.tensor1d([-1, 2, 3, -5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const gradients = tf.grad(a => tf.log(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [1 / -1, 2 / 2, 3 / 3, 4 / -5]);
  });

  it('gradients: Tensor2D', () => {
    const a = tf.tensor2d([-3, 1, 2, 3], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = tf.grad(a => tf.log(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [1 / -3, 2 / 1, 3 / 2, 4 / 3]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.log({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'log' must be a Tensor/);
  });
});

describeWithFlags('log1p', ALL_ENVS, () => {
  it('log1p', () => {
    const a = tf.tensor1d([1, 2]);
    const r = tf.log1p(a);
    expectNumbersClose(r.get(0), Math.log1p(1));
    expectNumbersClose(r.get(1), Math.log1p(2));
  });

  it('log1p propagates NaNs', () => {
    const a = tf.tensor1d([1, NaN]);
    const r = tf.log1p(a);
    expectArraysClose(r, [Math.log1p(1), NaN]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(5);
    const dy = tf.scalar(3);

    const gradients = tf.grad(a => tf.log1p(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [3 / (1 + 5)]);
  });

  it('gradients: Tensor1D', () => {
    const a = tf.tensor1d([-1, 2, 3, -5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const gradients = tf.grad(a => tf.log1p(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        gradients, [Infinity, 2 / (1 + 2), 3 / (1 + 3), 4 / (1 + -5)]);
  });

  it('gradients: Tensor2D', () => {
    const a = tf.tensor2d([-3, 1, 2, 3], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = tf.grad(a => tf.log1p(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        gradients, [1 / (1 + -3), 2 / (1 + 1), 3 / (1 + 2), 4 / (1 + 3)]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.log1p({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'log1p' must be a Tensor/);
  });
});

describeWithFlags('ceil', ALL_ENVS, () => {
  it('basic', () => {
    const a = tf.tensor1d([1.5, 2.1, -1.4]);
    const r = tf.ceil(a);
    expectNumbersClose(r.get(0), 2);
    expectNumbersClose(r.get(1), 3);
    expectNumbersClose(r.get(2), -1);
  });

  it('propagates NaNs', () => {
    const a = tf.tensor1d([1.5, NaN, -1.4]);
    const r = tf.ceil(a);
    expectArraysClose(r, [2, NaN, -1]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(5.2);
    const dy = tf.scalar(3);

    const gradients = tf.grad(a => tf.ceil(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [0]);
  });

  it('gradients: Tensor1D', () => {
    const a = tf.tensor1d([-1.1, 2.6, 3, -5.9]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const gradients = tf.grad(a => tf.ceil(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [0, 0, 0, 0]);
  });

  it('gradients: Tensor2D', () => {
    const a = tf.tensor2d([-3, 1, 2.2, 3], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = tf.grad(a => tf.ceil(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [0, 0, 0, 0]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.ceil({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'ceil' must be a Tensor/);
  });
});

describeWithFlags('floor', ALL_ENVS, () => {
  it('basic', () => {
    const a = tf.tensor1d([1.5, 2.1, -1.4]);
    const r = tf.floor(a);

    expectNumbersClose(r.get(0), 1);
    expectNumbersClose(r.get(1), 2);
    expectNumbersClose(r.get(2), -2);
  });

  it('propagates NaNs', () => {
    const a = tf.tensor1d([1.5, NaN, -1.4]);
    const r = tf.floor(a);
    expectArraysClose(r, [1, NaN, -2]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(5.2);
    const dy = tf.scalar(3);

    const gradients = tf.grad(a => tf.floor(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [0]);
  });

  it('gradients: Tensor1D', () => {
    const a = tf.tensor1d([-1.1, 2.6, 3, -5.9]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const gradients = tf.grad(a => tf.floor(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [0, 0, 0, 0]);
  });

  it('gradients: Tensor2D', () => {
    const a = tf.tensor2d([-3, 1, 2.2, 3], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = tf.grad(a => tf.floor(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [0, 0, 0, 0]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.floor({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'floor' must be a Tensor/);
  });
});

describeWithFlags('sign', ALL_ENVS, () => {
  it('basic', () => {
    const a = tf.tensor1d([1.5, 0, NaN, -1.4]);
    const r = tf.sign(a);
    expectNumbersClose(r.get(0), 1);
    expectNumbersClose(r.get(1), 0);
    expectNumbersClose(r.get(2), 0);
    expectNumbersClose(r.get(3), -1);
  });

  it('propagates NaNs', () => {
    const a = tf.tensor1d([1.5, NaN, -1.4]);
    const r = tf.sign(a);
    expectArraysClose(r, [1, 0, -1]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(5.2);
    const dy = tf.scalar(3);

    const gradients = tf.grad(a => tf.sign(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [0]);
  });

  it('gradients: Tensor1D', () => {
    const a = tf.tensor1d([-1.1, 2.6, 3, -5.9]);
    const dy = tf.tensor1d([-1, 1, 1, -1]);

    const gradients = tf.grad(a => tf.sign(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [0, 0, 0, 0]);
  });

  it('gradients: Tensor2D', () => {
    const a = tf.tensor2d([-3, 1, 2.2, 3], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = tf.grad(a => tf.sign(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [0, 0, 0, 0]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.sign({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'sign' must be a Tensor/);
  });
});

describeWithFlags('exp', ALL_ENVS, () => {
  it('exp', () => {
    const a = tf.tensor1d([1, 2, 0]);
    const r = tf.exp(a);

    expectNumbersClose(r.get(0), Math.exp(1));
    expectNumbersClose(r.get(1), Math.exp(2));
    expectNumbersClose(r.get(2), 1);
  });

  it('exp propagates NaNs', () => {
    const a = tf.tensor1d([1, NaN, 0]);
    const r = tf.exp(a);
    expectArraysClose(r, [Math.exp(1), NaN, 1]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(0.5);
    const dy = tf.scalar(3);

    const gradients = tf.grad(a => tf.exp(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [3 * Math.exp(0.5)]);
  });

  it('gradients: Tensor1D', () => {
    const a = tf.tensor1d([-1, 2, 3, -5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const gradients = tf.grad(a => tf.exp(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        gradients,
        [1 * Math.exp(-1), 2 * Math.exp(2), 3 * Math.exp(3), 4 * Math.exp(-5)],
        1e-1);
  });

  it('gradients: Tensor2D', () => {
    const a = tf.tensor2d([-3, 1, 2, 3], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = tf.grad(a => tf.exp(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        gradients,
        [1 * Math.exp(-3), 2 * Math.exp(1), 3 * Math.exp(2), 4 * Math.exp(3)],
        1e-1);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.exp({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'exp' must be a Tensor/);
  });
});

describeWithFlags('expm1', ALL_ENVS, () => {
  it('expm1', () => {
    const a = tf.tensor1d([1, 2, 0]);
    const r = tf.expm1(a);

    expectNumbersClose(r.get(0), Math.expm1(1));
    expectNumbersClose(r.get(1), Math.expm1(2));
    expectNumbersClose(r.get(2), Math.expm1(0));
  });

  it('expm1 propagates NaNs', () => {
    const a = tf.tensor1d([1, NaN, 0]);
    const r = tf.expm1(a);
    expectArraysClose(r, [Math.expm1(1), NaN, Math.expm1(0)]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(0.5);
    const dy = tf.scalar(3);

    const gradients = tf.grad(a => tf.expm1(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [3 * Math.exp(0.5)]);
  });

  it('gradients: Tensor1D', () => {
    const a = tf.tensor1d([-1, 2, 3, -5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const gradients = tf.grad(a => tf.expm1(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        gradients,
        [1 * Math.exp(-1), 2 * Math.exp(2), 3 * Math.exp(3), 4 * Math.exp(-5)],
        1e-1);
  });

  it('gradients: Tensor2D', () => {
    const a = tf.tensor2d([-3, 1, 2, 3], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = tf.grad(a => tf.expm1(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        gradients,
        [1 * Math.exp(-3), 2 * Math.exp(1), 3 * Math.exp(2), 4 * Math.exp(3)],
        1e-1);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.expm1({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'expm1' must be a Tensor/);
  });
});

describeWithFlags('sin', ALL_ENVS, () => {
  it('basic', () => {
    const values = [1, -3, 2, 7, -4];
    const a = tf.tensor1d(values);
    const result = tf.sin(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.sin(values[i]);
    }
    expectArraysClose(result, expected);
  });

  it('propagates NaNs', () => {
    const a = tf.tensor1d([4, NaN, 0]);
    const res = tf.sin(a);
    expectArraysClose(res, [Math.sin(4), NaN, Math.sin(0)]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(5);
    const dy = tf.scalar(8);

    const gradients = tf.grad(a => tf.sin(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [8 * Math.cos(5)]);
  });

  it('gradients: Tensor1D', () => {
    const a = tf.tensor1d([-1, 2, 3, -5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const gradients = tf.grad(a => tf.sin(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        gradients,
        [1 * Math.cos(-1), 2 * Math.cos(2), 3 * Math.cos(3), 4 * Math.cos(-5)],
        1e-1);
  });

  it('gradients: Tensor2D', () => {
    const a = tf.tensor2d([-3, 1, 2, 3], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = tf.grad(a => tf.sin(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        gradients,
        [1 * Math.cos(-3), 2 * Math.cos(1), 3 * Math.cos(2), 4 * Math.cos(3)],
        1e-1);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.sin({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'sin' must be a Tensor/);
  });
});

describeWithFlags('cos', ALL_ENVS, () => {
  it('basic', () => {
    const values = [1, -3, 2, 7, -4];
    const a = tf.tensor1d(values);
    const result = tf.cos(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.cos(values[i]);
    }
    expectArraysClose(result, expected);
  });

  it('propagates NaNs', () => {
    const a = tf.tensor1d([4, NaN, 0]);
    const res = tf.cos(a);
    expectArraysClose(res, [Math.cos(4), NaN, Math.cos(0)]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(5);
    const dy = tf.scalar(8);

    const gradients = tf.grad(a => tf.cos(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [8 * Math.sin(5) * -1]);
  });

  it('gradients: Tensor1D', () => {
    const a = tf.tensor1d([-1, 2, 3, -5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const gradients = tf.grad(a => tf.cos(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        gradients,
        [
          1 * Math.sin(-1) * -1, 2 * Math.sin(2) * -1, 3 * Math.sin(3) * -1,
          4 * Math.sin(-5) * -1
        ],
        1e-1);
  });

  it('gradients: Tensor2D', () => {
    const a = tf.tensor2d([-3, 1, 2, 3], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = tf.grad(a => tf.cos(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        gradients,
        [
          1 * Math.sin(-3) * -1, 2 * Math.sin(1) * -1, 3 * Math.sin(2) * -1,
          4 * Math.sin(3) * -1
        ],
        1e-1);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.cos({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'cos' must be a Tensor/);
  });
});

describeWithFlags('tan', ALL_ENVS, () => {
  it('basic', () => {
    const values = [1, -3, 2, 7, -4];
    const a = tf.tensor1d(values);
    const result = tf.tan(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.tan(values[i]);
    }
    expectArraysClose(result, expected);
  });

  it('propagates NaNs', () => {
    const a = tf.tensor1d([4, NaN, 0]);
    const res = tf.tan(a);
    expectArraysClose(res, [Math.tan(4), NaN, Math.tan(0)]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(0.5);
    const dy = tf.scalar(8);

    const gradients = tf.grad(a => tf.tan(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [8 / (Math.cos(0.5) * Math.cos(0.5))]);
  });

  it('gradients: Tensor1D', () => {
    const aValues = [-1, 2, 3, -5];
    const dyValues = [1, 2, 3, 4];
    const a = tf.tensor1d(aValues);
    const dy = tf.tensor1d(dyValues);

    const gradients = tf.grad(a => tf.tan(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = dyValues[i] / (Math.cos(aValues[i]) * Math.cos(aValues[i]));
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, expected);
  });

  it('gradients: Tensor2D', () => {
    const aValues = [-3, 1, 2, 3];
    const dyValues = [1, 2, 3, 4];
    const a = tf.tensor2d(aValues, [2, 2]);
    const dy = tf.tensor2d(dyValues, [2, 2]);

    const gradients = tf.grad(a => tf.tan(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = dyValues[i] / (Math.cos(aValues[i]) * Math.cos(aValues[i]));
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, expected);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.tan({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'tan' must be a Tensor/);
  });
});

describeWithFlags('asin', ALL_ENVS, () => {
  it('basic', () => {
    const values = [.1, -3, 2, 7, -4];
    const a = tf.tensor1d(values);
    const result = tf.asin(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.asin(values[i]);
    }
    expectArraysClose(result, expected);
  });

  it('propagates NaNs', () => {
    const a = tf.tensor1d([4, NaN, 0]);
    const res = tf.asin(a);
    expectArraysClose(res, [Math.asin(4), NaN, Math.asin(0)]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(0.5);
    const dy = tf.scalar(8);

    const gradients = tf.grad(a => tf.asin(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [8 / Math.sqrt(1 - (0.5 * 0.5))]);
  });

  it('gradients: Tensor1D', () => {
    const aValues = [-0.1, 0.2, 0.3, -0.5];
    const dyValues = [1, 2, 3, 4];
    const a = tf.tensor1d(aValues);
    const dy = tf.tensor1d(dyValues);

    const gradients = tf.grad(a => tf.asin(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = dyValues[i] / Math.sqrt(1 - (aValues[i] * aValues[i]));
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, expected);
  });

  it('gradients: Tensor2D', () => {
    const aValues = [-0.3, 0.1, 0.2, 0.3];
    const dyValues = [1, 2, 3, 4];
    const a = tf.tensor2d(aValues, [2, 2]);
    const dy = tf.tensor2d(dyValues, [2, 2]);

    const gradients = tf.grad(a => tf.asin(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = dyValues[i] / Math.sqrt(1 - (aValues[i] * aValues[i]));
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, expected);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.asin({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'asin' must be a Tensor/);
  });
});

describeWithFlags('acos', ALL_ENVS, () => {
  it('basic', () => {
    const values = [.1, -3, 2, 7, -4];
    const a = tf.tensor1d(values);
    const result = tf.acos(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.acos(values[i]);
    }
    expectArraysClose(result, expected);
  });

  it('propagates NaNs', () => {
    const a = tf.tensor1d([4, NaN, 0]);
    const res = tf.acos(a);
    expectArraysClose(res, [Math.acos(4), NaN, Math.acos(0)]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(0.5);
    const dy = tf.scalar(8);

    const gradients = tf.grad(a => tf.acos(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [(-1 * 8) / Math.sqrt(1 - (0.5 * 0.5))]);
  });

  it('gradients: Tensor1D', () => {
    const aValues = [-0.1, 0.2, 0.3, -0.5];
    const dyValues = [1, 2, 3, 4];
    const a = tf.tensor1d(aValues);
    const dy = tf.tensor1d(dyValues);

    const gradients = tf.grad(a => tf.acos(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] =
          (-1 * dyValues[i]) / Math.sqrt(1 - (aValues[i] * aValues[i]));
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, expected);
  });

  it('gradients: Tensor2D', () => {
    const aValues = [-0.3, 0.1, 0.2, 0.3];
    const dyValues = [1, 2, 3, 4];
    const a = tf.tensor2d(aValues, [2, 2]);
    const dy = tf.tensor2d(dyValues, [2, 2]);

    const gradients = tf.grad(a => tf.acos(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] =
          (-1 * dyValues[i]) / Math.sqrt(1 - (aValues[i] * aValues[i]));
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, expected);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.acos({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'acos' must be a Tensor/);
  });
});

describeWithFlags('atan', ALL_ENVS, () => {
  it('basic', () => {
    const values = [1, -3, 2, 7, -4];
    const a = tf.tensor1d(values);
    const result = tf.atan(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.atan(values[i]);
    }
    expectArraysClose(result, expected);
  });

  it('6D atan', () => {
    const a = tf.range(1, 65).reshape([2, 2, 2, 2, 2, 2]);
    const result = tf.atan(a);

    const expected = [];
    for (let i = 1; i < 65; ++i) {
      expected[i - 1] = Math.atan(i);
    }
    expectArraysClose(result, expected);
  });

  it('propagates NaNs', () => {
    const a = tf.tensor1d([4, NaN, 0]);
    const res = tf.atan(a);
    expectArraysClose(res, [Math.atan(4), NaN, Math.atan(0)]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(0.5);
    const dy = tf.scalar(8);

    const gradients = tf.grad(a => tf.atan(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [8 / (1 + (0.5 * 0.5))]);
  });

  it('gradients: Tensor1D', () => {
    const aValues = [-0.1, 0.2, 0.3, -0.5];
    const dyValues = [1, 2, 3, 4];
    const a = tf.tensor1d(aValues);
    const dy = tf.tensor1d(dyValues);

    const gradients = tf.grad(a => tf.atan(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = dyValues[i] / (1 + (aValues[i] * aValues[i]));
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, expected);
  });

  it('gradients: Tensor2D', () => {
    const aValues = [-0.3, 0.1, 0.2, 0.3];
    const dyValues = [1, 2, 3, 4];
    const a = tf.tensor2d(aValues, [2, 2]);
    const dy = tf.tensor2d(dyValues, [2, 2]);

    const gradients = tf.grad(a => tf.atan(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = dyValues[i] / (1 + (aValues[i] * aValues[i]));
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, expected);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.atan({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'atan' must be a Tensor/);
  });
});

describeWithFlags('sinh', ALL_ENVS, () => {
  it('basic', () => {
    const values = [1, -3, 2, -1, -4];
    const a = tf.tensor1d(values);
    const result = tf.sinh(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.sinh(values[i]);
    }
    expectArraysClose(result, expected);
  });

  it('propagates NaNs', () => {
    const a = tf.tensor1d([4, NaN, 0]);
    const res = tf.sinh(a);
    expectArraysClose(res, [Math.sinh(4), NaN, Math.sinh(0)]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(0.5);
    const dy = tf.scalar(8);

    const gradients = tf.grad(a => tf.sinh(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [8 * Math.cosh(0.5)]);
  });

  it('gradients: Tensor1D', () => {
    const aValues = [-1, 2, 3, -5];
    const dyValues = [1, 2, 3, 4];
    const a = tf.tensor1d(aValues);
    const dy = tf.tensor1d(dyValues);

    const gradients = tf.grad(a => tf.sinh(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = dyValues[i] * Math.cosh(aValues[i]);
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, expected);
  });

  it('gradients: Tensor2D', () => {
    const aValues = [-3, 1, 2, 3];
    const dyValues = [1, 2, 3, 4];
    const a = tf.tensor2d(aValues, [2, 2]);
    const dy = tf.tensor2d(dyValues, [2, 2]);

    const gradients = tf.grad(a => tf.sinh(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = dyValues[i] * Math.cosh(aValues[i]);
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, expected);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.sinh({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'sinh' must be a Tensor/);
  });
});

describeWithFlags('cosh', ALL_ENVS, () => {
  it('basic', () => {
    const values = [1, -3, 2, -1, -4];
    const a = tf.tensor1d(values);
    const result = tf.cosh(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.cosh(values[i]);
    }

    // TODO(nsthorat): Fix the precision problem here.
    expectArraysClose(result, expected);
  });

  it('propagates NaNs', () => {
    const a = tf.tensor1d([4, NaN, 0]);
    const res = tf.cosh(a);
    expectArraysClose(res, [Math.cosh(4), NaN, Math.cosh(0)]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(0.5);
    const dy = tf.scalar(8);

    const gradients = tf.grad(a => tf.cosh(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [8 * Math.sinh(0.5)]);
  });

  it('gradients: Tensor1D', () => {
    const aValues = [-1, 2, 3, -5];
    const dyValues = [1, 2, 3, 4];
    const a = tf.tensor1d(aValues);
    const dy = tf.tensor1d(dyValues);

    const gradients = tf.grad(a => tf.cosh(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = dyValues[i] * Math.sinh(aValues[i]);
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, expected);
  });

  it('gradients: Tensor2D', () => {
    const aValues = [-3, 1, 2, 3];
    const dyValues = [1, 2, 3, 4];
    const a = tf.tensor2d(aValues, [2, 2]);
    const dy = tf.tensor2d(dyValues, [2, 2]);

    const gradients = tf.grad(a => tf.cosh(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = dyValues[i] * Math.sinh(aValues[i]);
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, expected);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.cosh({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'cosh' must be a Tensor/);
  });
});

describeWithFlags('tanh', ALL_ENVS, () => {
  it('basic', () => {
    const values = [1, -3, 2, 7, -4];
    const a = tf.tensor1d(values);
    const result = tf.tanh(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = util.tanh(values[i]);
    }
    expectArraysClose(result, expected);
  });

  it('propagates NaNs', () => {
    const a = tf.tensor1d([4, NaN, 0]);
    const res = tf.tanh(a);
    expectArraysClose(res, [util.tanh(4), NaN, util.tanh(0)]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(0.5);
    const dy = tf.scalar(8);

    const gradients = tf.grad(a => tf.tanh(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [8 * (1 - (Math.tanh(0.5) * Math.tanh(0.5)))]);
  });

  it('gradients: Tensor1D', () => {
    const aValues = [-1, 2, 3, -5];
    const dyValues = [1, 2, 3, 4];
    const a = tf.tensor1d(aValues);
    const dy = tf.tensor1d(dyValues);

    const gradients = tf.grad(a => tf.tanh(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] =
          dyValues[i] * (1 - (Math.tanh(aValues[i]) * Math.tanh(aValues[i])));
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, expected);
  });

  it('gradients: Tensor2D', () => {
    const aValues = [-3, 1, 2, 3];
    const dyValues = [1, 2, 3, 4];
    const a = tf.tensor2d(aValues, [2, 2]);
    const dy = tf.tensor2d(dyValues, [2, 2]);

    const gradients = tf.grad(a => tf.tanh(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] =
          dyValues[i] * (1 - (Math.tanh(aValues[i]) * Math.tanh(aValues[i])));
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, expected);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.tanh({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'tanh' must be a Tensor/);
  });
});

describeWithFlags('leakyRelu', ALL_ENVS, () => {
  it('basic', () => {
    const a = tf.tensor1d([0, 1, -2]);
    const result = tf.leakyRelu(a);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [0, 1, -0.4]);
  });

  it('propagates NaN', () => {
    const a = tf.tensor1d([0, 1, NaN]);
    const result = tf.leakyRelu(a);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [0, 1, NaN]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(-4);
    const dy = tf.scalar(8);
    const alpha = 0.1;

    const gradients = tf.grad((a) => tf.leakyRelu(a, alpha))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [8 * alpha]);
  });

  it('gradients: Tensor1D', () => {
    const aValues = [1, -1, 0.1];
    const dyValues = [1, 2, 3];
    const alpha = 0.1;

    const a = tf.tensor1d(aValues);
    const dy = tf.tensor1d(dyValues);

    const gradients = tf.grad((a) => tf.leakyRelu(a, alpha))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');

    expectArraysClose(gradients, [1, 2 * alpha, 3]);
  });

  it('gradients: Tensor2D', () => {
    const aValues = [1, -1, 0.1, 0.5];
    const dyValues = [1, 2, 3, 4];
    const alpha = 0.1;

    const a = tf.tensor2d(aValues, [2, 2]);
    const dy = tf.tensor2d(dyValues, [2, 2]);

    const gradients = tf.grad((a) => tf.leakyRelu(a, alpha))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');

    expectArraysClose(gradients, [1, 2 * alpha, 3, 4]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.leakyRelu({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'leakyRelu' must be a Tensor/);
  });
});

describeWithFlags('elu', ALL_ENVS, () => {
  it('calculate elu', () => {
    const a = tf.tensor1d([1, -1, 0]);
    const result = tf.elu(a);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [1, -0.6321, 0]);
  });

  it('elu propagates NaN', () => {
    const a = tf.tensor1d([1, NaN]);
    const result = tf.elu(a);
    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [1, NaN]);
  });

  it('derivative', () => {
    const x = tf.tensor1d([1, 3, -2]);
    const dy = tf.tensor1d([5, 50, 500]);
    const gradients = tf.grad(a => tf.elu(a))(x, dy);

    expect(gradients.shape).toEqual(x.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [5, 50, 500 * Math.exp(-2)]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.elu({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'elu' must be a Tensor/);
  });
});

describeWithFlags('selu', ALL_ENVS, () => {
  const scaleAlpha = selu_util.SELU_SCALEALPHA;
  const scale = selu_util.SELU_SCALE;

  it('calculate selu', () => {
    const a = tf.tensor1d([1, -1, 0]);
    const result = tf.selu(a);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [1.0507, -1.1113, 0]);
  });

  it('selu propagates NaN', () => {
    const a = tf.tensor1d([1, NaN]);
    const result = tf.selu(a);
    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [1.0507, NaN]);
  });

  it('gradients: Scalar', () => {
    let aValue = 1;
    let dyValue = 1;
    let a = tf.scalar(aValue);
    let dy = tf.scalar(dyValue);

    let gradients = tf.grad(a => tf.selu(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [dyValue * scale]);

    aValue = -1;
    dyValue = 2;
    a = tf.scalar(aValue);
    dy = tf.scalar(dyValue);

    gradients = tf.grad(a => tf.selu(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [dyValue * scaleAlpha * Math.exp(aValue)]);
  });

  it('gradients: Tensor1D', () => {
    const aValues = [1, -1, 0];
    const dyValues = [1, 2, 3];
    const a = tf.tensor1d(aValues);
    const dy = tf.tensor1d(dyValues);

    const gradients = tf.grad(a => tf.selu(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      if (aValues[i] > 0) {
        expected[i] = dyValues[i] * scale;
      } else {
        expected[i] = dyValues[i] * scaleAlpha * Math.exp(aValues[i]);
      }
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, expected);
  });

  it('gradients: Tensor2D', () => {
    const aValues = [1, -1, 0, 0.5];
    const dyValues = [1, 2, 3, 4];
    const a = tf.tensor2d(aValues, [2, 2]);
    const dy = tf.tensor2d(dyValues, [2, 2]);

    const gradients = tf.grad(a => tf.selu(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      if (aValues[i] > 0) {
        expected[i] = dyValues[i] * scale;
      } else {
        expected[i] = dyValues[i] * scaleAlpha * Math.exp(aValues[i]);
      }
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, expected);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.selu({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'selu' must be a Tensor/);
  });
});

describeWithFlags('clip', ALL_ENVS, () => {
  it('basic', () => {
    const a = tf.tensor1d([3, -1, 0, 100, -7, 2]);
    const min = -1;
    const max = 50;

    const result = tf.clipByValue(a, min, max);

    expectArraysClose(result, [3, -1, 0, 50, -1, 2]);
  });

  it('propagates NaNs', () => {
    const a = tf.tensor1d([3, -1, 0, 100, -7, 2, NaN]);
    const min = -1;
    const max = 50;

    const result = tf.clipByValue(a, min, max);

    expectArraysClose(result, [3, -1, 0, 50, -1, 2, NaN]);
  });

  it('min greater than max', () => {
    const a = tf.tensor1d([3, -1, 0, 100, -7, 2]);
    const min = 1;
    const max = -1;

    const f = () => {
      tf.clipByValue(a, min, max);
    };
    expect(f).toThrowError();
  });

  it('derivative: 1D tensor', () => {
    const min = -1;
    const max = 2;
    const x = tf.tensor1d([3, -2, 1]);  // Only 1 is not clipped.
    const dy = tf.tensor1d([5, 50, 500]);
    const gradients = tf.grad(x => x.clipByValue(min, max))(x, dy);

    expect(gradients.shape).toEqual(x.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [0, 0, 500]);
  });

  it('derivative: 1D tensor with max or min value', () => {
    const min = -1;
    const max = 2;
    const x = tf.tensor1d([-1, 1, 2, 3]);
    const dy = tf.tensor1d([1, 10, 100, 1000]);
    const gradients = tf.grad(x => x.clipByValue(min, max))(x, dy);

    expect(gradients.shape).toEqual(x.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [1, 10, 100, 0]);
  });

  it('derivative: scalar', () => {
    const min = -1;
    const max = 2;
    const x = tf.scalar(-10);  // Clipped.
    const dy = tf.scalar(5);
    const gradients = tf.grad(x => x.clipByValue(min, max))(x, dy);

    expect(gradients.shape).toEqual(x.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [0]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.clipByValue({} as tf.Tensor, 0, 1))
        .toThrowError(/Argument 'x' passed to 'clipByValue' must be a Tensor/);
  });
});

describeWithFlags('round', ALL_ENVS, () => {
  it('basic', () => {
    const a = tf.tensor1d([0.9, 2.5, 2.3, 1.5, -4.5]);
    const r = a.round();

    expectNumbersClose(r.get(0), 1.0);
    expectNumbersClose(r.get(1), 2.0);
    expectNumbersClose(r.get(2), 2.0);
    expectNumbersClose(r.get(3), 2.0);
    expectNumbersClose(r.get(4), -4.0);
  });

  it('propagates NaNs', () => {
    const a = tf.tensor1d([1.5, NaN, -1.4]);
    const r = tf.round(a);
    expectArraysClose(r, [2, NaN, -1]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(5.2);
    const dy = tf.scalar(3);

    const gradients = tf.grad(a => tf.round(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [0]);
  });

  it('gradients: Tensor1D', () => {
    const a = tf.tensor1d([-1.1, 2.6, 3, -5.9]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const gradients = tf.grad(a => tf.round(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [0, 0, 0, 0]);
  });

  it('gradients: Tensor2D', () => {
    const a = tf.tensor2d([-3, 1, 2.2, 3], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = tf.grad(a => tf.round(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [0, 0, 0, 0]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.round({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'round' must be a Tensor/);
  });
});

describeWithFlags('asinh', ALL_ENVS, () => {
  it('basic', () => {
    const values = [1, -3, 2, 7, -4];
    const a = tf.tensor1d(values);
    const result = tf.asinh(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.asinh(values[i]);
    }
    expectArraysClose(result, expected);
  });

  it('scalar', () => {
    const a = tf.scalar(1);
    const result = tf.asinh(a);

    const expected = [Math.asinh(1)];
    expectArraysClose(result, expected);
  });

  it('tensor2D', () => {
    const values = [1, -3, 2, 7];
    const a = tf.tensor2d(values, [2, 2]);
    const result = tf.asinh(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.asinh(values[i]);
    }
    expectArraysClose(result, expected);
  });

  it('propagates NaNs', () => {
    const a = tf.tensor1d([4, NaN, 0]);
    const res = tf.asinh(a);
    expectArraysClose(res, [Math.asinh(4), NaN, Math.asinh(0)]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(0.5);
    const dy = tf.scalar(8);

    const gradients = tf.grad(a => tf.asinh(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [8 / Math.sqrt(1.0 + 0.5 * 0.5)]);
  });

  it('gradients: Tensor1D', () => {
    const aValues = [-1, 2, 3, -5];
    const dyValues = [1, 2, 3, 4];
    const a = tf.tensor1d(aValues);
    const dy = tf.tensor1d(dyValues);

    const gradients = tf.grad(a => tf.asinh(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = dyValues[i] / Math.sqrt(1 + aValues[i] * aValues[i]);
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, expected);
  });

  it('gradients: Tensor2D', () => {
    const aValues = [-3, 1, 2, 3];
    const dyValues = [1, 2, 3, 4];
    const a = tf.tensor2d(aValues, [2, 2]);
    const dy = tf.tensor2d(dyValues, [2, 2]);

    const gradients = tf.grad(a => tf.asinh(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = dyValues[i] / Math.sqrt(1 + aValues[i] * aValues[i]);
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, expected);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.asinh({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'asinh' must be a Tensor/);
  });
});

describeWithFlags('acosh', ALL_ENVS, () => {
  it('basic', () => {
    const values = [2, 3, 4, 5, 6];
    const a = tf.tensor1d(values);
    const result = tf.acosh(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.acosh(values[i]);
    }
    expectArraysClose(result, expected);
  });

  it('scalar', () => {
    const value = 2;
    const a = tf.scalar(value);
    const result = tf.acosh(a);

    const expected = [Math.acosh(value)];
    expectArraysClose(result, expected);
  });

  it('tensor2d', () => {
    const values = [2, 3, 4, 5];
    const a = tf.tensor2d(values, [2, 2]);
    const result = tf.acosh(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.acosh(values[i]);
    }
    expectArraysClose(result, expected);
  });

  it('propagates NaNs', () => {
    const a = tf.tensor1d([4, NaN, 2]);
    const res = tf.acosh(a);
    expectArraysClose(res, [Math.acosh(4), NaN, Math.acosh(2)]);
  });

  it('NaN outside function domain', () => {
    const a = tf.tensor1d([4, -1, 2]);
    const res = tf.acosh(a);
    expectArraysClose(res, [Math.acosh(4), NaN, Math.acosh(2)]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(1.5);
    const dy = tf.scalar(8);

    const gradients = tf.grad(a => tf.acosh(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [8.0 / Math.sqrt(1.5 * 1.5 - 1.0)]);
  });

  it('gradients: Tensor1D', () => {
    const aValues = [2, 3, 5, 10];
    const dyValues = [1, 2, 3, 4];
    const a = tf.tensor1d(aValues);
    const dy = tf.tensor1d(dyValues);

    const gradients = tf.grad(a => tf.acosh(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = dyValues[i] / Math.sqrt(Math.pow(aValues[i], 2) - 1.0);
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, expected);
  });

  it('gradients: Tensor2D', () => {
    const aValues = [2, 3, 5, 7];
    const dyValues = [1, 2, 3, 4];
    const a = tf.tensor2d(aValues, [2, 2]);
    const dy = tf.tensor2d(dyValues, [2, 2]);

    const gradients = tf.grad(a => tf.acosh(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = dyValues[i] / Math.sqrt(Math.pow(aValues[i], 2) - 1.0);
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, expected);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.acosh({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'acosh' must be a Tensor/);
  });
});

describeWithFlags('atanh', ALL_ENVS, () => {
  it('basic', () => {
    const values = [-0.25, 0.25, 0.5, .75, -0.4];
    const a = tf.tensor1d(values);
    const result = tf.atanh(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.atanh(values[i]);
    }
    expectArraysClose(result, expected);
  });

  it('scalar', () => {
    const value = 0.2;
    const a = tf.scalar(value);
    const result = tf.atanh(a);

    const expected = [Math.atanh(value)];
    expectArraysClose(result, expected);
  });

  it('tensor2d', () => {
    const values = [0.2, 0.3, 0.4, 0.5];
    const a = tf.tensor2d(values, [2, 2]);
    const result = tf.atanh(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.atanh(values[i]);
    }
    expectArraysClose(result, expected);
  });

  it('propagates NaNs', () => {
    const a = tf.tensor1d([0.5, NaN, 0]);
    const res = tf.atanh(a);
    expectArraysClose(res, [Math.atanh(0.5), NaN, Math.atanh(0)]);
  });

  it('NaN outside function domain', () => {
    const a = tf.tensor1d([-2, 0, 2]);
    const res = tf.atanh(a);
    expectArraysClose(res, [NaN, Math.atanh(0), NaN]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(0.5);
    const dy = tf.scalar(8);

    const gradients = tf.grad(a => tf.atanh(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [8 / (1 - 0.5 * 0.5)]);
  });

  it('gradients: Tensor1D', () => {
    const aValues = [-0.1, 0.2, 0.3, -0.5];
    const dyValues = [1, 2, 3, 4];
    const a = tf.tensor1d(aValues);
    const dy = tf.tensor1d(dyValues);

    const gradients = tf.grad(a => tf.atanh(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = dyValues[i] / (1 - Math.pow(aValues[i], 2));
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, expected);
  });

  it('gradients: Tensor2D', () => {
    const aValues = [-0.3, 0.1, 0.2, 0.3];
    const dyValues = [1, 2, 3, 4];
    const a = tf.tensor2d(aValues, [2, 2]);
    const dy = tf.tensor2d(dyValues, [2, 2]);

    const gradients = tf.grad(a => tf.atanh(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = dyValues[i] / (1 - Math.pow(aValues[i], 2));
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, expected);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.atanh({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'atanh' must be a Tensor/);
  });
});

describeWithFlags('erf', ALL_ENVS, () => {
  it('basic', () => {
    const values = [-0.25, 0.25, 0.5, .75, -0.4];
    const a = tf.tensor1d(values);
    const result = tf.erf(a);
    const expected = [-0.2763264, 0.2763264, 0.5204999, 0.7111556, -0.4283924];
    expectArraysClose(result, expected);
  });

  it('scalar', () => {
    const a = tf.scalar(1);
    const result = tf.erf(a);
    const expected = [0.8427008];
    expectArraysClose(result, expected);
  });

  it('scalar in int32', () => {
    const a = tf.scalar(1, 'int32');
    const result = tf.erf(a);
    const expected = [0.8427008];
    expectArraysClose(result, expected);
  });

  it('tensor2d', () => {
    const values = [0.2, 0.3, 0.4, 0.5];
    const a = tf.tensor2d(values, [2, 2]);
    const result = tf.erf(a);
    const expected = [0.2227026, 0.32862678, 0.42839235, 0.5204999];
    expectArraysClose(result, expected);
  });

  it('propagates NaNs', () => {
    const a = tf.tensor1d([0.5, NaN, 0]);
    const res = tf.erf(a);
    expectArraysClose(res, [0.5204999, NaN, 0.0]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(0.5);
    const dy = tf.scalar(8);
    const gradients = tf.grad(a => tf.erf(a))(a, dy);
    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        gradients, [8 * 2 * Math.exp(-0.5 * 0.5) / Math.sqrt(Math.PI)]);
  });

  it('gradients: Tensor1D', () => {
    const aValues = [-0.1, 0.2, 0.3, -0.5];
    const dyValues = [1, 2, 3, 4];
    const a = tf.tensor1d(aValues);
    const dy = tf.tensor1d(dyValues);
    const gradients = tf.grad(a => tf.erf(a))(a, dy);
    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = dyValues[i] * 2 * Math.exp(-aValues[i] * aValues[i]) /
          Math.sqrt(Math.PI);
    }
    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, expected);
  });

  it('gradients: Tensor2D', () => {
    const aValues = [-0.3, 0.1, 0.2, 0.3];
    const dyValues = [1, 2, 3, 4];
    const a = tf.tensor2d(aValues, [2, 2]);
    const dy = tf.tensor2d(dyValues, [2, 2]);
    const gradients = tf.grad(a => tf.erf(a))(a, dy);
    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = dyValues[i] * 2 * Math.exp(-aValues[i] * aValues[i]) /
          Math.sqrt(Math.PI);
    }
    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, expected);
  });
});
