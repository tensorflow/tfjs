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
// tslint:disable-next-line:max-line-length
import {ALL_ENVS, describeWithFlags, expectArraysClose, expectNumbersClose} from '../test_util';
import * as util from '../util';

import * as selu_util from './selu_util';

describeWithFlags('relu', ALL_ENVS, () => {
  it('basic', () => {
    const a = dl.tensor1d([1, -2, 0, 3, -0.1]);
    const result = dl.relu(a);
    expectArraysClose(result, [1, 0, 0, 3, 0]);
  });

  it('does nothing to positive values', () => {
    const a = dl.scalar(1);
    const result = dl.relu(a);
    expectNumbersClose(result.get(), 1);
  });

  it('sets negative values to 0', () => {
    const a = dl.scalar(-1);
    const result = dl.relu(a);
    expectNumbersClose(result.get(), 0);
  });

  it('preserves zero values', () => {
    const a = dl.scalar(0);
    const result = dl.relu(a);
    expectNumbersClose(result.get(), 0);
  });

  it('propagates NaNs, float32', () => {
    const a = dl.tensor1d([1, -2, 0, 3, -0.1, NaN]);
    const result = dl.relu(a);
    expect(result.dtype).toBe('float32');
    expectArraysClose(result, [1, 0, 0, 3, 0, NaN]);
  });

  it('propagates NaNs, int32', () => {
    const a = dl.tensor1d([1, -2, 0, 3, -1, util.NAN_INT32], 'int32');
    const result = dl.relu(a);
    expect(result.dtype).toBe('int32');
    expectArraysClose(result, [1, 0, 0, 3, 0, util.NAN_INT32]);
  });

  it('propagates NaNs, bool', () => {
    const a = dl.tensor1d([1, 0, 0, 1, 0, util.NAN_BOOL], 'bool');
    const result = dl.relu(a);
    expect(result.dtype).toBe('bool');
    expectArraysClose(result, [1, 0, 0, 1, 0, util.NAN_BOOL]);
  });

  it('gradients: positive scalar', () => {
    const a = dl.scalar(3);
    const dy = dl.scalar(5);

    const grad = dl.grad(a => dl.relu(a));
    const da = grad(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [5]);
  });

  it('gradients: negative scalar', () => {
    const a = dl.scalar(-3);
    const dy = dl.scalar(5);

    const grad = dl.grad(a => dl.relu(a));
    const da = grad(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [0]);
  });

  it('gradients: array', () => {
    const a = dl.tensor2d([1, -1, 0, .1], [2, 2]);
    const dy = dl.tensor2d([1, 2, 3, 4], [2, 2]);

    const grad = dl.grad(a => dl.relu(a));
    const da = grad(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [1, 0, 0, 4]);
  });
});

describeWithFlags('abs', ALL_ENVS, () => {
  it('basic', () => {
    const a = dl.tensor1d([1, -2, 0, 3, -0.1]);
    const result = dl.abs(a);
    expectArraysClose(result, [1, 2, 0, 3, 0.1]);
  });

  it('propagates NaNs', () => {
    const a = dl.tensor1d([1, -2, 0, 3, -0.1, NaN]);
    const result = dl.abs(a);
    expectArraysClose(result, [1, 2, 0, 3, 0.1, NaN]);
  });

  it('gradients: Scalar', () => {
    const a = dl.scalar(4);
    const dy = dl.scalar(8);

    const da = dl.grad(a => dl.abs(a))(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [8 * 1]);
  });

  it('gradients: Tensor1D', () => {
    const a = dl.tensor1d([1, 2, -3, 5]);
    const dy = dl.tensor1d([1, 2, 3, 4]);

    const da = dl.grad(a => dl.abs(a))(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [1 * 1, 2 * 1, 3 * -1, 4 * 1]);
  });

  it('gradients: Tensor2D', () => {
    const a = dl.tensor2d([3, -1, -2, 3], [2, 2]);
    const dy = dl.tensor2d([1, 2, 3, 4], [2, 2]);

    const da = dl.grad(a => dl.abs(a))(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [1 * 1, 2 * -1, 3 * -1, 4 * 1]);
  });
});

describeWithFlags('step', ALL_ENVS, () => {
  it('with 1d tensor', () => {
    const a = dl.tensor1d([1, -2, -.01, 3, -0.1]);
    const result = dl.step(a);
    expectArraysClose(result, [1, 0, 0, 1, 0]);
  });

  it('with 1d tensor and alpha', () => {
    const a = dl.tensor1d([1, -2, -.01, 3, NaN]);
    const result = dl.step(a, 0.1);
    expectArraysClose(result, [1, 0.1, 0.1, 1, NaN]);
  });

  it('with 2d tensor', () => {
    const a = dl.tensor2d([1, -5, -3, 4], [2, 2]);
    const result = dl.step(a);
    expect(result.shape).toEqual([2, 2]);
    expectArraysClose(result, [1, 0, 0, 1]);
  });

  it('propagates NaNs', () => {
    const a = dl.tensor1d([1, -2, -.01, 3, NaN]);
    const result = dl.step(a);
    expectArraysClose(result, [1, 0, 0, 1, NaN]);
  });

  it('gradients: Scalar', () => {
    const a = dl.scalar(-4);
    const dy = dl.scalar(8);

    const gradients = dl.grad(a => dl.step(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [0]);
  });

  it('gradients: Tensor1D', () => {
    const a = dl.tensor1d([1, 2, -3, 5]);
    const dy = dl.tensor1d([1, 2, 3, 4]);

    const gradients = dl.grad(a => dl.step(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [0, 0, 0, 0]);
  });

  it('gradients: Tensor2D', () => {
    const a = dl.tensor2d([3, -1, -2, 3], [2, 2]);
    const dy = dl.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = dl.grad(a => dl.step(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [0, 0, 0, 0]);
  });
});

describeWithFlags('neg', ALL_ENVS, () => {
  it('basic', () => {
    const a = dl.tensor1d([1, -3, 2, 7, -4]);
    const result = dl.neg(a);
    expectArraysClose(result, [-1, 3, -2, -7, 4]);
  });

  it('propagate NaNs', () => {
    const a = dl.tensor1d([1, -3, 2, 7, NaN]);
    const result = dl.neg(a);
    const expected = [-1, 3, -2, -7, NaN];
    expectArraysClose(result, expected);
  });

  it('gradients: Scalar', () => {
    const a = dl.scalar(4);
    const dy = dl.scalar(8);

    const da = dl.grad(a => dl.neg(a))(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [8 * -1]);
  });

  it('gradients: Tensor1D', () => {
    const a = dl.tensor1d([1, 2, -3, 5]);
    const dy = dl.tensor1d([1, 2, 3, 4]);

    const da = dl.grad(a => dl.neg(a))(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [1 * -1, 2 * -1, 3 * -1, 4 * -1]);
  });

  it('gradients: Tensor2D', () => {
    const a = dl.tensor2d([3, -1, -2, 3], [2, 2]);
    const dy = dl.tensor2d([1, 2, 3, 4], [2, 2]);

    const da = dl.grad(a => dl.neg(a))(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [1 * -1, 2 * -1, 3 * -1, 4 * -1]);
  });
});

describeWithFlags('sigmoid', ALL_ENVS, () => {
  it('basic', () => {
    const values = [1, -3, 2, 7, -4];
    const a = dl.tensor1d(values);

    const result = dl.sigmoid(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = 1 / (1 + Math.exp(-values[i]));
    }
    expectArraysClose(result, expected);
  });

  it('propagates NaNs', () => {
    const a = dl.tensor1d([3, NaN]);
    const res = dl.sigmoid(a);
    expectArraysClose(res, [1 / (1 + Math.exp(-3)), NaN]);
  });

  it('gradients: Tensor1D', () => {
    const a = dl.tensor1d([1, 2, -3, 5]);
    const dy = dl.tensor1d([1, 2, 3, 4]);

    const da = dl.grad(a => dl.sigmoid(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      const y = 1 / (1 + Math.exp(-a.get(i)));
      expected[i] = dy.get(i) * y * (1 - y);
    }

    expectArraysClose(da, expected);
  });
});

describeWithFlags('sqrt', ALL_ENVS, () => {
  it('sqrt', () => {
    const a = dl.tensor1d([2, 4]);
    const r = dl.sqrt(a);
    expectNumbersClose(r.get(0), Math.sqrt(2));
    expectNumbersClose(r.get(1), Math.sqrt(4));
  });

  it('sqrt propagates NaNs', () => {
    const a = dl.tensor1d([1, NaN]);
    const r = dl.sqrt(a);
    expectArraysClose(r, [Math.sqrt(1), NaN]);
  });

  it('gradients: Scalar', () => {
    const a = dl.scalar(4);
    const dy = dl.scalar(8);

    const da = dl.grad(a => dl.sqrt(a))(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [8 / (2 * Math.sqrt(4))]);
  });

  it('gradients: Tensor1D', () => {
    const a = dl.tensor1d([1, 2, 3, 5]);
    const dy = dl.tensor1d([1, 2, 3, 4]);

    const gradients = dl.grad(a => dl.sqrt(a))(a, dy);

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
    const a = dl.tensor2d([3, 1, 2, 3], [2, 2]);
    const dy = dl.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = dl.grad(a => dl.sqrt(a))(a, dy);

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
});

describeWithFlags('rsqrt', ALL_ENVS, () => {
  it('rsqrt', () => {
    const a = dl.tensor1d([2, 4]);
    const r = dl.rsqrt(a);
    expectNumbersClose(r.get(0), 1 / Math.sqrt(2));
    expectNumbersClose(r.get(1), 1 / Math.sqrt(4));
  });

  it('rsqrt propagates NaNs', () => {
    const a = dl.tensor1d([1, NaN]);
    const r = dl.rsqrt(a);
    expectArraysClose(r, [1 / Math.sqrt(1), NaN]);
  });

  it('gradients: Scalar', () => {
    const a = dl.scalar(4);
    const dy = dl.scalar(8);

    const da = dl.grad(a => dl.rsqrt(a))(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [(-1 * 8) / (2 * Math.pow(4, 1.5))]);
  });

  it('gradients: Tensor1D', () => {
    const a = dl.tensor1d([1, 2, 3, 5]);
    const dy = dl.tensor1d([1, 2, 3, 4]);

    const gradients = dl.grad(a => dl.rsqrt(a))(a, dy);

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
    const a = dl.tensor2d([3, 1, 2, 3], [2, 2]);
    const dy = dl.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = dl.grad(a => dl.rsqrt(a))(a, dy);

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
});

describeWithFlags('square', ALL_ENVS, () => {
  it('1D array', () => {
    const a = dl.tensor1d([2, 4, Math.sqrt(2)]);
    const r = dl.square(a);
    expectArraysClose(r, [4, 16, 2]);
  });

  it('2D array', () => {
    const a = dl.tensor2d([1, 2, Math.sqrt(2), Math.sqrt(3)], [2, 2]);
    const r = dl.square(a);
    expect(r.shape).toEqual([2, 2]);
    expectArraysClose(r, [1, 4, 2, 3]);
  });

  it('square propagates NaNs', () => {
    const a = dl.tensor1d([1.5, NaN]);
    const r = dl.square(a);
    expectArraysClose(r, [2.25, NaN]);
  });

  it('gradients: Scalar', () => {
    const a = dl.scalar(5);
    const dy = dl.scalar(8);

    const gradients = dl.grad(a => dl.square(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [2 * 5 * 8]);
  });

  it('gradients: Tensor1D', () => {
    const a = dl.tensor1d([-1, 2, 3, -5]);
    const dy = dl.tensor1d([1, 2, 3, 4]);

    const gradients = dl.grad(a => dl.square(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [-2, 4 * 2, 6 * 3, -10 * 4]);
  });

  it('gradients: Tensor2D', () => {
    const a = dl.tensor2d([-3, 1, 2, 3], [2, 2]);
    const dy = dl.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = dl.grad(a => dl.square(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [-6 * 1, 2 * 2, 4 * 3, 6 * 4]);
  });
});

describeWithFlags('reciprocal', ALL_ENVS, () => {
  it('1D array', () => {
    const a = dl.tensor1d([2, 3, 0, NaN]);
    const r = dl.reciprocal(a);
    expectArraysClose(r, [1 / 2, 1 / 3, Infinity, NaN]);
  });

  it('2D array', () => {
    const a = dl.tensor2d([1, Infinity, 0, NaN], [2, 2]);
    const r = dl.reciprocal(a);
    expect(r.shape).toEqual([2, 2]);
    expectArraysClose(r, [1 / 1, 0, Infinity, NaN]);
  });

  it('reciprocal propagates NaNs', () => {
    const a = dl.tensor1d([1.5, NaN]);
    const r = dl.reciprocal(a);
    expectArraysClose(r, [1 / 1.5, NaN]);
  });

  it('gradients: Scalar', () => {
    const a = dl.scalar(5);
    const dy = dl.scalar(8);

    const gradients = dl.grad(a => dl.reciprocal(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [-1 * 8 * (1 / (5 * 5))]);
  });

  it('gradients: Tensor1D', () => {
    const a = dl.tensor1d([-1, 2, 3, -5]);
    const dy = dl.tensor1d([1, 2, 3, 4]);

    const gradients = dl.grad(a => dl.reciprocal(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [
      -1 * 1 * (1 / (-1 * -1)), -1 * 2 * (1 / (2 * 2)), -1 * 3 * (1 / (3 * 3)),
      -1 * 4 * (1 / (-5 * -5))
    ]);
  });

  it('gradients: Tensor2D', () => {
    const a = dl.tensor2d([-1, 2, 3, -5], [2, 2]);
    const dy = dl.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = dl.grad(a => dl.reciprocal(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [
      -1 * 1 * (1 / (-1 * -1)), -1 * 2 * (1 / (2 * 2)), -1 * 3 * (1 / (3 * 3)),
      -1 * 4 * (1 / (-5 * -5))
    ]);
  });
});

describeWithFlags('log', ALL_ENVS, () => {
  it('log', () => {
    const a = dl.tensor1d([1, 2]);
    const r = dl.log(a);
    expectNumbersClose(r.get(0), Math.log(1));
    expectNumbersClose(r.get(1), Math.log(2));
  });

  it('log propagates NaNs', () => {
    const a = dl.tensor1d([1, NaN]);
    const r = dl.log(a);
    expectArraysClose(r, [Math.log(1), NaN]);
  });

  it('gradients: Scalar', () => {
    const a = dl.scalar(5);
    const dy = dl.scalar(3);

    const gradients = dl.grad(a => dl.log(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [3 / 5]);
  });

  it('gradients: Tensor1D', () => {
    const a = dl.tensor1d([-1, 2, 3, -5]);
    const dy = dl.tensor1d([1, 2, 3, 4]);

    const gradients = dl.grad(a => dl.log(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [1 / -1, 2 / 2, 3 / 3, 4 / -5]);
  });

  it('gradients: Tensor2D', () => {
    const a = dl.tensor2d([-3, 1, 2, 3], [2, 2]);
    const dy = dl.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = dl.grad(a => dl.log(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [1 / -3, 2 / 1, 3 / 2, 4 / 3]);
  });
});

describeWithFlags('log1p', ALL_ENVS, () => {
  it('log1p', () => {
    const a = dl.tensor1d([1, 2]);
    const r = dl.log1p(a);
    expectNumbersClose(r.get(0), Math.log1p(1));
    expectNumbersClose(r.get(1), Math.log1p(2));
  });

  it('log1p propagates NaNs', () => {
    const a = dl.tensor1d([1, NaN]);
    const r = dl.log1p(a);
    expectArraysClose(r, [Math.log1p(1), NaN]);
  });

  it('gradients: Scalar', () => {
    const a = dl.scalar(5);
    const dy = dl.scalar(3);

    const gradients = dl.grad(a => dl.log1p(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [3 / (1 + 5)]);
  });

  it('gradients: Tensor1D', () => {
    const a = dl.tensor1d([-1, 2, 3, -5]);
    const dy = dl.tensor1d([1, 2, 3, 4]);

    const gradients = dl.grad(a => dl.log1p(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        gradients, [Infinity, 2 / (1 + 2), 3 / (1 + 3), 4 / (1 + -5)]);
  });

  it('gradients: Tensor2D', () => {
    const a = dl.tensor2d([-3, 1, 2, 3], [2, 2]);
    const dy = dl.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = dl.grad(a => dl.log1p(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        gradients, [1 / (1 + -3), 2 / (1 + 1), 3 / (1 + 2), 4 / (1 + 3)]);
  });
});

describeWithFlags('ceil', ALL_ENVS, () => {
  it('basic', () => {
    const a = dl.tensor1d([1.5, 2.1, -1.4]);
    const r = dl.ceil(a);
    expectNumbersClose(r.get(0), 2);
    expectNumbersClose(r.get(1), 3);
    expectNumbersClose(r.get(2), -1);
  });

  it('propagates NaNs', () => {
    const a = dl.tensor1d([1.5, NaN, -1.4]);
    const r = dl.ceil(a);
    expectArraysClose(r, [2, NaN, -1]);
  });

  it('gradients: Scalar', () => {
    const a = dl.scalar(5.2);
    const dy = dl.scalar(3);

    const gradients = dl.grad(a => dl.ceil(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [0]);
  });

  it('gradients: Tensor1D', () => {
    const a = dl.tensor1d([-1.1, 2.6, 3, -5.9]);
    const dy = dl.tensor1d([1, 2, 3, 4]);

    const gradients = dl.grad(a => dl.ceil(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [0, 0, 0, 0]);
  });

  it('gradients: Tensor2D', () => {
    const a = dl.tensor2d([-3, 1, 2.2, 3], [2, 2]);
    const dy = dl.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = dl.grad(a => dl.ceil(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [0, 0, 0, 0]);
  });
});

describeWithFlags('floor', ALL_ENVS, () => {
  it('basic', () => {
    const a = dl.tensor1d([1.5, 2.1, -1.4]);
    const r = dl.floor(a);

    expectNumbersClose(r.get(0), 1);
    expectNumbersClose(r.get(1), 2);
    expectNumbersClose(r.get(2), -2);
  });

  it('propagates NaNs', () => {
    const a = dl.tensor1d([1.5, NaN, -1.4]);
    const r = dl.floor(a);
    expectArraysClose(r, [1, NaN, -2]);
  });

  it('gradients: Scalar', () => {
    const a = dl.scalar(5.2);
    const dy = dl.scalar(3);

    const gradients = dl.grad(a => dl.floor(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [0]);
  });

  it('gradients: Tensor1D', () => {
    const a = dl.tensor1d([-1.1, 2.6, 3, -5.9]);
    const dy = dl.tensor1d([1, 2, 3, 4]);

    const gradients = dl.grad(a => dl.floor(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [0, 0, 0, 0]);
  });

  it('gradients: Tensor2D', () => {
    const a = dl.tensor2d([-3, 1, 2.2, 3], [2, 2]);
    const dy = dl.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = dl.grad(a => dl.floor(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [0, 0, 0, 0]);
  });
});

describeWithFlags('sign', ALL_ENVS, () => {
  it('basic', () => {
    const a = dl.tensor1d([1.5, 0, NaN, -1.4]);
    const r = dl.sign(a);
    expectNumbersClose(r.get(0), 1);
    expectNumbersClose(r.get(1), 0);
    expectNumbersClose(r.get(2), 0);
    expectNumbersClose(r.get(3), -1);
  });

  it('propagates NaNs', () => {
    const a = dl.tensor1d([1.5, NaN, -1.4]);
    const r = dl.sign(a);
    expectArraysClose(r, [1, 0, -1]);
  });

  it('gradients: Scalar', () => {
    const a = dl.scalar(5.2);
    const dy = dl.scalar(3);

    const gradients = dl.grad(a => dl.sign(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [0]);
  });

  it('gradients: Tensor1D', () => {
    const a = dl.tensor1d([-1.1, 2.6, 3, -5.9]);
    const dy = dl.tensor1d([-1, 1, 1, -1]);

    const gradients = dl.grad(a => dl.sign(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [0, 0, 0, 0]);
  });

  it('gradients: Tensor2D', () => {
    const a = dl.tensor2d([-3, 1, 2.2, 3], [2, 2]);
    const dy = dl.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = dl.grad(a => dl.sign(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [0, 0, 0, 0]);
  });
});

describeWithFlags('exp', ALL_ENVS, () => {
  it('exp', () => {
    const a = dl.tensor1d([1, 2, 0]);
    const r = dl.exp(a);

    expectNumbersClose(r.get(0), Math.exp(1));
    expectNumbersClose(r.get(1), Math.exp(2));
    expectNumbersClose(r.get(2), 1);
  });

  it('exp propagates NaNs', () => {
    const a = dl.tensor1d([1, NaN, 0]);
    const r = dl.exp(a);
    expectArraysClose(r, [Math.exp(1), NaN, 1]);
  });

  it('gradients: Scalar', () => {
    const a = dl.scalar(0.5);
    const dy = dl.scalar(3);

    const gradients = dl.grad(a => dl.exp(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [3 * Math.exp(0.5)]);
  });

  it('gradients: Tensor1D', () => {
    const a = dl.tensor1d([-1, 2, 3, -5]);
    const dy = dl.tensor1d([1, 2, 3, 4]);

    const gradients = dl.grad(a => dl.exp(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        gradients,
        [1 * Math.exp(-1), 2 * Math.exp(2), 3 * Math.exp(3), 4 * Math.exp(-5)],
        1e-1);
  });

  it('gradients: Tensor2D', () => {
    const a = dl.tensor2d([-3, 1, 2, 3], [2, 2]);
    const dy = dl.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = dl.grad(a => dl.exp(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        gradients,
        [1 * Math.exp(-3), 2 * Math.exp(1), 3 * Math.exp(2), 4 * Math.exp(3)],
        1e-1);
  });
});

describeWithFlags('expm1', ALL_ENVS, () => {
  it('expm1', () => {
    const a = dl.tensor1d([1, 2, 0]);
    const r = dl.expm1(a);

    expectNumbersClose(r.get(0), Math.expm1(1));
    expectNumbersClose(r.get(1), Math.expm1(2));
    expectNumbersClose(r.get(2), Math.expm1(0));
  });

  it('expm1 propagates NaNs', () => {
    const a = dl.tensor1d([1, NaN, 0]);
    const r = dl.expm1(a);
    expectArraysClose(r, [Math.expm1(1), NaN, Math.expm1(0)]);
  });

  it('gradients: Scalar', () => {
    const a = dl.scalar(0.5);
    const dy = dl.scalar(3);

    const gradients = dl.grad(a => dl.expm1(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [3 * Math.exp(0.5)]);
  });

  it('gradients: Tensor1D', () => {
    const a = dl.tensor1d([-1, 2, 3, -5]);
    const dy = dl.tensor1d([1, 2, 3, 4]);

    const gradients = dl.grad(a => dl.expm1(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        gradients,
        [1 * Math.exp(-1), 2 * Math.exp(2), 3 * Math.exp(3), 4 * Math.exp(-5)],
        1e-1);
  });

  it('gradients: Tensor2D', () => {
    const a = dl.tensor2d([-3, 1, 2, 3], [2, 2]);
    const dy = dl.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = dl.grad(a => dl.expm1(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        gradients,
        [1 * Math.exp(-3), 2 * Math.exp(1), 3 * Math.exp(2), 4 * Math.exp(3)],
        1e-1);
  });
});

describeWithFlags('sin', ALL_ENVS, () => {
  it('basic', () => {
    const values = [1, -3, 2, 7, -4];
    const a = dl.tensor1d(values);
    const result = dl.sin(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.sin(values[i]);
    }
    expectArraysClose(result, expected);
  });

  it('propagates NaNs', () => {
    const a = dl.tensor1d([4, NaN, 0]);
    const res = dl.sin(a);
    expectArraysClose(res, [Math.sin(4), NaN, Math.sin(0)]);
  });

  it('gradients: Scalar', () => {
    const a = dl.scalar(5);
    const dy = dl.scalar(8);

    const gradients = dl.grad(a => dl.sin(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [8 * Math.cos(5)]);
  });

  it('gradients: Tensor1D', () => {
    const a = dl.tensor1d([-1, 2, 3, -5]);
    const dy = dl.tensor1d([1, 2, 3, 4]);

    const gradients = dl.grad(a => dl.sin(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        gradients,
        [1 * Math.cos(-1), 2 * Math.cos(2), 3 * Math.cos(3), 4 * Math.cos(-5)],
        1e-1);
  });

  it('gradients: Tensor2D', () => {
    const a = dl.tensor2d([-3, 1, 2, 3], [2, 2]);
    const dy = dl.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = dl.grad(a => dl.sin(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        gradients,
        [1 * Math.cos(-3), 2 * Math.cos(1), 3 * Math.cos(2), 4 * Math.cos(3)],
        1e-1);
  });
});

describeWithFlags('cos', ALL_ENVS, () => {
  it('basic', () => {
    const values = [1, -3, 2, 7, -4];
    const a = dl.tensor1d(values);
    const result = dl.cos(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.cos(values[i]);
    }
    expectArraysClose(result, expected);
  });

  it('propagates NaNs', () => {
    const a = dl.tensor1d([4, NaN, 0]);
    const res = dl.cos(a);
    expectArraysClose(res, [Math.cos(4), NaN, Math.cos(0)]);
  });

  it('gradients: Scalar', () => {
    const a = dl.scalar(5);
    const dy = dl.scalar(8);

    const gradients = dl.grad(a => dl.cos(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [8 * Math.sin(5) * -1]);
  });

  it('gradients: Tensor1D', () => {
    const a = dl.tensor1d([-1, 2, 3, -5]);
    const dy = dl.tensor1d([1, 2, 3, 4]);

    const gradients = dl.grad(a => dl.cos(a))(a, dy);

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
    const a = dl.tensor2d([-3, 1, 2, 3], [2, 2]);
    const dy = dl.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = dl.grad(a => dl.cos(a))(a, dy);

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
});

describeWithFlags('tan', ALL_ENVS, () => {
  it('basic', () => {
    const values = [1, -3, 2, 7, -4];
    const a = dl.tensor1d(values);
    const result = dl.tan(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.tan(values[i]);
    }
    expectArraysClose(result, expected);
  });

  it('propagates NaNs', () => {
    const a = dl.tensor1d([4, NaN, 0]);
    const res = dl.tan(a);
    expectArraysClose(res, [Math.tan(4), NaN, Math.tan(0)]);
  });

  it('gradients: Scalar', () => {
    const a = dl.scalar(0.5);
    const dy = dl.scalar(8);

    const gradients = dl.grad(a => dl.tan(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [8 / (Math.cos(0.5) * Math.cos(0.5))]);
  });

  it('gradients: Tensor1D', () => {
    const aValues = [-1, 2, 3, -5];
    const dyValues = [1, 2, 3, 4];
    const a = dl.tensor1d(aValues);
    const dy = dl.tensor1d(dyValues);

    const gradients = dl.grad(a => dl.tan(a))(a, dy);

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
    const a = dl.tensor2d(aValues, [2, 2]);
    const dy = dl.tensor2d(dyValues, [2, 2]);

    const gradients = dl.grad(a => dl.tan(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = dyValues[i] / (Math.cos(aValues[i]) * Math.cos(aValues[i]));
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, expected);
  });
});

describeWithFlags('asin', ALL_ENVS, () => {
  it('basic', () => {
    const values = [.1, -3, 2, 7, -4];
    const a = dl.tensor1d(values);
    const result = dl.asin(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.asin(values[i]);
    }
    expectArraysClose(result, expected);
  });

  it('propagates NaNs', () => {
    const a = dl.tensor1d([4, NaN, 0]);
    const res = dl.asin(a);
    expectArraysClose(res, [Math.asin(4), NaN, Math.asin(0)]);
  });

  it('gradients: Scalar', () => {
    const a = dl.scalar(0.5);
    const dy = dl.scalar(8);

    const gradients = dl.grad(a => dl.asin(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [8 / Math.sqrt(1 - (0.5 * 0.5))]);
  });

  it('gradients: Tensor1D', () => {
    const aValues = [-0.1, 0.2, 0.3, -0.5];
    const dyValues = [1, 2, 3, 4];
    const a = dl.tensor1d(aValues);
    const dy = dl.tensor1d(dyValues);

    const gradients = dl.grad(a => dl.asin(a))(a, dy);

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
    const a = dl.tensor2d(aValues, [2, 2]);
    const dy = dl.tensor2d(dyValues, [2, 2]);

    const gradients = dl.grad(a => dl.asin(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = dyValues[i] / Math.sqrt(1 - (aValues[i] * aValues[i]));
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, expected);
  });
});

describeWithFlags('acos', ALL_ENVS, () => {
  it('basic', () => {
    const values = [.1, -3, 2, 7, -4];
    const a = dl.tensor1d(values);
    const result = dl.acos(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.acos(values[i]);
    }
    expectArraysClose(result, expected);
  });

  it('propagates NaNs', () => {
    const a = dl.tensor1d([4, NaN, 0]);
    const res = dl.acos(a);
    expectArraysClose(res, [Math.acos(4), NaN, Math.acos(0)]);
  });

  it('gradients: Scalar', () => {
    const a = dl.scalar(0.5);
    const dy = dl.scalar(8);

    const gradients = dl.grad(a => dl.acos(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [(-1 * 8) / Math.sqrt(1 - (0.5 * 0.5))]);
  });

  it('gradients: Tensor1D', () => {
    const aValues = [-0.1, 0.2, 0.3, -0.5];
    const dyValues = [1, 2, 3, 4];
    const a = dl.tensor1d(aValues);
    const dy = dl.tensor1d(dyValues);

    const gradients = dl.grad(a => dl.acos(a))(a, dy);

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
    const a = dl.tensor2d(aValues, [2, 2]);
    const dy = dl.tensor2d(dyValues, [2, 2]);

    const gradients = dl.grad(a => dl.acos(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] =
          (-1 * dyValues[i]) / Math.sqrt(1 - (aValues[i] * aValues[i]));
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, expected);
  });
});

describeWithFlags('atan', ALL_ENVS, () => {
  it('basic', () => {
    const values = [1, -3, 2, 7, -4];
    const a = dl.tensor1d(values);
    const result = dl.atan(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.atan(values[i]);
    }
    expectArraysClose(result, expected);
  });

  it('propagates NaNs', () => {
    const a = dl.tensor1d([4, NaN, 0]);
    const res = dl.atan(a);
    expectArraysClose(res, [Math.atan(4), NaN, Math.atan(0)]);
  });

  it('gradients: Scalar', () => {
    const a = dl.scalar(0.5);
    const dy = dl.scalar(8);

    const gradients = dl.grad(a => dl.atan(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [8 / (1 + (0.5 * 0.5))]);
  });

  it('gradients: Tensor1D', () => {
    const aValues = [-0.1, 0.2, 0.3, -0.5];
    const dyValues = [1, 2, 3, 4];
    const a = dl.tensor1d(aValues);
    const dy = dl.tensor1d(dyValues);

    const gradients = dl.grad(a => dl.atan(a))(a, dy);

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
    const a = dl.tensor2d(aValues, [2, 2]);
    const dy = dl.tensor2d(dyValues, [2, 2]);

    const gradients = dl.grad(a => dl.atan(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = dyValues[i] / (1 + (aValues[i] * aValues[i]));
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, expected);
  });
});

describeWithFlags('sinh', ALL_ENVS, () => {
  it('basic', () => {
    const values = [1, -3, 2, 7, -4];
    const a = dl.tensor1d(values);
    const result = dl.sinh(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.sinh(values[i]);
    }
    expectArraysClose(result, expected);
  });

  it('propagates NaNs', () => {
    const a = dl.tensor1d([4, NaN, 0]);
    const res = dl.sinh(a);
    expectArraysClose(res, [Math.sinh(4), NaN, Math.sinh(0)]);
  });

  it('gradients: Scalar', () => {
    const a = dl.scalar(0.5);
    const dy = dl.scalar(8);

    const gradients = dl.grad(a => dl.sinh(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [8 * Math.cosh(0.5)]);
  });

  it('gradients: Tensor1D', () => {
    const aValues = [-1, 2, 3, -5];
    const dyValues = [1, 2, 3, 4];
    const a = dl.tensor1d(aValues);
    const dy = dl.tensor1d(dyValues);

    const gradients = dl.grad(a => dl.sinh(a))(a, dy);

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
    const a = dl.tensor2d(aValues, [2, 2]);
    const dy = dl.tensor2d(dyValues, [2, 2]);

    const gradients = dl.grad(a => dl.sinh(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = dyValues[i] * Math.cosh(aValues[i]);
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, expected);
  });
});

describeWithFlags('cosh', ALL_ENVS, () => {
  it('basic', () => {
    const values = [1, -3, 2, -1, -4];
    const a = dl.tensor1d(values);
    const result = dl.cosh(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.cosh(values[i]);
    }

    // TODO(nsthorat): Fix the precision problem here.
    expectArraysClose(result, expected);
  });

  it('propagates NaNs', () => {
    const a = dl.tensor1d([4, NaN, 0]);
    const res = dl.cosh(a);
    expectArraysClose(res, [Math.cosh(4), NaN, Math.cosh(0)]);
  });

  it('gradients: Scalar', () => {
    const a = dl.scalar(0.5);
    const dy = dl.scalar(8);

    const gradients = dl.grad(a => dl.cosh(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [8 * Math.sinh(0.5)]);
  });

  it('gradients: Tensor1D', () => {
    const aValues = [-1, 2, 3, -5];
    const dyValues = [1, 2, 3, 4];
    const a = dl.tensor1d(aValues);
    const dy = dl.tensor1d(dyValues);

    const gradients = dl.grad(a => dl.cosh(a))(a, dy);

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
    const a = dl.tensor2d(aValues, [2, 2]);
    const dy = dl.tensor2d(dyValues, [2, 2]);

    const gradients = dl.grad(a => dl.cosh(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = dyValues[i] * Math.sinh(aValues[i]);
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, expected);
  });
});

describeWithFlags('tanh', ALL_ENVS, () => {
  it('basic', () => {
    const values = [1, -3, 2, 7, -4];
    const a = dl.tensor1d(values);
    const result = dl.tanh(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = util.tanh(values[i]);
    }
    expectArraysClose(result, expected);
  });

  it('propagates NaNs', () => {
    const a = dl.tensor1d([4, NaN, 0]);
    const res = dl.tanh(a);
    expectArraysClose(res, [util.tanh(4), NaN, util.tanh(0)]);
  });

  it('gradients: Scalar', () => {
    const a = dl.scalar(0.5);
    const dy = dl.scalar(8);

    const gradients = dl.grad(a => dl.tanh(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [8 * (1 - (Math.tanh(0.5) * Math.tanh(0.5)))]);
  });

  it('gradients: Tensor1D', () => {
    const aValues = [-1, 2, 3, -5];
    const dyValues = [1, 2, 3, 4];
    const a = dl.tensor1d(aValues);
    const dy = dl.tensor1d(dyValues);

    const gradients = dl.grad(a => dl.tanh(a))(a, dy);

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
    const a = dl.tensor2d(aValues, [2, 2]);
    const dy = dl.tensor2d(dyValues, [2, 2]);

    const gradients = dl.grad(a => dl.tanh(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] =
          dyValues[i] * (1 - (Math.tanh(aValues[i]) * Math.tanh(aValues[i])));
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, expected);
  });
});

describeWithFlags('leakyRelu', ALL_ENVS, () => {
  it('basic', () => {
    const a = dl.tensor1d([0, 1, -2]);
    const result = dl.leakyRelu(a);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [0, 1, -0.4]);
  });

  it('propagates NaN', () => {
    const a = dl.tensor1d([0, 1, NaN]);
    const result = dl.leakyRelu(a);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [0, 1, NaN]);
  });

  it('gradients: Scalar', () => {
    const a = dl.scalar(-4);
    const dy = dl.scalar(8);
    const alpha = 0.1;

    const gradients = dl.grad((a) => dl.leakyRelu(a, alpha))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [8 * alpha]);
  });

  it('gradients: Tensor1D', () => {
    const aValues = [1, -1, 0.1];
    const dyValues = [1, 2, 3];
    const alpha = 0.1;

    const a = dl.tensor1d(aValues);
    const dy = dl.tensor1d(dyValues);

    const gradients = dl.grad((a) => dl.leakyRelu(a, alpha))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');

    expectArraysClose(gradients, [1, 2 * alpha, 3]);
  });

  it('gradients: Tensor2D', () => {
    const aValues = [1, -1, 0.1, 0.5];
    const dyValues = [1, 2, 3, 4];
    const alpha = 0.1;

    const a = dl.tensor2d(aValues, [2, 2]);
    const dy = dl.tensor2d(dyValues, [2, 2]);

    const gradients = dl.grad((a) => dl.leakyRelu(a, alpha))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');

    expectArraysClose(gradients, [1, 2 * alpha, 3, 4]);
  });
});

describeWithFlags('elu', ALL_ENVS, () => {
  it('calculate elu', () => {
    const a = dl.tensor1d([1, -1, 0]);
    const result = dl.elu(a);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [1, -0.6321, 0]);
  });

  it('elu propagates NaN', () => {
    const a = dl.tensor1d([1, NaN]);
    const result = dl.elu(a);
    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [1, NaN]);
  });

  it('derivative', () => {
    const x = dl.tensor1d([1, 3, -2]);
    const dy = dl.tensor1d([5, 50, 500]);
    const gradients = dl.grad(a => dl.elu(a))(x, dy);

    expect(gradients.shape).toEqual(x.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [5, 50, 500 * Math.exp(-2)]);
  });
});

describeWithFlags('selu', ALL_ENVS, () => {
  const scaleAlpha = selu_util.SELU_SCALEALPHA;
  const scale = selu_util.SELU_SCALE;

  it('calculate selu', () => {
    const a = dl.tensor1d([1, -1, 0]);
    const result = dl.selu(a);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [1.0507, -1.1113, 0]);
  });

  it('selu propagates NaN', () => {
    const a = dl.tensor1d([1, NaN]);
    const result = dl.selu(a);
    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [1.0507, NaN]);
  });

  it('gradients: Scalar', () => {
    let aValue = 1;
    let dyValue = 1;
    let a = dl.scalar(aValue);
    let dy = dl.scalar(dyValue);

    let gradients = dl.grad(a => dl.selu(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [dyValue * scale]);

    aValue = -1;
    dyValue = 2;
    a = dl.scalar(aValue);
    dy = dl.scalar(dyValue);

    gradients = dl.grad(a => dl.selu(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [dyValue * scaleAlpha * Math.exp(aValue)]);
  });

  it('gradients: Tensor1D', () => {
    const aValues = [1, -1, 0];
    const dyValues = [1, 2, 3];
    const a = dl.tensor1d(aValues);
    const dy = dl.tensor1d(dyValues);

    const gradients = dl.grad(a => dl.selu(a))(a, dy);

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
    const a = dl.tensor2d(aValues, [2, 2]);
    const dy = dl.tensor2d(dyValues, [2, 2]);

    const gradients = dl.grad(a => dl.selu(a))(a, dy);

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
});

describeWithFlags('clip', ALL_ENVS, () => {
  it('basic', () => {
    const a = dl.tensor1d([3, -1, 0, 100, -7, 2]);
    const min = -1;
    const max = 50;

    const result = dl.clipByValue(a, min, max);

    expectArraysClose(result, [3, -1, 0, 50, -1, 2]);
  });

  it('propagates NaNs', () => {
    const a = dl.tensor1d([3, -1, 0, 100, -7, 2, NaN]);
    const min = -1;
    const max = 50;

    const result = dl.clipByValue(a, min, max);

    expectArraysClose(result, [3, -1, 0, 50, -1, 2, NaN]);
  });

  it('min greater than max', () => {
    const a = dl.tensor1d([3, -1, 0, 100, -7, 2]);
    const min = 1;
    const max = -1;

    const f = () => {
      dl.clipByValue(a, min, max);
    };
    expect(f).toThrowError();
  });

  it('derivative: 1D tensor', () => {
    const min = -1;
    const max = 2;
    const x = dl.tensor1d([3, -2, 1]);  // Only 1 is not clipped.
    const dy = dl.tensor1d([5, 50, 500]);
    const gradients = dl.grad(x => x.clipByValue(min, max))(x, dy);

    expect(gradients.shape).toEqual(x.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [0, 0, 500]);
  });

  it('derivative: scalar', () => {
    const min = -1;
    const max = 2;
    const x = dl.scalar(-10);  // Clipped.
    const dy = dl.scalar(5);
    const gradients = dl.grad(x => x.clipByValue(min, max))(x, dy);

    expect(gradients.shape).toEqual(x.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [0]);
  });
});
