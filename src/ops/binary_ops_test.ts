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
import {ALL_ENVS, describeWithFlags, expectArraysClose, expectArraysEqual} from '../test_util';

describeWithFlags('prelu', ALL_ENVS, () => {
  it('basic', () => {
    const x = dl.tensor1d([0, 1, -2, -4]);
    const a = dl.tensor1d([0.15, 0.2, 0.25, 0.15]);
    const result = dl.prelu(x, a);

    expect(result.shape).toEqual(x.shape);
    expectArraysClose(result, [0, 1, -0.5, -0.6]);
  });

  it('propagates NaN', () => {
    const x = dl.tensor1d([0, 1, NaN]);
    const a = dl.tensor1d([0.15, 0.2, 0.25]);
    const result = dl.prelu(x, a);

    expect(result.shape).toEqual(x.shape);
    expectArraysClose(result, [0, 1, NaN]);
  });

  it('derivative', () => {
    const x = dl.tensor1d([0.5, 3, -0.1, -4]);
    const a = dl.tensor1d([0.2, 0.4, 0.25, 0.15]);
    const dy = dl.tensor1d([1, 1, 1, 1]);

    const dx = dl.grad(x => dl.prelu(x, a))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expect(dx.dtype).toEqual('float32');
    expectArraysClose(dx, [1, 1, 0.25, 0.15]);
  });

  it('derivative propagates NaN', () => {
    const x = dl.tensor1d([0.5, -0.1, NaN]);
    const a = dl.tensor1d([0.2, 0.3, 0.25]);
    const dy = dl.tensor1d([5, 50, 500]);

    const dx = dl.grad(x => dl.prelu(x, a))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expect(dx.dtype).toEqual('float32');
    expectArraysClose(dx, [5, 50 * 0.3, NaN]);
  });
});

describeWithFlags('maximum', ALL_ENVS, () => {
  it('float32 and float32', () => {
    const a = dl.tensor1d([0.5, 3, -0.1, -4]);
    const b = dl.tensor1d([0.2, 0.4, 0.25, 0.15]);
    const result = dl.maximum(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [0.5, 3, 0.25, 0.15]);
  });

  it('int32 and int32', () => {
    const a = dl.tensor1d([1, 5, 2, 3], 'int32');
    const b = dl.tensor1d([2, 3, 1, 4], 'int32');
    const result = dl.maximum(a, b);

    expect(result.shape).toEqual(a.shape);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(result, [2, 5, 2, 4]);
  });

  it('bool and bool', () => {
    const a = dl.tensor1d([true, false, false, true], 'bool');
    const b = dl.tensor1d([false, false, true, true], 'bool');
    const result = dl.maximum(a, b);

    expect(result.shape).toEqual(a.shape);
    expect(result.dtype).toBe('bool');
    expectArraysEqual(result, [true, false, true, true]);
  });

  it('different dtypes throws error', () => {
    const a = dl.tensor1d([true, false, false, true], 'float32');
    const b = dl.tensor1d([false, false, true, true], 'int32');
    // tslint:disable-next-line:no-any
    expect(() => dl.maximum(a, b as any)).toThrowError();
  });

  it('propagates NaN', () => {
    const a = dl.tensor1d([0.5, -0.1, NaN]);
    const b = dl.tensor1d([0.2, 0.3, 0.25]);
    const result = dl.maximum(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [0.5, 0.3, NaN]);
  });

  it('broadcasts Tensor1D and scalar', () => {
    const a = dl.tensor1d([0.5, 3, -0.1, -4]);
    const b = dl.scalar(0.6);
    const result = dl.maximum(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [0.6, 3, 0.6, 0.6]);
  });

  it('broadcasts scalar and Tensor1D', () => {
    const a = dl.scalar(0.6);
    const b = dl.tensor1d([0.5, 3, -0.1, -4]);
    const result = dl.maximum(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(result, [0.6, 3, 0.6, 0.6]);
  });

  it('broadcasts Tensor1D and Tensor2D', () => {
    const a = dl.tensor1d([0.5, 0.3]);
    const b = dl.tensor2d([0.2, 0.4, 0.6, 0.15], [2, 2]);
    const result = dl.maximum(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(result, [0.5, 0.4, 0.6, 0.3]);
  });

  it('broadcasts 2x1 Tensor2D and 2x2 Tensor2D', () => {
    const a = dl.tensor2d([0.5, 0.3], [2, 1]);
    const b = dl.tensor2d([0.2, 0.4, 0.6, 0.15], [2, 2]);
    const result = dl.maximum(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(result, [0.5, 0.5, 0.6, 0.3]);
  });

  it('gradients: Scalar', () => {
    const a = dl.scalar(5.2);
    const b = dl.scalar(0.6);
    const dy = dl.scalar(3);

    const grads = dl.grads((a, b) => dl.maximum(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(da, [3 * 1]);
    expectArraysClose(db, [3 * 0]);
  });

  it('gradients: Tensor1D', () => {
    const a = dl.tensor1d([1.1, 2.6, 3, 5.9]);
    const b = dl.tensor1d([1.0, 2.7, 3, 5.8]);
    const dy = dl.tensor1d([1, 2, 3, 4]);

    const grads = dl.grads((a, b) => dl.maximum(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(da, [1 * 1, 2 * 0, 3 * 1, 4 * 1]);
    expectArraysClose(db, [1 * 0, 2 * 1, 3 * 0, 4 * 0]);
  });

  it('gradients: Tensor2D', () => {
    const a = dl.tensor2d([0.5, 0.3, 0.7, 0.9], [2, 2]);
    const b = dl.tensor2d([0.2, 0.4, 0.7, 0.15], [2, 2]);
    const dy = dl.tensor2d([1, 2, 3, 4], [2, 2]);

    const grads = dl.grads((a, b) => dl.maximum(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(da, [1 * 1, 2 * 0, 3 * 1, 4 * 1]);
    expectArraysClose(db, [1 * 0, 2 * 1, 3 * 0, 4 * 0]);
  });
});

describeWithFlags('squaredDifference', ALL_ENVS, () => {
  it('float32 and float32', () => {
    const a = dl.tensor1d([0.5, 3, -0.1, -4]);
    const b = dl.tensor1d([0.2, 0.4, 0.25, 0.15]);
    const result = dl.squaredDifference(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [
      Math.pow(0.5 - 0.2, 2), Math.pow(3 - 0.4, 2), Math.pow(-0.1 - 0.25, 2),
      Math.pow(-4 - 0.15, 2)
    ]);
  });

  it('int32 and int32', () => {
    const a = dl.tensor1d([1, 5, 2, 3], 'int32');
    const b = dl.tensor1d([2, 3, 1, 4], 'int32');
    const result = dl.squaredDifference(a, b);

    expect(result.shape).toEqual(a.shape);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(result, [
      Math.pow(1 - 2, 2), Math.pow(5 - 3, 2), Math.pow(2 - 1, 2),
      Math.pow(3 - 4, 2)
    ]);
  });

  it('different dtypes throws error', () => {
    const a = dl.tensor1d([0.5, 3, -0.1, -4], 'float32');
    const b = dl.tensor1d([2, 3, 1, 4], 'int32');
    // tslint:disable-next-line:no-any
    expect(() => dl.squaredDifference(a, b as any)).toThrowError();
  });

  it('propagates NaN', () => {
    const a = dl.tensor1d([0.5, -0.1, NaN]);
    const b = dl.tensor1d([0.2, 0.3, 0.25]);
    const result = dl.squaredDifference(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(
        result, [Math.pow(0.5 - 0.2, 2), Math.pow(-0.1 - 0.3, 2), NaN]);
  });

  it('broadcasts Tensor1D and scalar', () => {
    const a = dl.tensor1d([0.5, 3, -0.1, -4]);
    const b = dl.scalar(0.6);
    const result = dl.squaredDifference(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [
      Math.pow(0.5 - 0.6, 2), Math.pow(3 - 0.6, 2), Math.pow(-0.1 - 0.6, 2),
      Math.pow(-4 - 0.6, 2)
    ]);
  });

  it('broadcasts scalar and Tensor1D', () => {
    const a = dl.scalar(0.6);
    const b = dl.tensor1d([0.5, 3, -0.1, -4]);
    const result = dl.squaredDifference(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(result, [
      Math.pow(0.6 - 0.5, 2), Math.pow(0.6 - 3, 2), Math.pow(0.6 - (-0.1), 2),
      Math.pow(0.6 - (-4), 2)
    ]);
  });

  it('broadcasts Tensor1D and Tensor2D', () => {
    const a = dl.tensor1d([0.5, 0.3]);
    const b = dl.tensor2d([0.2, 0.4, 0.6, 0.15], [2, 2]);
    const result = dl.squaredDifference(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(result, [
      Math.pow(0.5 - 0.2, 2), Math.pow(0.3 - 0.4, 2), Math.pow(0.5 - 0.6, 2),
      Math.pow(0.3 - 0.15, 2)
    ]);
  });

  it('broadcasts 2x1 Tensor2D and 2x2 Tensor2D', () => {
    const a = dl.tensor2d([0.5, 0.3], [2, 1]);
    const b = dl.tensor2d([0.2, 0.4, 0.6, 0.15], [2, 2]);
    const result = dl.squaredDifference(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(result, [
      Math.pow(0.5 - 0.2, 2), Math.pow(0.5 - 0.4, 2), Math.pow(0.3 - 0.6, 2),
      Math.pow(0.3 - 0.15, 2)
    ]);
  });

  it('gradients: Scalar', () => {
    const a = dl.scalar(5.2);
    const b = dl.scalar(0.6);
    const dy = dl.scalar(3);

    const grads = dl.grads((a, b) => dl.squaredDifference(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(da, [3 * 2 * (5.2 - 0.6)]);
    expectArraysClose(db, [3 * 2 * (0.6 - 5.2)]);
  });

  it('gradients: Tensor1D', () => {
    const a = dl.tensor1d([1.1, 2.6, 3, 5.9]);
    const b = dl.tensor1d([1.0, 2.7, 3, 5.8]);
    const dy = dl.tensor1d([1, 2, 3, 1]);

    const grads = dl.grads((a, b) => dl.squaredDifference(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(da, [
      1 * 2 * (1.1 - 1.0), 2 * 2 * (2.6 - 2.7), 3 * 2 * (3 - 3),
      1 * 2 * (5.9 - 5.8)
    ]);
    expectArraysClose(db, [
      1 * 2 * (1.0 - 1.1), 2 * 2 * (2.7 - 2.6), 3 * 2 * (3 - 3),
      1 * 2 * (5.8 - 5.9)
    ]);
  });

  it('gradients: Tensor2D', () => {
    const a = dl.tensor2d([0.5, 0.3, 0.7, 0.9], [2, 2]);
    const b = dl.tensor2d([0.2, 0.4, 0.7, 0.15], [2, 2]);
    const dy = dl.tensor2d([1, 2, 3, 4], [2, 2]);

    const grads = dl.grads((a, b) => dl.squaredDifference(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(da, [
      1 * 2 * (0.5 - 0.2), 2 * 2 * (0.3 - 0.4), 3 * 2 * (0.7 - 0.7),
      4 * 2 * (0.9 - 0.15)
    ]);
    expectArraysClose(db, [
      1 * 2 * (0.2 - 0.5), 2 * 2 * (0.4 - 0.3), 3 * 2 * (0.7 - 0.7),
      4 * 2 * (0.15 - 0.9)
    ]);
  });
});

describeWithFlags('minimum', ALL_ENVS, () => {
  it('float32 and float32', () => {
    const a = dl.tensor1d([0.5, 3, -0.1, -4]);
    const b = dl.tensor1d([0.2, 0.4, 0.25, 0.15]);
    const result = dl.minimum(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [0.2, 0.4, -0.1, -4]);
  });

  it('int32 and int32', () => {
    const a = dl.tensor1d([1, 5, 2, 3], 'int32');
    const b = dl.tensor1d([2, 3, 1, 4], 'int32');
    const result = dl.minimum(a, b);

    expect(result.shape).toEqual(a.shape);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(result, [1, 3, 1, 3]);
  });

  it('bool and bool', () => {
    const a = dl.tensor1d([true, false, false, true], 'bool');
    const b = dl.tensor1d([false, false, true, true], 'bool');
    const result = dl.minimum(a, b);

    expect(result.shape).toEqual(a.shape);
    expect(result.dtype).toBe('bool');
    expectArraysEqual(result, [false, false, false, true]);
  });

  it('different dtypes throws error', () => {
    const a = dl.tensor1d([true, false, false, true], 'float32');
    const b = dl.tensor1d([false, false, true, true], 'int32');
    // tslint:disable-next-line:no-any
    expect(() => dl.minimum(a, b as any)).toThrowError();
  });

  it('propagates NaN', () => {
    const a = dl.tensor1d([0.5, -0.1, NaN]);
    const b = dl.tensor1d([0.2, 0.3, 0.25]);
    const result = dl.minimum(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [0.2, -0.1, NaN]);
  });

  it('broadcasts Tensor1D and scalar', () => {
    const a = dl.tensor1d([0.5, 3, -0.1, -4]);
    const b = dl.scalar(0.6);
    const result = dl.minimum(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [0.5, 0.6, -0.1, -4]);
  });

  it('broadcasts scalar and Tensor1D', () => {
    const a = dl.scalar(0.6);
    const b = dl.tensor1d([0.5, 3, -0.1, -4]);
    const result = dl.minimum(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(result, [0.5, 0.6, -0.1, -4]);
  });

  it('broadcasts Tensor1D and Tensor2D', () => {
    const a = dl.tensor1d([0.5, 0.3]);
    const b = dl.tensor2d([0.2, 0.4, 0.6, 0.15], [2, 2]);
    const result = dl.minimum(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(result, [0.2, 0.3, 0.5, 0.15]);
  });

  it('broadcasts 2x1 Tensor2D and 2x2 Tensor2D', () => {
    const a = dl.tensor2d([0.5, 0.3], [2, 1]);
    const b = dl.tensor2d([0.2, 0.4, 0.6, 0.15], [2, 2]);
    const result = dl.minimum(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(result, [0.2, 0.4, 0.3, 0.15]);
  });

  it('gradients: Scalar', () => {
    const a = dl.scalar(5.2);
    const b = dl.scalar(0.6);
    const dy = dl.scalar(3);

    const grads = dl.grads((a, b) => dl.minimum(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(da, [3 * 0]);
    expectArraysClose(db, [3 * 1]);
  });

  it('gradients: Tensor1D', () => {
    const a = dl.tensor1d([1.1, 2.6, 3, 5.9]);
    const b = dl.tensor1d([1.0, 2.7, 3, 5.8]);
    const dy = dl.tensor1d([1, 2, 3, 4]);

    const grads = dl.grads((a, b) => dl.minimum(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(da, [1 * 0, 2 * 1, 3 * 1, 4 * 0]);
    expectArraysClose(db, [1 * 1, 2 * 0, 3 * 0, 4 * 1]);
  });

  it('gradients: Tensor2D', () => {
    const a = dl.tensor2d([0.5, 0.3, 0.7, 0.9], [2, 2]);
    const b = dl.tensor2d([0.2, 0.4, 0.7, 0.15], [2, 2]);
    const dy = dl.tensor2d([1, 2, 3, 4], [2, 2]);

    const grads = dl.grads((a, b) => dl.minimum(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(da, [1 * 0, 2 * 1, 3 * 1, 4 * 0]);
    expectArraysClose(db, [1 * 1, 2 * 0, 3 * 0, 4 * 1]);
  });
});

describeWithFlags('atan2', ALL_ENVS, () => {
  it('same shape', () => {
    const aValues = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    const bValues = [1.0, 2.5, 3.5, 4.5, 2.0, 5.0];

    const a = dl.tensor2d(aValues, [2, 3]);
    const c = dl.tensor2d(bValues, [2, 3]);

    const r = dl.atan2(a, c);
    const expected = [];

    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.atan2(aValues[i], bValues[i]);
    }
    expectArraysClose(r, expected);
  });

  it('propagates NaNs', () => {
    const a = dl.tensor2d([1.0, 2.0], [2, 1]);
    const c = dl.tensor2d([3.0, NaN], [2, 1]);

    const r = dl.atan2(a, c);

    expectArraysClose(r, [Math.atan2(1.0, 3.0), NaN]);
  });

  it('broadcasting same rank Tensors different shape', () => {
    const aValues = [1.0, 2.0, -3.0, -4.0];
    const bValues = [2.0, 3.0];

    const a = dl.tensor2d(aValues, [2, 2]);
    const b = dl.tensor2d(bValues, [2, 1]);

    const result = dl.atan2(a, b);

    expect(result.shape).toEqual([2, 2]);
    const expected = [
      Math.atan2(1.0, 2.0), Math.atan2(2.0, 2.0), Math.atan2(-3.0, 3.0),
      Math.atan2(-4.0, 3.0)
    ];
    expectArraysClose(result, expected);
  });

  it('throws when passed tensors of different shapes', () => {
    const a = dl.tensor2d([1, 2, -3, -4, 5, 6], [2, 3]);
    const b = dl.tensor2d([5, 3, 4, -7], [2, 2]);

    expect(() => dl.atan2(a, b)).toThrowError();
    expect(() => dl.atan2(b, a)).toThrowError();
  });

  it('throws when passed tensors of different types', () => {
    const a = dl.tensor2d([1, 2, -3, -4, 5, 6], [2, 3]);
    const b = dl.tensor2d([5.0, 3.0, 4.0, -7.0], [2, 2]);

    expect(() => dl.atan2(a, b)).toThrowError();
    expect(() => dl.atan2(b, a)).toThrowError();
  });

  it('atan2 of scalar and array propagates NaNs', () => {
    const c = dl.scalar(NaN);
    const a = dl.tensor2d([1, 2, 3], [1, 3]);

    const r = dl.atan2(c, a);

    expectArraysEqual(r, [NaN, NaN, NaN]);
  });

  it('atan2 of scalar and array', () => {
    const aValues = [1, 2, 3, 4, 5, 6];

    const a = dl.tensor2d(aValues, [2, 3]);
    const c = dl.scalar(2);

    const r = dl.atan2(a, c);
    const expected = [];

    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.atan2(aValues[i], 2);
    }
    expectArraysClose(r, expected);
  });

  it('gradient: Scalar', () => {
    const a = dl.scalar(5);
    const b = dl.scalar(2);
    const dy = dl.scalar(4);

    const grads = dl.grads((a, b) => dl.atan2(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [4 * 2 / 29]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(db, [4 * -5 / 29]);
  });

  it('gradient: Tensor1D', () => {
    const a = dl.tensor1d([1, 2, 3]);
    const b = dl.tensor1d([3, 4, 5]);
    const dy = dl.tensor1d([1, 10, 20]);

    const grads = dl.grads((a, b) => dl.atan2(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(da, [1 * 3 / 10, 10 * 4 / 20, 20 * 5 / 34]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(db, [-1 * 1 / 10, -10 * 2 / 20, -20 * 3 / 34]);
  });

  it('gradient: Tensor2D', () => {
    const a = dl.tensor2d([3, 1, 2, 3], [2, 2]);
    const b = dl.tensor2d([1, 3, 4, 5], [2, 2]);
    const dy = dl.tensor2d([1, 10, 15, 20], [2, 2]);

    const grads = dl.grads((a, b) => dl.atan2(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [1 * 1 / 10, 10 * 3 / 10, 15 * 4 / 20, 20 * 5 / 34]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(
        db, [-1 * 3 / 10, -10 * 1 / 10, -15 * 2 / 20, -20 * 3 / 34]);
  });

  it('gradient: scalar / Tensor1D', () => {
    const a = dl.scalar(2);
    const b = dl.tensor1d([3, 4, 5]);
    const dy = dl.tensor1d([6, 7, 8]);

    const grads = dl.grads((a, b) => dl.atan2(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [6 * 3 / 13 + 7 * 4 / 20 + 8 * 5 / 29]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(db, [-6 * 2 / 13, -7 * 2 / 20, -8 * 2 / 29]);
  });

  it('gradient: Tensor2D / scalar', () => {
    const a = dl.tensor2d([[2, 3], [4, 5]], [2, 2]);
    const b = dl.scalar(2);
    const dy = dl.tensor2d([[6, 7], [8, 9]], [2, 2]);

    const grads = dl.grads((a, b) => dl.atan2(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [6 * 2 / 8, 7 * 2 / 13, 8 * 2 / 20, 9 * 2 / 29]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(
        db, [-6 * 2 / 8 + -7 * 3 / 13 + -8 * 4 / 20 + -9 * 5 / 29]);
  });

  it('gradient: Tensor2D / Tensor2D w/ broadcast', () => {
    const a = dl.tensor2d([3, 4], [2, 1]);
    const b = dl.tensor2d([[2, 3], [4, 5]], [2, 2]);
    const dy = dl.tensor2d([[6, 7], [8, 9]], [2, 2]);

    const grads = dl.grads((a, b) => dl.atan2(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [6 * 2 / 13 + 7 * 3 / 18, 8 * 4 / 32 + 9 * 5 / 41]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(db, [-6 * 3 / 13, -7 * 3 / 18, -8 * 4 / 32, -9 * 4 / 41]);
  });
});
