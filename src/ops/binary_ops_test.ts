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
import {ALL_ENVS, expectArraysClose, expectArraysEqual} from '../test_util';

describeWithFlags('prelu', ALL_ENVS, () => {
  it('basic', () => {
    const x = tf.tensor1d([0, 1, -2, -4]);
    const a = tf.tensor1d([0.15, 0.2, 0.25, 0.15]);
    const result = tf.prelu(x, a);

    expect(result.shape).toEqual(x.shape);
    expectArraysClose(result, [0, 1, -0.5, -0.6]);
  });

  it('derivative', () => {
    const x = tf.tensor1d([0.5, 3, -0.1, -4]);
    const a = tf.tensor1d([0.2, 0.4, 0.25, 0.15]);
    const dy = tf.tensor1d([1, 1, 1, 1]);

    const dx = tf.grad(x => tf.prelu(x, a))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expect(dx.dtype).toEqual('float32');
    expectArraysClose(dx, [1, 1, 0.25, 0.15]);
  });

  it('throws when passed x as a non-tensor', () => {
    expect(() => tf.prelu({} as tf.Tensor, tf.scalar(1)))
        .toThrowError(/Argument 'x' passed to 'prelu' must be a Tensor/);
  });
  it('throws when passed alpha as a non-tensor', () => {
    expect(() => tf.prelu(tf.scalar(1), {} as tf.Tensor))
        .toThrowError(/Argument 'alpha' passed to 'prelu' must be a Tensor/);
  });
});

describeWithFlags('maximum', ALL_ENVS, () => {
  it('float32 and float32', () => {
    const a = tf.tensor1d([0.5, 3, -0.1, -4]);
    const b = tf.tensor1d([0.2, 0.4, 0.25, 0.15]);
    const result = tf.maximum(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [0.5, 3, 0.25, 0.15]);
  });

  it('int32 and int32', () => {
    const a = tf.tensor1d([1, 5, 2, 3], 'int32');
    const b = tf.tensor1d([2, 3, 1, 4], 'int32');
    const result = tf.maximum(a, b);

    expect(result.shape).toEqual(a.shape);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(result, [2, 5, 2, 4]);
  });

  it('bool and bool', () => {
    const a = tf.tensor1d([true, false, false, true], 'bool');
    const b = tf.tensor1d([false, false, true, true], 'bool');
    const result = tf.maximum(a, b);

    expect(result.shape).toEqual(a.shape);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(result, [1, 0, 1, 1]);
  });

  it('different dtypes throws error', () => {
    const a = tf.tensor1d([true, false, false, true], 'float32');
    const b = tf.tensor1d([false, false, true, true], 'int32');
    // tslint:disable-next-line:no-any
    expect(() => tf.maximum(a, b as any)).toThrowError();
  });

  it('propagates NaN', () => {
    const a = tf.tensor1d([0.5, -0.1, NaN]);
    const b = tf.tensor1d([0.2, 0.3, 0.25]);
    const result = tf.maximum(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [0.5, 0.3, NaN]);
  });

  it('broadcasts Tensor1D and scalar', () => {
    const a = tf.tensor1d([0.5, 3, -0.1, -4]);
    const b = tf.scalar(0.6);
    const result = tf.maximum(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [0.6, 3, 0.6, 0.6]);
  });

  it('broadcasts scalar and Tensor1D', () => {
    const a = tf.scalar(0.6);
    const b = tf.tensor1d([0.5, 3, -0.1, -4]);
    const result = tf.maximum(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(result, [0.6, 3, 0.6, 0.6]);
  });

  it('broadcasts Tensor1D and Tensor2D', () => {
    const a = tf.tensor1d([0.5, 0.3]);
    const b = tf.tensor2d([0.2, 0.4, 0.6, 0.15], [2, 2]);
    const result = tf.maximum(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(result, [0.5, 0.4, 0.6, 0.3]);
  });

  it('broadcasts 2x1 Tensor2D and 2x2 Tensor2D', () => {
    const a = tf.tensor2d([0.5, 0.3], [2, 1]);
    const b = tf.tensor2d([0.2, 0.4, 0.6, 0.15], [2, 2]);
    const result = tf.maximum(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(result, [0.5, 0.5, 0.6, 0.3]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(5.2);
    const b = tf.scalar(0.6);
    const dy = tf.scalar(3);

    const grads = tf.grads((a, b) => tf.maximum(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(da, [3 * 1]);
    expectArraysClose(db, [3 * 0]);
  });

  it('gradients: Tensor1D', () => {
    const a = tf.tensor1d([1.1, 2.6, 3, 5.9]);
    const b = tf.tensor1d([1.0, 2.7, 3, 5.8]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const grads = tf.grads((a, b) => tf.maximum(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(da, [1 * 1, 2 * 0, 3 * 1, 4 * 1]);
    expectArraysClose(db, [1 * 0, 2 * 1, 3 * 0, 4 * 0]);
  });

  it('gradients: Tensor2D', () => {
    const a = tf.tensor2d([0.5, 0.3, 0.7, 0.9], [2, 2]);
    const b = tf.tensor2d([0.2, 0.4, 0.7, 0.15], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const grads = tf.grads((a, b) => tf.maximum(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(da, [1 * 1, 2 * 0, 3 * 1, 4 * 1]);
    expectArraysClose(db, [1 * 0, 2 * 1, 3 * 0, 4 * 0]);
  });

  it('throws when passed a as a non-tensor', () => {
    expect(() => tf.maximum({} as tf.Tensor, tf.scalar(1)))
        .toThrowError(/Argument 'a' passed to 'maximum' must be a Tensor/);
  });
  it('throws when passed b as a non-tensor', () => {
    expect(() => tf.maximum(tf.scalar(1), {} as tf.Tensor))
        .toThrowError(/Argument 'b' passed to 'maximum' must be a Tensor/);
  });

  it('accepts a tensor-like object', () => {
    const a = [[0.5, 3], [-0.1, -4]];
    const b = [[0.2, 0.4], [0.25, 0.15]];
    const result = tf.maximum(a, b);

    expect(result.shape).toEqual([2, 2]);
    expectArraysClose(result, [0.5, 3, 0.25, 0.15]);
  });
});

describeWithFlags('squaredDifference', ALL_ENVS, () => {
  it('float32 and float32', () => {
    const a = tf.tensor1d([0.5, 3, -0.1, -4]);
    const b = tf.tensor1d([0.2, 0.4, 0.25, 0.15]);
    const result = tf.squaredDifference(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [
      Math.pow(0.5 - 0.2, 2), Math.pow(3 - 0.4, 2), Math.pow(-0.1 - 0.25, 2),
      Math.pow(-4 - 0.15, 2)
    ]);
  });

  it('int32 and int32', () => {
    const a = tf.tensor1d([1, 5, 2, 3], 'int32');
    const b = tf.tensor1d([2, 3, 1, 4], 'int32');
    const result = tf.squaredDifference(a, b);

    expect(result.shape).toEqual(a.shape);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(result, [
      Math.pow(1 - 2, 2), Math.pow(5 - 3, 2), Math.pow(2 - 1, 2),
      Math.pow(3 - 4, 2)
    ]);
  });

  it('different dtypes throws error', () => {
    const a = tf.tensor1d([0.5, 3, -0.1, -4], 'float32');
    const b = tf.tensor1d([2, 3, 1, 4], 'int32');
    // tslint:disable-next-line:no-any
    expect(() => tf.squaredDifference(a, b as any)).toThrowError();
  });

  it('propagates NaN', () => {
    const a = tf.tensor1d([0.5, -0.1, NaN]);
    const b = tf.tensor1d([0.2, 0.3, 0.25]);
    const result = tf.squaredDifference(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(
        result, [Math.pow(0.5 - 0.2, 2), Math.pow(-0.1 - 0.3, 2), NaN]);
  });

  it('broadcasts Tensor1D and scalar', () => {
    const a = tf.tensor1d([0.5, 3, -0.1, -4]);
    const b = tf.scalar(0.6);
    const result = tf.squaredDifference(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [
      Math.pow(0.5 - 0.6, 2), Math.pow(3 - 0.6, 2), Math.pow(-0.1 - 0.6, 2),
      Math.pow(-4 - 0.6, 2)
    ]);
  });

  it('broadcasts scalar and Tensor1D', () => {
    const a = tf.scalar(0.6);
    const b = tf.tensor1d([0.5, 3, -0.1, -4]);
    const result = tf.squaredDifference(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(result, [
      Math.pow(0.6 - 0.5, 2), Math.pow(0.6 - 3, 2), Math.pow(0.6 - (-0.1), 2),
      Math.pow(0.6 - (-4), 2)
    ]);
  });

  it('broadcasts Tensor1D and Tensor2D', () => {
    const a = tf.tensor1d([0.5, 0.3]);
    const b = tf.tensor2d([0.2, 0.4, 0.6, 0.15], [2, 2]);
    const result = tf.squaredDifference(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(result, [
      Math.pow(0.5 - 0.2, 2), Math.pow(0.3 - 0.4, 2), Math.pow(0.5 - 0.6, 2),
      Math.pow(0.3 - 0.15, 2)
    ]);
  });

  it('broadcasts 2x1 Tensor2D and 2x2 Tensor2D', () => {
    const a = tf.tensor2d([0.5, 0.3], [2, 1]);
    const b = tf.tensor2d([0.2, 0.4, 0.6, 0.15], [2, 2]);
    const result = tf.squaredDifference(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(result, [
      Math.pow(0.5 - 0.2, 2), Math.pow(0.5 - 0.4, 2), Math.pow(0.3 - 0.6, 2),
      Math.pow(0.3 - 0.15, 2)
    ]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(5.2);
    const b = tf.scalar(0.6);
    const dy = tf.scalar(3);

    const grads = tf.grads((a, b) => tf.squaredDifference(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(da, [3 * 2 * (5.2 - 0.6)]);
    expectArraysClose(db, [3 * 2 * (0.6 - 5.2)]);
  });

  it('gradients: Tensor1D', () => {
    const a = tf.tensor1d([1.1, 2.6, 3, 5.9]);
    const b = tf.tensor1d([1.0, 2.7, 3, 5.8]);
    const dy = tf.tensor1d([1, 2, 3, 1]);

    const grads = tf.grads((a, b) => tf.squaredDifference(a, b));
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
    const a = tf.tensor2d([0.5, 0.3, 0.7, 0.9], [2, 2]);
    const b = tf.tensor2d([0.2, 0.4, 0.7, 0.15], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const grads = tf.grads((a, b) => tf.squaredDifference(a, b));
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

  it('throws when passed a as a non-tensor', () => {
    expect(() => tf.squaredDifference({} as tf.Tensor, tf.scalar(1)))
        .toThrowError(
            /Argument 'a' passed to 'squaredDifference' must be a Tensor/);
  });
  it('throws when passed b as a non-tensor', () => {
    expect(() => tf.squaredDifference(tf.scalar(1), {} as tf.Tensor))
        .toThrowError(
            /Argument 'b' passed to 'squaredDifference' must be a Tensor/);
  });

  it('accepts a tensor-like object', () => {
    const a = [[0.5, 3], [-0.1, -4]];
    const b = 0.6;
    const result = tf.squaredDifference(a, b);

    expect(result.shape).toEqual([2, 2]);
    expectArraysClose(result, [
      Math.pow(0.5 - 0.6, 2), Math.pow(3 - 0.6, 2), Math.pow(-0.1 - 0.6, 2),
      Math.pow(-4 - 0.6, 2)
    ]);
  });
});

describeWithFlags('minimum', ALL_ENVS, () => {
  it('float32 and float32', () => {
    const a = tf.tensor1d([0.5, 3, -0.1, -4]);
    const b = tf.tensor1d([0.2, 0.4, 0.25, 0.15]);
    const result = tf.minimum(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [0.2, 0.4, -0.1, -4]);
  });

  it('int32 and int32', () => {
    const a = tf.tensor1d([1, 5, 2, 3], 'int32');
    const b = tf.tensor1d([2, 3, 1, 4], 'int32');
    const result = tf.minimum(a, b);

    expect(result.shape).toEqual(a.shape);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(result, [1, 3, 1, 3]);
  });

  it('bool and bool', () => {
    const a = tf.tensor1d([true, false, false, true], 'bool');
    const b = tf.tensor1d([false, false, true, true], 'bool');
    const result = tf.minimum(a, b);

    expect(result.shape).toEqual(a.shape);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(result, [0, 0, 0, 1]);
  });

  it('different dtypes throws error', () => {
    const a = tf.tensor1d([true, false, false, true], 'float32');
    const b = tf.tensor1d([false, false, true, true], 'int32');
    // tslint:disable-next-line:no-any
    expect(() => tf.minimum(a, b as any)).toThrowError();
  });

  it('propagates NaN', () => {
    const a = tf.tensor1d([0.5, -0.1, NaN]);
    const b = tf.tensor1d([0.2, 0.3, 0.25]);
    const result = tf.minimum(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [0.2, -0.1, NaN]);
  });

  it('broadcasts Tensor1D and scalar', () => {
    const a = tf.tensor1d([0.5, 3, -0.1, -4]);
    const b = tf.scalar(0.6);
    const result = tf.minimum(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [0.5, 0.6, -0.1, -4]);
  });

  it('broadcasts scalar and Tensor1D', () => {
    const a = tf.scalar(0.6);
    const b = tf.tensor1d([0.5, 3, -0.1, -4]);
    const result = tf.minimum(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(result, [0.5, 0.6, -0.1, -4]);
  });

  it('broadcasts Tensor1D and Tensor2D', () => {
    const a = tf.tensor1d([0.5, 0.3]);
    const b = tf.tensor2d([0.2, 0.4, 0.6, 0.15], [2, 2]);
    const result = tf.minimum(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(result, [0.2, 0.3, 0.5, 0.15]);
  });

  it('broadcasts 2x1 Tensor2D and 2x2 Tensor2D', () => {
    const a = tf.tensor2d([0.5, 0.3], [2, 1]);
    const b = tf.tensor2d([0.2, 0.4, 0.6, 0.15], [2, 2]);
    const result = tf.minimum(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(result, [0.2, 0.4, 0.3, 0.15]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(5.2);
    const b = tf.scalar(0.6);
    const dy = tf.scalar(3);

    const grads = tf.grads((a, b) => tf.minimum(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(da, [3 * 0]);
    expectArraysClose(db, [3 * 1]);
  });

  it('gradients: Tensor1D', () => {
    const a = tf.tensor1d([1.1, 2.6, 3, 5.9]);
    const b = tf.tensor1d([1.0, 2.7, 3, 5.8]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const grads = tf.grads((a, b) => tf.minimum(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(da, [1 * 0, 2 * 1, 3 * 1, 4 * 0]);
    expectArraysClose(db, [1 * 1, 2 * 0, 3 * 0, 4 * 1]);
  });

  it('gradients: Tensor2D', () => {
    const a = tf.tensor2d([0.5, 0.3, 0.7, 0.9], [2, 2]);
    const b = tf.tensor2d([0.2, 0.4, 0.7, 0.15], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const grads = tf.grads((a, b) => tf.minimum(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(da, [1 * 0, 2 * 1, 3 * 1, 4 * 0]);
    expectArraysClose(db, [1 * 1, 2 * 0, 3 * 0, 4 * 1]);
  });

  it('throws when passed a as a non-tensor', () => {
    expect(() => tf.minimum({} as tf.Tensor, tf.scalar(1)))
        .toThrowError(/Argument 'a' passed to 'minimum' must be a Tensor/);
  });
  it('throws when passed b as a non-tensor', () => {
    expect(() => tf.minimum(tf.scalar(1), {} as tf.Tensor))
        .toThrowError(/Argument 'b' passed to 'minimum' must be a Tensor/);
  });

  it('accepts a tensor-like object', () => {
    const a = [[0.5, 3], [-0.1, -4]];
    const b = [[0.2, 0.4], [0.25, 0.15]];
    const result = tf.minimum(a, b);

    expect(result.shape).toEqual([2, 2]);
    expectArraysClose(result, [0.2, 0.4, -0.1, -4]);
  });
});

describeWithFlags('mod', ALL_ENVS, () => {
  it('float32 and float32', () => {
    const a = tf.tensor1d([0.5, 3, -0.1, -4]);
    const b = tf.tensor1d([0.2, 0.4, 0.25, 0.15]);
    const result = tf.mod(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [0.1, 0.2, 0.15, 0.05]);
  });

  it('int32 and int32', () => {
    const a = tf.tensor1d([1, 5, 2, 3], 'int32');
    const b = tf.tensor1d([2, 3, 1, 4], 'int32');
    const result = tf.mod(a, b);

    expect(result.shape).toEqual(a.shape);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(result, [1, 2, 0, 3]);
  });

  it('different dtypes throws error', () => {
    const a = tf.tensor1d([1.1, 2.2, 3.3, 4.4], 'float32');
    const b = tf.tensor1d([1, 2, 3, 4], 'int32');
    // tslint:disable-next-line:no-any
    expect(() => tf.mod(a, b as any)).toThrowError();
  });

  it('propagates NaN', () => {
    const a = tf.tensor1d([5, -1, NaN]);
    const b = tf.tensor1d([2, 3, 0.25]);
    const result = tf.mod(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [1, 2, NaN]);
  });

  it('broadcasts Tensor1D and scalar', () => {
    const a = tf.tensor1d([0.5, 2.5, -0.1, -4], 'float32');
    const b = tf.scalar(0.6);
    const result = tf.mod(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [0.5, 0.1, 0.5, 0.2]);
  });

  it('broadcasts scalar and Tensor1D', () => {
    // TODO(manraj): Fix for case fmod(0.6, -0.1)
    const a = tf.scalar(2);
    const b = tf.tensor1d([3, 3, -1, -4]);
    const result = tf.mod(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(result, [2, 2, 0, -2]);
  });

  it('broadcasts Tensor1D and Tensor2D', () => {
    const a = tf.tensor1d([0.5, 0.3]);
    const b = tf.tensor2d([0.2, 0.4, 0.6, 0.15], [2, 2]);
    const result = tf.mod(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(result, [0.1, 0.3, 0.5, 0.0]);
  });

  it('broadcasts 2x1 Tensor2D and 2x2 Tensor2D', () => {
    const a = tf.tensor2d([0.5, 0.3], [2, 1]);
    const b = tf.tensor2d([0.2, 0.4, 0.6, 0.15], [2, 2]);
    const result = tf.mod(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(result, [0.1, 0.1, 0.3, 0.0]);
  });

  it('gradients: Scalar', () => {
    const a = tf.scalar(5.2);
    const b = tf.scalar(0.6);
    const dy = tf.scalar(3);

    const grads = tf.grads((a, b) => tf.mod(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(da, [3]);
    expectArraysClose(db, [3 * -1 * Math.floor(5.2 / 0.6)]);
  });

  it('gradients: Tensor1D', () => {
    const a = tf.tensor1d([1.1, 2.6, 3, 5.9]);
    const b = tf.tensor1d([1.0, 2.7, 3, 5.8]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const grads = tf.grads((a, b) => tf.mod(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(da, [1 * 1, 2 * 1, 3 * 1, 4 * 1]);
    expectArraysClose(db, [
      1 * -1 * Math.floor(1.1 / 1.0), 2 * -1 * Math.floor(2.6 / 2.7),
      3 * -1 * Math.floor(3 / 3), 4 * -1 * Math.floor(5.9 / 5.8)
    ]);
  });

  it('gradients: Tensor2D', () => {
    const a = tf.tensor2d([0.5, 0.3, 0.7, 0.91], [2, 2]);
    const b = tf.tensor2d([0.2, 0.4, 0.7, 0.15], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const grads = tf.grads((a, b) => tf.mod(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(da, [1 * 1, 2 * 1, 3 * 1, 4 * 1]);
    expectArraysClose(db, [
      1 * -1 * Math.floor(0.5 / 0.2), 2 * -1 * Math.floor(0.3 / 0.4),
      3 * -1 * Math.floor(0.7 / 0.7), 4 * -1 * Math.floor(0.91 / 0.15)
    ]);
  });

  it('gradients: broadcasts scalar and Tensor1D', () => {
    const a = tf.scalar(0.7);
    const b = tf.tensor1d([0.2, 0.3, 0.4, 0.5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const grads = tf.grads((a, b) => tf.mod(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(da, [1 + 2 + 3 + 4]);
    expectArraysClose(db, [
      1 * -1 * Math.floor(0.7 / 0.2), 2 * -1 * Math.floor(0.7 / 0.3),
      3 * -1 * Math.floor(0.7 / 0.4), 4 * -1 * Math.floor(0.7 / 0.5)
    ]);
  });

  it('broadcasts Tensor1D and Tensor2D', () => {
    const a = tf.tensor1d([0.5, 0.3]);
    const b = tf.tensor2d([0.2, 0.4, 0.7, 0.15], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const grads = tf.grads((a, b) => tf.mod(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(da, [1 * 1 + 3 * 1, 2 * 1 + 4 * 1]);
    expectArraysClose(db, [
      1 * -1 * Math.floor(0.5 / 0.2), 2 * -1 * Math.floor(0.3 / 0.4),
      3 * -1 * Math.floor(0.5 / 0.7), 4 * -1 * Math.floor(0.3 / 0.15)
    ]);
  });

  it('throws when passed a as a non-tensor', () => {
    expect(() => tf.mod({} as tf.Tensor, tf.scalar(1)))
        .toThrowError(/Argument 'a' passed to 'mod' must be a Tensor/);
  });
  it('throws when passed b as a non-tensor', () => {
    expect(() => tf.mod(tf.scalar(1), {} as tf.Tensor))
        .toThrowError(/Argument 'b' passed to 'mod' must be a Tensor/);
  });

  it('accepts a tensor-like object', () => {
    const a = [[0.5, 3], [-0.1, -4]];
    const b = [[0.2, 0.4], [0.25, 0.15]];
    const result = tf.mod(a, b);

    expect(result.shape).toEqual([2, 2]);
    expectArraysClose(result, [0.1, 0.2, 0.15, 0.05]);
  });
});

describeWithFlags('atan2', ALL_ENVS, () => {
  it('same shape', () => {
    const aValues = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    const bValues = [1.0, 2.5, 3.5, 4.5, 2.0, 5.0];

    const a = tf.tensor2d(aValues, [2, 3]);
    const c = tf.tensor2d(bValues, [2, 3]);

    const r = tf.atan2(a, c);
    const expected = [];

    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.atan2(aValues[i], bValues[i]);
    }
    expectArraysClose(r, expected);
  });

  it('propagates NaNs', () => {
    const a = tf.tensor2d([1.0, 2.0], [2, 1]);
    const c = tf.tensor2d([3.0, NaN], [2, 1]);

    const r = tf.atan2(a, c);

    expectArraysClose(r, [Math.atan2(1.0, 3.0), NaN]);
  });

  it('broadcasting same rank Tensors different shape', () => {
    const aValues = [1.0, 2.0, -3.0, -4.0];
    const bValues = [2.0, 3.0];

    const a = tf.tensor2d(aValues, [2, 2]);
    const b = tf.tensor2d(bValues, [2, 1]);

    const result = tf.atan2(a, b);

    expect(result.shape).toEqual([2, 2]);
    const expected = [
      Math.atan2(1.0, 2.0), Math.atan2(2.0, 2.0), Math.atan2(-3.0, 3.0),
      Math.atan2(-4.0, 3.0)
    ];
    expectArraysClose(result, expected);
  });

  it('throws when passed tensors of different shapes', () => {
    const a = tf.tensor2d([1, 2, -3, -4, 5, 6], [2, 3]);
    const b = tf.tensor2d([5, 3, 4, -7], [2, 2]);

    expect(() => tf.atan2(a, b)).toThrowError();
    expect(() => tf.atan2(b, a)).toThrowError();
  });

  it('throws when passed tensors of different types', () => {
    const a = tf.tensor2d([1, 2, -3, -4, 5, 6], [2, 3]);
    const b = tf.tensor2d([5.0, 3.0, 4.0, -7.0], [2, 2]);

    expect(() => tf.atan2(a, b)).toThrowError();
    expect(() => tf.atan2(b, a)).toThrowError();
  });

  it('atan2 of scalar and array propagates NaNs', () => {
    const c = tf.scalar(NaN);
    const a = tf.tensor2d([1, 2, 3], [1, 3]);

    const r = tf.atan2(c, a);

    expectArraysEqual(r, [NaN, NaN, NaN]);
  });

  it('atan2 of scalar and array', () => {
    const aValues = [1, 2, 3, 4, 5, 6];

    const a = tf.tensor2d(aValues, [2, 3]);
    const c = tf.scalar(2);

    const r = tf.atan2(a, c);
    const expected = [];

    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.atan2(aValues[i], 2);
    }
    expectArraysClose(r, expected);
  });

  it('gradient: Scalar', () => {
    const a = tf.scalar(5);
    const b = tf.scalar(2);
    const dy = tf.scalar(4);

    const grads = tf.grads((a, b) => tf.atan2(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [4 * 2 / 29]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(db, [4 * -5 / 29]);
  });

  it('gradient: Tensor1D', () => {
    const a = tf.tensor1d([1, 2, 3]);
    const b = tf.tensor1d([3, 4, 5]);
    const dy = tf.tensor1d([1, 10, 20]);

    const grads = tf.grads((a, b) => tf.atan2(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(da, [1 * 3 / 10, 10 * 4 / 20, 20 * 5 / 34]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(db, [-1 * 1 / 10, -10 * 2 / 20, -20 * 3 / 34]);
  });

  it('gradient: Tensor2D', () => {
    const a = tf.tensor2d([3, 1, 2, 3], [2, 2]);
    const b = tf.tensor2d([1, 3, 4, 5], [2, 2]);
    const dy = tf.tensor2d([1, 10, 15, 20], [2, 2]);

    const grads = tf.grads((a, b) => tf.atan2(a, b));
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
    const a = tf.scalar(2);
    const b = tf.tensor1d([3, 4, 5]);
    const dy = tf.tensor1d([6, 7, 8]);

    const grads = tf.grads((a, b) => tf.atan2(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [6 * 3 / 13 + 7 * 4 / 20 + 8 * 5 / 29]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(db, [-6 * 2 / 13, -7 * 2 / 20, -8 * 2 / 29]);
  });

  it('gradient: Tensor2D / scalar', () => {
    const a = tf.tensor2d([[2, 3], [4, 5]], [2, 2]);
    const b = tf.scalar(2);
    const dy = tf.tensor2d([[6, 7], [8, 9]], [2, 2]);

    const grads = tf.grads((a, b) => tf.atan2(a, b));
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
    const a = tf.tensor2d([3, 4], [2, 1]);
    const b = tf.tensor2d([[2, 3], [4, 5]], [2, 2]);
    const dy = tf.tensor2d([[6, 7], [8, 9]], [2, 2]);

    const grads = tf.grads((a, b) => tf.atan2(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(da, [6 * 2 / 13 + 7 * 3 / 18, 8 * 4 / 32 + 9 * 5 / 41]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(db, [-6 * 3 / 13, -7 * 3 / 18, -8 * 4 / 32, -9 * 4 / 41]);
  });

  it('throws when passed a as a non-tensor', () => {
    expect(() => tf.atan2({} as tf.Tensor, tf.scalar(1)))
        .toThrowError(/Argument 'a' passed to 'atan2' must be a Tensor/);
  });
  it('throws when passed b as a non-tensor', () => {
    expect(() => tf.atan2(tf.scalar(1), {} as tf.Tensor))
        .toThrowError(/Argument 'b' passed to 'atan2' must be a Tensor/);
  });

  it('accepts a tensor-like object', () => {
    const a = [[1, 2, 3], [4, 5, 6]];
    const c = 2;

    const r = tf.atan2(a, c);
    const expected = [];

    for (let i = 0; i < 6; i++) {
      expected[i] = Math.atan2(i + 1, 2);
    }
    expectArraysClose(r, expected);
  });
});

describeWithFlags('div', ALL_ENVS, () => {
  it('basic', () => {
    const a = tf.tensor1d([0, 1, -2, -4, 4, -4]);
    const b = tf.tensor1d([0.15, 0.2, 0.25, 0.5, 0.7, 1.2]);
    const result = tf.div(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(
        result, [0, 5.0, -8.0, -8.0, 5.714285850524902, -3.3333332538604736]);
  });

  it('floored internally', () => {
    const a = tf.tensor1d([10, 20, -20, -40], 'int32');
    const b = tf.tensor1d([10, 12, 8, 5], 'int32');
    const result = tf.div(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [1, 1, -3, -8]);
  });

  it('floorDiv', () => {
    const a = tf.tensor1d([10, 20, -20, -40], 'int32');
    const b = tf.tensor1d([10, 12, 8, 5], 'int32');
    const result = tf.floorDiv(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [1, 1, -3, -8]);
  });
});
