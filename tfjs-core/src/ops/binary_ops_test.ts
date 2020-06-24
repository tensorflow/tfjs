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
import {expectArraysClose, expectArraysEqual} from '../test_util';

describeWithFlags('prelu', ALL_ENVS, () => {
  it('basic', async () => {
    const x = tf.tensor1d([0, 1, -2, -4]);
    const a = tf.tensor1d([0.15, 0.2, 0.25, 0.15]);
    const result = tf.prelu(x, a);

    expect(result.shape).toEqual(x.shape);
    expectArraysClose(await result.data(), [0, 1, -0.5, -0.6]);
  });

  it('basic TensorLike', async () => {
    const x = [0, 1, -2, -4];
    const a = [0.15, 0.2, 0.25, 0.15];
    const result = tf.prelu(x, a);

    expect(result.shape).toEqual([4]);
    expectArraysClose(await result.data(), [0, 1, -0.5, -0.6]);
  });

  it('basic TensorLike chained', async () => {
    const x = tf.tensor1d([0, 1, -2, -4]);
    const a = [0.15, 0.2, 0.25, 0.15];
    const result = x.prelu(a);

    expect(result.shape).toEqual(x.shape);
    expectArraysClose(await result.data(), [0, 1, -0.5, -0.6]);
  });

  it('derivative', async () => {
    const x = tf.tensor1d([0.5, 3, -0.1, -4]);
    const a = tf.tensor1d([0.2, 0.4, 0.25, 0.15]);
    const dy = tf.tensor1d([1, 1, 1, 1]);

    const dx = tf.grad(x => tf.prelu(x, a))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expect(dx.dtype).toEqual('float32');
    expectArraysClose(await dx.data(), [1, 1, 0.25, 0.15]);
  });

  it('gradient with clones', async () => {
    const x = tf.tensor1d([0.5, 3, -0.1, -4]);
    const a = tf.tensor1d([0.2, 0.4, 0.25, 0.15]);
    const dx = tf.grad(x => tf.prelu(x.clone(), a).clone())(x);

    expect(dx.shape).toEqual(x.shape);
    expect(dx.dtype).toEqual('float32');
    expectArraysClose(await dx.data(), [1, 1, 0.25, 0.15]);
  });

  it('derivative where alpha got broadcasted', async () => {
    const x = tf.tensor2d([[0.5, 3, -0.1, -4]]);
    const a = tf.tensor2d([[0.2]]);
    const dy = tf.tensor2d([[1, 1, 1, 1]]);

    const da = tf.grad(a => tf.prelu(x, a))(a, dy);
    expect(da.shape).toEqual(a.shape);
    expectArraysClose(await da.data(), [-4.1]);
  });

  it('throws when passed x as a non-tensor', () => {
    expect(() => tf.prelu({} as tf.Tensor, tf.scalar(1)))
        .toThrowError(/Argument 'x' passed to 'prelu' must be a Tensor/);
  });
  it('throws when passed alpha as a non-tensor', () => {
    expect(() => tf.prelu(tf.scalar(1), {} as tf.Tensor))
        .toThrowError(/Argument 'alpha' passed to 'prelu' must be a Tensor/);
  });

  it('throws for string tensor', () => {
    expect(() => tf.prelu(['a'], 0.1))
        .toThrowError(/Argument 'x' passed to 'prelu' must be numeric tensor/);
  });
});

describeWithFlags('maximum', ALL_ENVS, () => {
  it('float32 and float32', async () => {
    const a = tf.tensor1d([0.5, 3, -0.1, -4]);
    const b = tf.tensor1d([0.2, 0.4, 0.25, 0.15]);
    const result = tf.maximum(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(await result.data(), [0.5, 3, 0.25, 0.15]);
  });

  it('TensorLike', async () => {
    const a = [0.5, 3, -0.1, -4];
    const b = [0.2, 0.4, 0.25, 0.15];
    const result = tf.maximum(a, b);

    expect(result.shape).toEqual([4]);
    expectArraysClose(await result.data(), [0.5, 3, 0.25, 0.15]);
  });

  it('TensorLike chained', async () => {
    const a = tf.tensor1d([0.5, 3, -0.1, -4]);
    const b = [0.2, 0.4, 0.25, 0.15];
    const result = a.maximum(b);

    expect(result.shape).toEqual([4]);
    expectArraysClose(await result.data(), [0.5, 3, 0.25, 0.15]);
  });

  it('int32 and int32', async () => {
    const a = tf.tensor1d([1, 5, 2, 3], 'int32');
    const b = tf.tensor1d([2, 3, 1, 4], 'int32');
    const result = tf.maximum(a, b);

    expect(result.shape).toEqual(a.shape);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(await result.data(), [2, 5, 2, 4]);
  });

  it('bool and bool', async () => {
    const a = tf.tensor1d([true, false, false, true], 'bool');
    const b = tf.tensor1d([false, false, true, true], 'bool');
    const result = tf.maximum(a, b);

    expect(result.shape).toEqual(a.shape);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(await result.data(), [1, 0, 1, 1]);
  });

  it('upcasts when dtypes dont match', async () => {
    const a = tf.tensor1d([1, 0, 0, 1], 'float32');
    const b = tf.tensor1d([0, 0, 1, 1], 'int32');
    const res = tf.maximum(a, b);
    expect(res.shape).toEqual(a.shape);
    expect(res.dtype).toBe('float32');
    expectArraysEqual(await res.data(), [1, 0, 1, 1]);
  });

  it('propagates NaN', async () => {
    const a = tf.tensor1d([0.5, -0.1, NaN]);
    const b = tf.tensor1d([0.2, 0.3, 0.25]);
    const result = tf.maximum(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(await result.data(), [0.5, 0.3, NaN]);
  });

  it('broadcasts Tensor1D and scalar', async () => {
    const a = tf.tensor1d([0.5, 3, -0.1, -4]);
    const b = tf.scalar(0.6);
    const result = tf.maximum(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(await result.data(), [0.6, 3, 0.6, 0.6]);
  });

  it('broadcasts scalar and Tensor1D', async () => {
    const a = tf.scalar(0.6);
    const b = tf.tensor1d([0.5, 3, -0.1, -4]);
    const result = tf.maximum(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(await result.data(), [0.6, 3, 0.6, 0.6]);
  });

  it('broadcasts Tensor1D and Tensor2D', async () => {
    const a = tf.tensor1d([0.5, 0.3]);
    const b = tf.tensor2d([0.2, 0.4, 0.6, 0.15], [2, 2]);
    const result = tf.maximum(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(await result.data(), [0.5, 0.4, 0.6, 0.3]);
  });

  it('broadcasts 2x1 Tensor2D and 2x2 Tensor2D', async () => {
    const a = tf.tensor2d([0.5, 0.3], [2, 1]);
    const b = tf.tensor2d([0.2, 0.4, 0.6, 0.15], [2, 2]);
    const result = tf.maximum(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(await result.data(), [0.5, 0.5, 0.6, 0.3]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(5.2);
    const b = tf.scalar(0.6);
    const dy = tf.scalar(3);

    const grads = tf.grads((a, b) => tf.maximum(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(await da.data(), [3 * 1]);
    expectArraysClose(await db.data(), [3 * 0]);
  });

  it('gradient with clones', async () => {
    const a = tf.scalar(5.2);
    const b = tf.scalar(0.6);
    const dy = tf.scalar(3);

    const grads = tf.grads((a, b) => tf.maximum(a.clone(), b.clone()).clone());
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(await da.data(), [3 * 1]);
    expectArraysClose(await db.data(), [3 * 0]);
  });

  it('gradients: Tensor1D', async () => {
    const a = tf.tensor1d([1.1, 2.6, 3, 5.9]);
    const b = tf.tensor1d([1.0, 2.7, 3, 5.8]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const grads = tf.grads((a, b) => tf.maximum(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(await da.data(), [1 * 1, 2 * 0, 3 * 1, 4 * 1]);
    expectArraysClose(await db.data(), [1 * 0, 2 * 1, 3 * 0, 4 * 0]);
  });

  it('gradients: Tensor2D', async () => {
    const a = tf.tensor2d([0.5, 0.3, 0.7, 0.9], [2, 2]);
    const b = tf.tensor2d([0.2, 0.4, 0.7, 0.15], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const grads = tf.grads((a, b) => tf.maximum(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(await da.data(), [1 * 1, 2 * 0, 3 * 1, 4 * 1]);
    expectArraysClose(await db.data(), [1 * 0, 2 * 1, 3 * 0, 4 * 0]);
  });

  it('throws when passed a as a non-tensor', () => {
    expect(() => tf.maximum({} as tf.Tensor, tf.scalar(1)))
        .toThrowError(/Argument 'a' passed to 'maximum' must be a Tensor/);
  });
  it('throws when passed b as a non-tensor', () => {
    expect(() => tf.maximum(tf.scalar(1), {} as tf.Tensor))
        .toThrowError(/Argument 'b' passed to 'maximum' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const a = [[0.5, 3], [-0.1, -4]];
    const b = [[0.2, 0.4], [0.25, 0.15]];
    const result = tf.maximum(a, b);

    expect(result.shape).toEqual([2, 2]);
    expectArraysClose(await result.data(), [0.5, 3, 0.25, 0.15]);
  });

  it('throws for string tensor', () => {
    expect(() => tf.maximum('q', 3))
        .toThrowError(
            /Argument 'a' passed to 'maximum' must be numeric tensor/);

    expect(() => tf.maximum(3, 'q'))
        .toThrowError(
            /Argument 'b' passed to 'maximum' must be numeric tensor/);
  });
});

describeWithFlags('squaredDifference', ALL_ENVS, () => {
  it('float32 and float32', async () => {
    const a = tf.tensor1d([0.5, 3, -0.1, -4]);
    const b = tf.tensor1d([0.2, 0.4, 0.25, 0.15]);
    const result = tf.squaredDifference(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(await result.data(), [
      Math.pow(0.5 - 0.2, 2), Math.pow(3 - 0.4, 2), Math.pow(-0.1 - 0.25, 2),
      Math.pow(-4 - 0.15, 2)
    ]);
  });

  it('TensorLike', async () => {
    const a = [0.5, 3, -0.1, -4];
    const b = [0.2, 0.4, 0.25, 0.15];
    const result = tf.squaredDifference(a, b);

    expect(result.shape).toEqual([4]);
    expectArraysClose(await result.data(), [
      Math.pow(0.5 - 0.2, 2), Math.pow(3 - 0.4, 2), Math.pow(-0.1 - 0.25, 2),
      Math.pow(-4 - 0.15, 2)
    ]);
  });

  it('TensorLike chained', async () => {
    const a = tf.tensor1d([0.5, 3, -0.1, -4]);
    const b = [0.2, 0.4, 0.25, 0.15];
    const result = a.squaredDifference(b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(await result.data(), [
      Math.pow(0.5 - 0.2, 2), Math.pow(3 - 0.4, 2), Math.pow(-0.1 - 0.25, 2),
      Math.pow(-4 - 0.15, 2)
    ]);
  });

  it('int32 and int32', async () => {
    const a = tf.tensor1d([1, 5, 2, 3], 'int32');
    const b = tf.tensor1d([2, 3, 1, 4], 'int32');
    const result = tf.squaredDifference(a, b);

    expect(result.shape).toEqual(a.shape);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(await result.data(), [
      Math.pow(1 - 2, 2), Math.pow(5 - 3, 2), Math.pow(2 - 1, 2),
      Math.pow(3 - 4, 2)
    ]);
  });

  it('upcasts when dtypes dont match', async () => {
    let res =
        tf.squaredDifference(tf.scalar(5, 'int32'), tf.scalar(2, 'float32'));
    expect(res.dtype).toBe('float32');
    expectArraysClose(await res.data(), [9]);

    res = tf.squaredDifference(tf.scalar(5, 'int32'), tf.scalar(true, 'bool'));
    expect(res.dtype).toBe('int32');
    expectArraysClose(await res.data(), [16]);

    res = tf.squaredDifference(tf.scalar(5, 'int32'), tf.scalar(false, 'bool'));
    expect(res.dtype).toBe('int32');
    expectArraysClose(await res.data(), [25]);
  });

  it('propagates NaN', async () => {
    const a = tf.tensor1d([0.5, -0.1, NaN]);
    const b = tf.tensor1d([0.2, 0.3, 0.25]);
    const result = tf.squaredDifference(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(
        await result.data(),
        [Math.pow(0.5 - 0.2, 2), Math.pow(-0.1 - 0.3, 2), NaN]);
  });

  it('broadcasts Tensor1D and scalar', async () => {
    const a = tf.tensor1d([0.5, 3, -0.1, -4]);
    const b = tf.scalar(0.6);
    const result = tf.squaredDifference(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(await result.data(), [
      Math.pow(0.5 - 0.6, 2), Math.pow(3 - 0.6, 2), Math.pow(-0.1 - 0.6, 2),
      Math.pow(-4 - 0.6, 2)
    ]);
  });

  it('broadcasts scalar and Tensor1D', async () => {
    const a = tf.scalar(0.6);
    const b = tf.tensor1d([0.5, 3, -0.1, -4]);
    const result = tf.squaredDifference(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(await result.data(), [
      Math.pow(0.6 - 0.5, 2), Math.pow(0.6 - 3, 2), Math.pow(0.6 - (-0.1), 2),
      Math.pow(0.6 - (-4), 2)
    ]);
  });

  it('broadcasts Tensor1D and Tensor2D', async () => {
    const a = tf.tensor1d([0.5, 0.3]);
    const b = tf.tensor2d([0.2, 0.4, 0.6, 0.15], [2, 2]);
    const result = tf.squaredDifference(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(await result.data(), [
      Math.pow(0.5 - 0.2, 2), Math.pow(0.3 - 0.4, 2), Math.pow(0.5 - 0.6, 2),
      Math.pow(0.3 - 0.15, 2)
    ]);
  });

  it('broadcasts 2x1 Tensor2D and 2x2 Tensor2D', async () => {
    const a = tf.tensor2d([0.5, 0.3], [2, 1]);
    const b = tf.tensor2d([0.2, 0.4, 0.6, 0.15], [2, 2]);
    const result = tf.squaredDifference(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(await result.data(), [
      Math.pow(0.5 - 0.2, 2), Math.pow(0.5 - 0.4, 2), Math.pow(0.3 - 0.6, 2),
      Math.pow(0.3 - 0.15, 2)
    ]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(5.2);
    const b = tf.scalar(0.6);
    const dy = tf.scalar(3);

    const grads = tf.grads((a, b) => tf.squaredDifference(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(await da.data(), [3 * 2 * (5.2 - 0.6)]);
    expectArraysClose(await db.data(), [3 * 2 * (0.6 - 5.2)]);
  });

  it('gradient with clones', async () => {
    const a = tf.scalar(5.2);
    const b = tf.scalar(0.6);
    const dy = tf.scalar(3);

    const grads =
        tf.grads((a, b) => tf.squaredDifference(a.clone(), b.clone()).clone());
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(await da.data(), [3 * 2 * (5.2 - 0.6)]);
    expectArraysClose(await db.data(), [3 * 2 * (0.6 - 5.2)]);
  });

  it('gradients: Tensor1D', async () => {
    const a = tf.tensor1d([1.1, 2.6, 3, 5.9]);
    const b = tf.tensor1d([1.0, 2.7, 3, 5.8]);
    const dy = tf.tensor1d([1, 2, 3, 1]);

    const grads = tf.grads((a, b) => tf.squaredDifference(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(await da.data(), [
      1 * 2 * (1.1 - 1.0), 2 * 2 * (2.6 - 2.7), 3 * 2 * (3 - 3),
      1 * 2 * (5.9 - 5.8)
    ]);
    expectArraysClose(await db.data(), [
      1 * 2 * (1.0 - 1.1), 2 * 2 * (2.7 - 2.6), 3 * 2 * (3 - 3),
      1 * 2 * (5.8 - 5.9)
    ]);
  });

  it('gradients: Tensor2D', async () => {
    const a = tf.tensor2d([0.5, 0.3, 0.7, 0.9], [2, 2]);
    const b = tf.tensor2d([0.2, 0.4, 0.7, 0.15], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const grads = tf.grads((a, b) => tf.squaredDifference(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(await da.data(), [
      1 * 2 * (0.5 - 0.2), 2 * 2 * (0.3 - 0.4), 3 * 2 * (0.7 - 0.7),
      4 * 2 * (0.9 - 0.15)
    ]);
    expectArraysClose(await db.data(), [
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

  it('accepts a tensor-like object', async () => {
    const a = [[0.5, 3], [-0.1, -4]];
    const b = 0.6;
    const result = tf.squaredDifference(a, b);

    expect(result.shape).toEqual([2, 2]);
    expectArraysClose(await result.data(), [
      Math.pow(0.5 - 0.6, 2), Math.pow(3 - 0.6, 2), Math.pow(-0.1 - 0.6, 2),
      Math.pow(-4 - 0.6, 2)
    ]);
  });

  it('throws for string tensor', () => {
    expect(() => tf.squaredDifference('q', 3))
        .toThrowError(
            /Argument 'a' passed to 'squaredDifference' must be numeric/);

    expect(() => tf.squaredDifference(3, 'q'))
        .toThrowError(
            /Argument 'b' passed to 'squaredDifference' must be numeric/);
  });
});

describeWithFlags('minimum', ALL_ENVS, () => {
  it('float32 and float32', async () => {
    const a = tf.tensor1d([0.5, 3, -0.1, -4]);
    const b = tf.tensor1d([0.2, 0.4, 0.25, 0.15]);
    const result = tf.minimum(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(await result.data(), [0.2, 0.4, -0.1, -4]);
  });

  it('TensorLike', async () => {
    const a = [0.5, 3, -0.1, -4];
    const b = [0.2, 0.4, 0.25, 0.15];
    const result = tf.minimum(a, b);

    expect(result.shape).toEqual([4]);
    expectArraysClose(await result.data(), [0.2, 0.4, -0.1, -4]);
  });

  it('TensorLike chained', async () => {
    const a = tf.tensor1d([0.5, 3, -0.1, -4]);
    const b = [0.2, 0.4, 0.25, 0.15];
    const result = a.minimum(b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(await result.data(), [0.2, 0.4, -0.1, -4]);
  });

  it('int32 and int32', async () => {
    const a = tf.tensor1d([1, 5, 2, 3], 'int32');
    const b = tf.tensor1d([2, 3, 1, 4], 'int32');
    const result = tf.minimum(a, b);

    expect(result.shape).toEqual(a.shape);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(await result.data(), [1, 3, 1, 3]);
  });

  it('bool and bool', async () => {
    const a = tf.tensor1d([true, false, false, true], 'bool');
    const b = tf.tensor1d([false, false, true, true], 'bool');
    const result = tf.minimum(a, b);

    expect(result.shape).toEqual(a.shape);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(await result.data(), [0, 0, 0, 1]);
  });

  it('upcasts when dtypes dont match', async () => {
    const a = tf.tensor1d([1, 0, 0, 1], 'float32');
    const b = tf.tensor1d([0, 0, 1, 1], 'int32');
    const res = tf.minimum(a, b);
    expect(res.shape).toEqual(a.shape);
    expect(res.dtype).toBe('float32');
    expectArraysEqual(await res.data(), [0, 0, 0, 1]);
  });

  it('propagates NaN', async () => {
    const a = tf.tensor1d([0.5, -0.1, NaN]);
    const b = tf.tensor1d([0.2, 0.3, 0.25]);
    const result = tf.minimum(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(await result.data(), [0.2, -0.1, NaN]);
  });

  it('broadcasts Tensor1D and scalar', async () => {
    const a = tf.tensor1d([0.5, 3, -0.1, -4]);
    const b = tf.scalar(0.6);
    const result = tf.minimum(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(await result.data(), [0.5, 0.6, -0.1, -4]);
  });

  it('broadcasts scalar and Tensor1D', async () => {
    const a = tf.scalar(0.6);
    const b = tf.tensor1d([0.5, 3, -0.1, -4]);
    const result = tf.minimum(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(await result.data(), [0.5, 0.6, -0.1, -4]);
  });

  it('broadcasts Tensor1D and Tensor2D', async () => {
    const a = tf.tensor1d([0.5, 0.3]);
    const b = tf.tensor2d([0.2, 0.4, 0.6, 0.15], [2, 2]);
    const result = tf.minimum(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(await result.data(), [0.2, 0.3, 0.5, 0.15]);
  });

  it('broadcasts 2x1 Tensor2D and 2x2 Tensor2D', async () => {
    const a = tf.tensor2d([0.5, 0.3], [2, 1]);
    const b = tf.tensor2d([0.2, 0.4, 0.6, 0.15], [2, 2]);
    const result = tf.minimum(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(await result.data(), [0.2, 0.4, 0.3, 0.15]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(5.2);
    const b = tf.scalar(0.6);
    const dy = tf.scalar(3);

    const grads = tf.grads((a, b) => tf.minimum(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(await da.data(), [3 * 0]);
    expectArraysClose(await db.data(), [3 * 1]);
  });

  it('gradient with clones', async () => {
    const a = tf.scalar(5.2);
    const b = tf.scalar(0.6);
    const dy = tf.scalar(3);

    const grads = tf.grads((a, b) => tf.minimum(a.clone(), b.clone()).clone());
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(await da.data(), [3 * 0]);
    expectArraysClose(await db.data(), [3 * 1]);
  });

  it('gradients: Tensor1D', async () => {
    const a = tf.tensor1d([1.1, 2.6, 3, 5.9]);
    const b = tf.tensor1d([1.0, 2.7, 3, 5.8]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const grads = tf.grads((a, b) => tf.minimum(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(await da.data(), [1 * 0, 2 * 1, 3 * 1, 4 * 0]);
    expectArraysClose(await db.data(), [1 * 1, 2 * 0, 3 * 0, 4 * 1]);
  });

  it('gradients: Tensor2D', async () => {
    const a = tf.tensor2d([0.5, 0.3, 0.7, 0.9], [2, 2]);
    const b = tf.tensor2d([0.2, 0.4, 0.7, 0.15], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const grads = tf.grads((a, b) => tf.minimum(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(await da.data(), [1 * 0, 2 * 1, 3 * 1, 4 * 0]);
    expectArraysClose(await db.data(), [1 * 1, 2 * 0, 3 * 0, 4 * 1]);
  });

  it('throws when passed a as a non-tensor', () => {
    expect(() => tf.minimum({} as tf.Tensor, tf.scalar(1)))
        .toThrowError(/Argument 'a' passed to 'minimum' must be a Tensor/);
  });
  it('throws when passed b as a non-tensor', () => {
    expect(() => tf.minimum(tf.scalar(1), {} as tf.Tensor))
        .toThrowError(/Argument 'b' passed to 'minimum' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const a = [[0.5, 3], [-0.1, -4]];
    const b = [[0.2, 0.4], [0.25, 0.15]];
    const result = tf.minimum(a, b);

    expect(result.shape).toEqual([2, 2]);
    expectArraysClose(await result.data(), [0.2, 0.4, -0.1, -4]);
  });

  it('throws for string tensor', () => {
    expect(() => tf.minimum('q', 3))
        .toThrowError(/Argument 'a' passed to 'minimum' must be numeric/);

    expect(() => tf.minimum(3, 'q'))
        .toThrowError(/Argument 'b' passed to 'minimum' must be numeric/);
  });
});

describeWithFlags('mod', ALL_ENVS, () => {
  it('float32 and float32', async () => {
    const a = tf.tensor1d([0.5, 3, -0.1, -4]);
    const b = tf.tensor1d([0.2, 0.4, 0.25, 0.15]);
    const result = tf.mod(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(await result.data(), [0.1, 0.2, 0.15, 0.05]);
  });

  it('TensorLike', async () => {
    const a = [0.5, 3, -0.1, -4];
    const b = [0.2, 0.4, 0.25, 0.15];
    const result = tf.mod(a, b);

    expect(result.shape).toEqual([4]);
    expectArraysClose(await result.data(), [0.1, 0.2, 0.15, 0.05]);
  });

  it('TensorLike chained', async () => {
    const a = tf.tensor1d([0.5, 3, -0.1, -4]);
    const b = [0.2, 0.4, 0.25, 0.15];
    const result = a.mod(b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(await result.data(), [0.1, 0.2, 0.15, 0.05]);
  });

  it('int32 and int32', async () => {
    const a = tf.tensor1d([1, 5, 2, 3], 'int32');
    const b = tf.tensor1d([2, 3, 1, 4], 'int32');
    const result = tf.mod(a, b);

    expect(result.shape).toEqual(a.shape);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(await result.data(), [1, 2, 0, 3]);
  });

  it('upcasts when dtypes dont match', async () => {
    let res = tf.mod(tf.scalar(5, 'int32'), tf.scalar(2, 'float32'));
    expect(res.dtype).toBe('float32');
    expectArraysClose(await res.data(), [1]);

    res = tf.mod(tf.scalar(5, 'int32'), tf.scalar(true, 'bool'));
    expect(res.dtype).toBe('int32');
    expectArraysClose(await res.data(), [0]);
  });

  it('propagates NaN', async () => {
    const a = tf.tensor1d([5, -1, NaN]);
    const b = tf.tensor1d([2, 3, 0.25]);
    const result = tf.mod(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(await result.data(), [1, 2, NaN]);
  });

  it('broadcasts Tensor1D and scalar', async () => {
    const a = tf.tensor1d([0.5, 2.5, -0.1, -4], 'float32');
    const b = tf.scalar(0.6);
    const result = tf.mod(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(await result.data(), [0.5, 0.1, 0.5, 0.2]);
  });

  it('broadcasts scalar and Tensor1D', async () => {
    // TODO(manraj): Fix for case fmod(0.6, -0.1)
    const a = tf.scalar(2);
    const b = tf.tensor1d([3, 3, -1, -4]);
    const result = tf.mod(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(await result.data(), [2, 2, 0, -2]);
  });

  it('broadcasts Tensor1D and Tensor2D', async () => {
    const a = tf.tensor1d([0.5, 0.3]);
    const b = tf.tensor2d([0.2, 0.4, 0.6, 0.15], [2, 2]);
    const result = tf.mod(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(await result.data(), [0.1, 0.3, 0.5, 0.0]);
  });

  it('broadcasts 2x1 Tensor2D and 2x2 Tensor2D', async () => {
    const a = tf.tensor2d([0.5, 0.3], [2, 1]);
    const b = tf.tensor2d([0.2, 0.4, 0.6, 0.15], [2, 2]);
    const result = tf.mod(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(await result.data(), [0.1, 0.1, 0.3, 0.0]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(5.2);
    const b = tf.scalar(0.6);
    const dy = tf.scalar(3);

    const grads = tf.grads((a, b) => tf.mod(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(await da.data(), [3]);
    expectArraysClose(await db.data(), [3 * -1 * Math.floor(5.2 / 0.6)]);
  });

  it('gradient with clones', async () => {
    const a = tf.scalar(5.2);
    const b = tf.scalar(0.6);
    const dy = tf.scalar(3);

    const grads = tf.grads((a, b) => tf.mod(a.clone(), b.clone()).clone());
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(await da.data(), [3]);
    expectArraysClose(await db.data(), [3 * -1 * Math.floor(5.2 / 0.6)]);
  });

  it('gradients: Tensor1D', async () => {
    const a = tf.tensor1d([1.1, 2.6, 3, 5.9]);
    const b = tf.tensor1d([1.0, 2.7, 3, 5.8]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const grads = tf.grads((a, b) => tf.mod(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(await da.data(), [1 * 1, 2 * 1, 3 * 1, 4 * 1]);
    expectArraysClose(await db.data(), [
      1 * -1 * Math.floor(1.1 / 1.0), 2 * -1 * Math.floor(2.6 / 2.7),
      3 * -1 * Math.floor(3 / 3), 4 * -1 * Math.floor(5.9 / 5.8)
    ]);
  });

  it('gradients: Tensor2D', async () => {
    const a = tf.tensor2d([0.5, 0.3, 0.7, 0.91], [2, 2]);
    const b = tf.tensor2d([0.2, 0.4, 0.7, 0.15], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const grads = tf.grads((a, b) => tf.mod(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(await da.data(), [1 * 1, 2 * 1, 3 * 1, 4 * 1]);
    expectArraysClose(await db.data(), [
      1 * -1 * Math.floor(0.5 / 0.2), 2 * -1 * Math.floor(0.3 / 0.4),
      3 * -1 * Math.floor(0.7 / 0.7), 4 * -1 * Math.floor(0.91 / 0.15)
    ]);
  });

  it('gradients: broadcasts scalar and Tensor1D', async () => {
    const a = tf.scalar(0.7);
    const b = tf.tensor1d([0.2, 0.3, 0.4, 0.5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const grads = tf.grads((a, b) => tf.mod(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(await da.data(), [1 + 2 + 3 + 4]);
    expectArraysClose(await db.data(), [
      1 * -1 * Math.floor(0.7 / 0.2), 2 * -1 * Math.floor(0.7 / 0.3),
      3 * -1 * Math.floor(0.7 / 0.4), 4 * -1 * Math.floor(0.7 / 0.5)
    ]);
  });

  it('broadcasts Tensor1D and Tensor2D', async () => {
    const a = tf.tensor1d([0.5, 0.3]);
    const b = tf.tensor2d([0.2, 0.4, 0.7, 0.15], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const grads = tf.grads((a, b) => tf.mod(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(await da.data(), [1 * 1 + 3 * 1, 2 * 1 + 4 * 1]);
    expectArraysClose(await db.data(), [
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

  it('accepts a tensor-like object', async () => {
    const a = [[0.5, 3], [-0.1, -4]];
    const b = [[0.2, 0.4], [0.25, 0.15]];
    const result = tf.mod(a, b);

    expect(result.shape).toEqual([2, 2]);
    expectArraysClose(await result.data(), [0.1, 0.2, 0.15, 0.05]);
  });

  it('throws for string tensor', () => {
    expect(() => tf.mod('q', 3))
        .toThrowError(/Argument 'a' passed to 'mod' must be numeric/);

    expect(() => tf.mod(3, 'q'))
        .toThrowError(/Argument 'b' passed to 'mod' must be numeric/);
  });
});

describeWithFlags('atan2', ALL_ENVS, () => {
  it('same shape', async () => {
    const aValues = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    const bValues = [1.0, 2.5, 3.5, 4.5, 2.0, 5.0];

    const a = tf.tensor2d(aValues, [2, 3]);
    const c = tf.tensor2d(bValues, [2, 3]);

    const r = tf.atan2(a, c);
    const expected = [];

    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.atan2(aValues[i], bValues[i]);
    }
    expectArraysClose(await r.data(), expected);
  });

  it('uses chaining', async () => {
    const aValues = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    const bValues = [1.0, 2.5, 3.5, 4.5, 2.0, 5.0];

    const a = tf.tensor2d(aValues, [2, 3]);
    const b = tf.tensor2d(bValues, [2, 3]);

    const r = a.atan2(b);
    const expected = [];

    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.atan2(aValues[i], bValues[i]);
    }
    expectArraysClose(await r.data(), expected);
  });

  it('propagates NaNs', async () => {
    const a = tf.tensor2d([1.0, 2.0], [2, 1]);
    const c = tf.tensor2d([3.0, NaN], [2, 1]);

    const r = tf.atan2(a, c);

    expectArraysClose(await r.data(), [Math.atan2(1.0, 3.0), NaN]);
  });

  it('broadcasting same rank Tensors different shape', async () => {
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
    expectArraysClose(await result.data(), expected);
  });

  it('throws when passed tensors of different shapes', () => {
    const a = tf.tensor2d([1, 2, -3, -4, 5, 6], [2, 3]);
    const b = tf.tensor2d([5, 3, 4, -7], [2, 2]);

    expect(() => tf.atan2(a, b)).toThrowError();
    expect(() => tf.atan2(b, a)).toThrowError();
  });

  it('upcasts when dtypes dont match', async () => {
    const aValues = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    const bValues = [1, 2, 3, 4, 2, 5];

    const a = tf.tensor2d(aValues, [2, 3], 'float32');
    const c = tf.tensor2d(bValues, [2, 3], 'int32');

    const r = tf.atan2(a, c);
    const expected = [];

    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.atan2(aValues[i], bValues[i]);
    }
    expect(r.shape).toEqual([2, 3]);
    expect(r.dtype).toBe('float32');
    expectArraysClose(await r.data(), expected);
  });

  it('atan2 of scalar and array propagates NaNs', async () => {
    const c = tf.scalar(NaN);
    const a = tf.tensor2d([1, 2, 3], [1, 3]);

    const r = tf.atan2(c, a);

    expectArraysEqual(await r.data(), [NaN, NaN, NaN]);
  });

  it('atan2 of scalar and array', async () => {
    const aValues = [1, 2, 3, 4, 5, 6];

    const a = tf.tensor2d(aValues, [2, 3]);
    const c = tf.scalar(2);

    const r = tf.atan2(a, c);
    const expected = [];

    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.atan2(aValues[i], 2);
    }
    expectArraysClose(await r.data(), expected);
  });

  it('gradient: Scalar', async () => {
    const a = tf.scalar(5);
    const b = tf.scalar(2);
    const dy = tf.scalar(4);

    const grads = tf.grads((a, b) => tf.atan2(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [4 * 2 / 29]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await db.data(), [4 * -5 / 29]);
  });

  it('gradient with clones', async () => {
    const a = tf.scalar(5);
    const b = tf.scalar(2);
    const dy = tf.scalar(4);

    const grads = tf.grads((a, b) => tf.atan2(a.clone(), b.clone()).clone());
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [4 * 2 / 29]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await db.data(), [4 * -5 / 29]);
  });

  it('gradient: Tensor1D', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    const b = tf.tensor1d([3, 4, 5]);
    const dy = tf.tensor1d([1, 10, 20]);

    const grads = tf.grads((a, b) => tf.atan2(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [1 * 3 / 10, 10 * 4 / 20, 20 * 5 / 34]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(
        await db.data(), [-1 * 1 / 10, -10 * 2 / 20, -20 * 3 / 34]);
  });

  it('gradient: Tensor2D', async () => {
    const a = tf.tensor2d([3, 1, 2, 3], [2, 2]);
    const b = tf.tensor2d([1, 3, 4, 5], [2, 2]);
    const dy = tf.tensor2d([1, 10, 15, 20], [2, 2]);

    const grads = tf.grads((a, b) => tf.atan2(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(
        await da.data(), [1 * 1 / 10, 10 * 3 / 10, 15 * 4 / 20, 20 * 5 / 34]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(
        await db.data(),
        [-1 * 3 / 10, -10 * 1 / 10, -15 * 2 / 20, -20 * 3 / 34]);
  });

  it('gradient: scalar / Tensor1D', async () => {
    const a = tf.scalar(2);
    const b = tf.tensor1d([3, 4, 5]);
    const dy = tf.tensor1d([6, 7, 8]);

    const grads = tf.grads((a, b) => tf.atan2(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [6 * 3 / 13 + 7 * 4 / 20 + 8 * 5 / 29]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(await db.data(), [-6 * 2 / 13, -7 * 2 / 20, -8 * 2 / 29]);
  });

  it('gradient: Tensor2D / scalar', async () => {
    const a = tf.tensor2d([[2, 3], [4, 5]], [2, 2]);
    const b = tf.scalar(2);
    const dy = tf.tensor2d([[6, 7], [8, 9]], [2, 2]);

    const grads = tf.grads((a, b) => tf.atan2(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(
        await da.data(), [6 * 2 / 8, 7 * 2 / 13, 8 * 2 / 20, 9 * 2 / 29]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(
        await db.data(),
        [-6 * 2 / 8 + -7 * 3 / 13 + -8 * 4 / 20 + -9 * 5 / 29]);
  });

  it('gradient: Tensor2D / Tensor2D w/ broadcast', async () => {
    const a = tf.tensor2d([3, 4], [2, 1]);
    const b = tf.tensor2d([[2, 3], [4, 5]], [2, 2]);
    const dy = tf.tensor2d([[6, 7], [8, 9]], [2, 2]);

    const grads = tf.grads((a, b) => tf.atan2(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(
        await da.data(), [6 * 2 / 13 + 7 * 3 / 18, 8 * 4 / 32 + 9 * 5 / 41]);

    expect(db.shape).toEqual(b.shape);
    expect(db.dtype).toEqual('float32');
    expectArraysClose(
        await db.data(), [-6 * 3 / 13, -7 * 3 / 18, -8 * 4 / 32, -9 * 4 / 41]);
  });

  it('throws when passed a as a non-tensor', () => {
    expect(() => tf.atan2({} as tf.Tensor, tf.scalar(1)))
        .toThrowError(/Argument 'a' passed to 'atan2' must be a Tensor/);
  });
  it('throws when passed b as a non-tensor', () => {
    expect(() => tf.atan2(tf.scalar(1), {} as tf.Tensor))
        .toThrowError(/Argument 'b' passed to 'atan2' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const a = [[1, 2, 3], [4, 5, 6]];
    const c = 2;

    const r = tf.atan2(a, c);
    const expected = [];

    for (let i = 0; i < 6; i++) {
      expected[i] = Math.atan2(i + 1, 2);
    }
    expectArraysClose(await r.data(), expected);
  });

  it('throws for string tensor', () => {
    expect(() => tf.atan2('q', 3))
        .toThrowError(/Argument 'a' passed to 'atan2' must be numeric/);

    expect(() => tf.atan2(3, 'q'))
        .toThrowError(/Argument 'b' passed to 'atan2' must be numeric/);
  });
});

describeWithFlags('div', ALL_ENVS, () => {
  it('divNoNan divide 0', async () => {
    // Broadcast div a with b.
    const a = tf.tensor1d([2, 4, 6, 8]);
    const b = tf.tensor1d([0, 0, 0, 0]);

    const c = a.divNoNan(b);
    expect(c.shape).toEqual(a.shape);
    expectArraysClose(await c.data(), [0, 0, 0, 0]);
  });

  it('divNoNan divide 0 and non-0', async () => {
    // Broadcast div a with b.
    const a = tf.tensor1d([2, 4, 6, 8]);
    const b = tf.tensor1d([2, 2, 0, 4]);

    const c = a.divNoNan(b);
    expect(c.shape).toEqual(a.shape);
    expectArraysClose(await c.data(), [1, 2, 0, 2]);
  });

  it('divNoNan divide 0 broadcast', async () => {
    // Broadcast div a with b.
    const a = tf.tensor1d([2, 4, 6, 8]);
    const b = tf.scalar(0);

    const c = a.divNoNan(b);
    expect(c.shape).toEqual(a.shape);
    expectArraysClose(await c.data(), [0, 0, 0, 0]);
  });
});

describeWithFlags('floorDiv', ALL_ENVS, () => {
  it('floorDiv', async () => {
    const a = tf.tensor1d([10, 20, -20, -40], 'int32');
    const b = tf.tensor1d([10, 12, 8, 5], 'int32');
    const result = tf.floorDiv(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(await result.data(), [1, 1, -3, -8]);
  });
});
