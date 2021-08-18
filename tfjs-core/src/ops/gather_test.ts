/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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

describeWithFlags('gather', ALL_ENVS, () => {
  it('1D (gather), scalar indices', async () => {
    const t = tf.tensor1d([1, 2, 3]);

    const t2 = tf.gather(t, tf.scalar(1, 'int32'), 0);

    expect(t2.shape).toEqual([]);
    expectArraysClose(await t2.data(), [2]);
  });

  it('1D (gather), 1D indices', async () => {
    const t = tf.tensor1d([1, 2, 3]);

    const t2 = tf.gather(t, tf.tensor1d([0, 2, 0, 1], 'int32'), 0);

    expect(t2.shape).toEqual([4]);
    expectArraysClose(await t2.data(), [1, 3, 1, 2]);
  });

  it('1D (gather), 2D indices', async () => {
    const t = tf.tensor1d([1, 2, 3]);

    const t2 = tf.gather(t, tf.tensor2d([0, 2, 0, 1], [1, 4], 'int32'), 0);

    expect(t2.shape).toEqual([1, 4]);
    expectArraysClose(await t2.data(), [1, 3, 1, 2]);
  });

  it('2D (gather), scalar indices', async () => {
    const t = tf.tensor2d([1, 11, 2, 22], [2, 2]);
    let t2 = tf.gather(t, tf.scalar(1, 'int32'), 0);
    expect(t2.shape).toEqual([2]);
    expectArraysClose(await t2.data(), [2, 22]);

    t2 = tf.gather(t, tf.scalar(1, 'int32'), 1);
    expect(t2.shape).toEqual([2]);
    expectArraysClose(await t2.data(), [11, 22]);
  });

  it('2D (gather), 1D indices', async () => {
    const t = tf.tensor2d([1, 11, 2, 22], [2, 2]);
    let t2 = tf.gather(t, tf.tensor1d([1, 0, 0, 1], 'int32'), 0);
    expect(t2.shape).toEqual([4, 2]);
    expectArraysClose(await t2.data(), [2, 22, 1, 11, 1, 11, 2, 22]);

    t2 = tf.gather(t, tf.tensor1d([1, 0, 0, 1], 'int32'), 1);
    expect(t2.shape).toEqual([2, 4]);
    expectArraysClose(await t2.data(), [11, 1, 1, 11, 22, 2, 2, 22]);
  });

  it('2D (gather), 2D indices', async () => {
    const t = tf.tensor2d([1, 11, 2, 22], [2, 2]);
    let t2 = tf.gather(t, tf.tensor2d([1, 0, 0, 1], [2, 2], 'int32'), 0);
    expect(t2.shape).toEqual([2, 2, 2]);
    expectArraysClose(await t2.data(), [2, 22, 1, 11, 1, 11, 2, 22]);

    t2 = tf.gather(t, tf.tensor2d([1, 0, 0, 1], [2, 2], 'int32'), 1);
    expect(t2.shape).toEqual([2, 2, 2]);
    expectArraysClose(await t2.data(), [11, 1, 1, 11, 22, 2, 2, 22]);
  });

  it('2D (gather), 2D indices, non-zero batchDims', async () => {
    const t = tf.tensor2d([1, 11, 2, 22], [2, 2]);
    const t2 = tf.gather(t, tf.tensor2d([1, 0, 0, 1], [2, 2], 'int32'), 1, 1);
    expect(t2.shape).toEqual([2, 2]);
    expectArraysClose(await t2.data(), [11, 1, 2, 22]);
  });

  it('2D (gather), 2D indices, negative batchDims', async () => {
    const t = tf.tensor2d([1, 11, 2, 22], [2, 2]);
    const t2 = tf.gather(t, tf.tensor2d([1, 0, 0, 1], [2, 2], 'int32'), 1, -1);
    expect(t2.shape).toEqual([2, 2]);
    expectArraysClose(await t2.data(), [11, 1, 2, 22]);
  });

  it('3D (gather), 1D indices', async () => {
    const t = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);

    const t2 = tf.gather(t, tf.tensor1d([1, 0, 0, 1], 'int32'), 2);

    expect(t2.shape).toEqual([2, 2, 4]);
    expectArraysClose(
        await t2.data(), [2, 1, 1, 2, 4, 3, 3, 4, 6, 5, 5, 6, 8, 7, 7, 8]);
  });

  it('3D (gather), 2D indices', async () => {
    const t = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);

    const t2 = tf.gather(t, tf.tensor2d([1, 0, 0, 1], [2, 2], 'int32'), 2);

    expect(t2.shape).toEqual([2, 2, 2, 2]);
    expectArraysClose(
        await t2.data(), [2, 1, 1, 2, 4, 3, 3, 4, 6, 5, 5, 6, 8, 7, 7, 8]);
  });

  it('3D (gather), 2D indices, non-zero batchDims', async () => {
    const t = tf.tensor3d([1, 2, 3, 4], [1, 2, 2]);

    const t2 = tf.gather(t, tf.tensor2d([1, 0, 1], [1, 3], 'int32'), 2, 1);

    expect(t2.shape).toEqual([1, 2, 3]);
    expectArraysClose(await t2.data(), [2, 1, 2, 4, 3, 4]);
  });

  it('throws when batch dims greater than axis', () => {
    const t = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);

    expect(() => tf.gather(t, tf.tensor3d([1, 0, 1], [1, 1, 3], 'int32'), 2, 3))
        .toThrowError(/must be less than or equal to axis/);
  });

  it('throws when batch dims greater than indices rank', () => {
    const t = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2]);

    expect(() => tf.gather(t, tf.tensor2d([1, 0, 1], [1, 3], 'int32'), 2, 3))
        .toThrowError(/Expect batchDims in the range of /);
  });

  it('throws when batch dims do not match', () => {
    const t = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);

    expect(() => tf.gather(t, tf.tensor2d([1, 0, 1], [1, 3], 'int32'), 2, 1))
        .toThrowError(/should be equal to indices.shape/);
  });

  it('bool (gather), 1D indices', async () => {
    const t = tf.tensor1d([true, false, true], 'bool');

    const t2 = tf.gather(t, tf.tensor1d([0, 2, 0, 1], 'int32'), 0);

    expect(t2.shape).toEqual([4]);
    expect(t2.dtype).toBe('bool');
    expect(await t2.data()).toEqual(new Uint8Array([1, 1, 1, 0]));
  });

  it('bool (gather), 2D indices', async () => {
    const t = tf.tensor1d([true, false, true], 'bool');

    const t2 = tf.gather(t, tf.tensor2d([0, 2, 0, 1], [2, 2], 'int32'), 0);

    expect(t2.shape).toEqual([2, 2]);
    expect(t2.dtype).toBe('bool');
    expect(await t2.data()).toEqual(new Uint8Array([1, 1, 1, 0]));
  });

  it('int32 (gather), 1D indices', async () => {
    const t = tf.tensor1d([1, 2, 5], 'int32');

    const t2 = tf.gather(t, tf.tensor1d([0, 2, 0, 1], 'int32'), 0);

    expect(t2.shape).toEqual([4]);
    expect(t2.dtype).toBe('int32');
    expect(await t2.data()).toEqual(new Int32Array([1, 5, 1, 2]));
  });

  it('int32 (gather), 2D indices', async () => {
    const t = tf.tensor1d([1, 2, 5], 'int32');

    const t2 = tf.gather(t, tf.tensor2d([0, 2, 0, 1], [2, 2], 'int32'), 0);

    expect(t2.shape).toEqual([2, 2]);
    expect(t2.dtype).toBe('int32');
    expect(await t2.data()).toEqual(new Int32Array([1, 5, 1, 2]));
  });

  it('propagates NaNs', async () => {
    const t = tf.tensor1d([1, 2, NaN]);

    const t2 = tf.gather(t, tf.tensor1d([0, 2, 0, 1], 'int32'), 0);

    expect(t2.shape).toEqual([4]);
    expectArraysClose(await t2.data(), [1, NaN, 1, 2]);
  });

  it('chaining, axis=1', () => {
    const x = tf.zeros([2, 4, 6]);
    // [0, 2, 4]
    const indices = tf.range(0, 6, 2, 'int32');
    const axis = 2;
    expect(x.gather(indices, axis).shape).toEqual([2, 4, 3]);
  });

  it('indices not int32 throws error', () => {
    const x = tf.zeros([2, 4, 6]);
    // [0, 2, 4]
    const indices = tf.range(0, 6, 2);
    const axis = 2;
    expect(() => x.gather(indices, axis)).toThrowError();
  });

  it('throws when passed x as a non-tensor', () => {
    expect(() => tf.gather({} as tf.Tensor, tf.tensor1d([1])))
        .toThrowError(/Argument 'x' passed to 'gather' must be a Tensor/);
  });

  it('throws when passed indices as a non-tensor', () => {
    // tslint:disable-next-line:no-any
    expect(() => tf.gather(tf.tensor1d([1]), {} as any))
        .toThrowError(/Argument 'indices' passed to 'gather' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const res = tf.gather([1, 2, 3], [0, 2, 0, 1], 0);
    expect(res.shape).toEqual([4]);
    expectArraysClose(await res.data(), [1, 3, 1, 2]);
  });

  it('gradient 1D (gather), 1D indices', async () => {
    const t = tf.tensor1d([1, 2, 3]);
    const indices = tf.tensor1d([0, 2, 0, 1], 'int32');
    const dy = tf.tensor([3, 4, 5, 6]);

    const gradients = tf.grad(t => tf.gather(t, indices))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(await gradients.data(), [8, 6, 4]);
  });

  it('gradient with clones', () => {
    const t = tf.tensor1d([1, 2, 3]);
    const indices = tf.tensor1d([0, 2, 0, 1], 'int32');
    const gradF = tf.grad(t => tf.gather(t.clone(), indices.clone()).clone());
    const dt = gradF(t);
    expect(dt.shape).toEqual(t.shape);
  });

  it('gradient 1D (gather), 2D indices', async () => {
    const t = tf.tensor1d([1, 2, 3]);
    const indices = tf.tensor2d([0, 2, 0, 1], [2, 2], 'int32');
    const dy = tf.tensor2d([3, 4, 5, 6], [2, 2]);

    const gradients = tf.grad(t => tf.gather(t, indices))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(await gradients.data(), [8, 6, 4]);
  });

  it('gradient 2D (gather) axis=0 shape=[2, 2] 1D indices', async () => {
    const t = tf.tensor2d([1, 11, 2, 22], [2, 2]);
    const indices = tf.tensor1d([1, 0, 0, 1], 'int32');
    const dy = tf.tensor([3, 4, 5, 6, 7, 8, 9, 10], [4, 2]);
    const axis = 0;

    const gradients = tf.grad(t => tf.gather(t, indices, axis))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(await gradients.data(), [12, 14, 12, 14]);
  });

  it('gradient 2D (gather) axis=0 shape=[2, 2] 2D indices', async () => {
    const t = tf.tensor2d([1, 11, 2, 22], [2, 2]);
    const indices = tf.tensor2d([1, 0, 0, 1], [2, 2], 'int32');
    const dy = tf.tensor([3, 4, 5, 6, 7, 8, 9, 10], [2, 2, 2]);
    const axis = 0;

    const gradients = tf.grad(t => tf.gather(t, indices, axis))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(await gradients.data(), [12, 14, 12, 14]);
  });

  it('gradient 2D (gather) axis=0 shape=[4, 1] 1D indices', async () => {
    const t = tf.tensor2d([1, 11, 2, 22], [4, 1]);
    const indices = tf.tensor1d([1, 0, 0, 1], 'int32');
    const dy = tf.tensor([23, 7, 19, 13], [4, 1]);
    const axis = 0;

    const gradients = tf.grad(t => tf.gather(t, indices, axis))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(await gradients.data(), [26, 36, 0, 0]);
  });

  it('gradient 2D (gather) axis=0 shape=[4, 1] 2D indices', async () => {
    const t = tf.tensor2d([1, 11, 2, 22], [4, 1]);
    const indices = tf.tensor2d([1, 0, 0, 1], [2, 2], 'int32');
    const dy = tf.tensor([23, 7, 19, 13], [2, 2, 1]);
    const axis = 0;

    const gradients = tf.grad(t => tf.gather(t, indices, axis))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(await gradients.data(), [26, 36, 0, 0]);
  });

  it('gradient 2D (gather) axis=1 shape=[2, 2] 1D indices', async () => {
    const t = tf.tensor2d([1, 11, 2, 22], [2, 2]);
    const indices = tf.tensor1d([1, 0, 0, 1], 'int32');
    const dy = tf.tensor([3, 4, 5, 6, 7, 8, 9, 10], [2, 4]);
    const axis = 1;

    const gradients = tf.grad(t => tf.gather(t, indices, axis))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(await gradients.data(), [9, 9, 17, 17]);
  });

  it('gradient 2D (gather) axis=1 shape=[2, 2] 2D indices', async () => {
    const t = tf.tensor2d([1, 11, 2, 22], [2, 2]);
    const indices = tf.tensor2d([1, 0, 0, 1], [2, 2], 'int32');
    const dy = tf.tensor([3, 4, 5, 6, 7, 8, 9, 10], [2, 2, 2]);
    const axis = 1;

    const gradients = tf.grad(t => tf.gather(t, indices, axis))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(await gradients.data(), [9, 9, 17, 17]);
  });

  it('gradient 2D (gather) axis=1 shape=[4, 1] 1D indices', async () => {
    const t = tf.tensor2d([1, 11, 2, 22], [4, 1]);
    const indices = tf.tensor1d([0, 0, 0, 0], 'int32');
    const dy = tf.tensor(
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], [4, 4]);
    const axis = 1;

    const gradients = tf.grad(t => tf.gather(t, indices, axis))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(await gradients.data(), [18, 34, 50, 66]);
  });

  it('gradient 2D (gather) axis=1 shape=[4, 1] 2D indices', async () => {
    const t = tf.tensor2d([1, 11, 2, 22], [4, 1]);
    const indices = tf.tensor2d([0, 0, 0, 0], [2, 2], 'int32');
    const dy = tf.tensor(
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], [4, 2, 2]);
    const axis = 1;

    const gradients = tf.grad(t => tf.gather(t, indices, axis))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(await gradients.data(), [18, 34, 50, 66]);
  });

  it('gradient 3D (gather) axis=0 shape=[2, 3, 2] 1D indices', async () => {
    const t = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 3, 2]);
    const indices = tf.tensor1d([1, 0, 0, 1], 'int32');
    const dy = tf.tensor(
        [
          2, -3, 4,  15, 6,  0.7, 1, 18, 0.01, 0,  12, 13,
          4, 15, 12, -7, 18, 19,  2, 21, 6,    23, 24, 25
        ],
        [4, 3, 2]);
    const axis = 0;

    const gradients = tf.grad(t => tf.gather(t, indices, axis))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(
        await gradients.data(),
        [5, 33, 12.01, -7, 30, 32, 4, 18, 10, 38, 30, 25.7]);
  });

  it('gradient 3D (gather) axis=0 shape=[2, 3, 2] 2D indices', async () => {
    const t = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 3, 2]);
    const indices = tf.tensor2d([1, 0, 0, 1], [2, 2], 'int32');
    const dy = tf.tensor(
        [
          2, -3, 4,  15, 6,  0.7, 1, 18, 0.01, 0,  12, 13,
          4, 15, 12, -7, 18, 19,  2, 21, 6,    23, 24, 25
        ],
        [2, 2, 3, 2]);
    const axis = 0;

    const gradients = tf.grad(t => tf.gather(t, indices, axis))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(
        await gradients.data(),
        [5, 33, 12.01, -7, 30, 32, 4, 18, 10, 38, 30, 25.7]);
  });

  it('gradient 3D (gather) axis=0 shape=[1, 4, 4]', async () => {
    const t = tf.tensor3d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [1, 4, 4]);
    const indices = tf.tensor1d([0, 0], 'int32');
    const dy = tf.tensor(
        [
          2,  -3, 4, 15, 6, 0.7, 1,  18, 0.01, 0,  12, 13, 4, 15, 12, -7,
          18, 19, 2, 21, 6, 23,  24, 25, 101,  31, 34, 54, 1, 0,  -3, -4
        ],
        [2, 4, 4]);
    const axis = 0;

    const gradients = tf.grad(t => tf.gather(t, indices, axis))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(
        await gradients.data(),
        [20, 16, 6, 36, 12, 23.7, 25, 43, 101.01, 31, 46, 67, 5, 15, 9, -11]);
  });

  it('gradient 3D (gather) axis=0 shape=[1, 4, 4] 1D indices', async () => {
    const t = tf.tensor3d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [1, 4, 4]);
    const indices = tf.tensor1d([0, 0], 'int32');
    const dy = tf.tensor(
        [
          2,  -3, 4, 15, 6, 0.7, 1,  18, 0.01, 0,  12, 13, 4, 15, 12, -7,
          18, 19, 2, 21, 6, 23,  24, 25, 101,  31, 34, 54, 1, 0,  -3, -4
        ],
        [2, 4, 4]);
    const axis = 0;

    const gradients = tf.grad(t => tf.gather(t, indices, axis))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(
        await gradients.data(),
        [20, 16, 6, 36, 12, 23.7, 25, 43, 101.01, 31, 46, 67, 5, 15, 9, -11]);
  });

  it('gradient 3D (gather) axis=1 shape=[2, 3, 2] 2D indices', async () => {
    const t = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 3, 2]);
    const indices = tf.tensor2d([1, 2, 2, 1], [2, 2], 'int32');
    const dy = tf.tensor(
        [2, -3, 4, 15, 6, 0.7, 1, 18, 0.01, 0, 12, 13, 4, 15, 12, -7],
        [2, 2, 2, 2]);
    const axis = 1;

    const gradients = tf.grad(t => tf.gather(t, indices, axis))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(
        await gradients.data(),
        [0, 0, 3, 15, 10, 15.7, 0, 0, 12.01, -7, 16, 28]);
  });

  it('gradient 3D (gather) axis=1 shape=[1, 4, 4] 1D indices', async () => {
    const t = tf.tensor3d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [1, 4, 4]);
    const indices = tf.tensor1d([1, 2, 2, 1], 'int32');
    const dy = tf.tensor(
        [2, -3, 4, 15, 6, 0.7, 1, 18, 0.01, 0, 12, 13, 4, 15, 12, -7],
        [1, 4, 4]);
    const axis = 1;

    const gradients = tf.grad(t => tf.gather(t, indices, axis))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(
        await gradients.data(),
        [0, 0, 0, 0, 6, 12, 16, 8, 6.01, .7, 13, 31, 0, 0, 0, 0]);
  });

  it('gradient 3D (gather) axis=1 shape=[1, 4, 4] 2D indices', async () => {
    const t = tf.tensor3d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [1, 4, 4]);
    const indices = tf.tensor2d([1, 2, 2, 1], [2, 2], 'int32');
    const dy = tf.tensor(
        [2, -3, 4, 15, 6, 0.7, 1, 18, 0.01, 0, 12, 13, 4, 15, 12, -7],
        [1, 2, 2, 4]);
    const axis = 1;

    const gradients = tf.grad(t => tf.gather(t, indices, axis))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(
        await gradients.data(),
        [0, 0, 0, 0, 6, 12, 16, 8, 6.01, .7, 13, 31, 0, 0, 0, 0]);
  });

  it('gradient 3D (gather) axis=2 shape=[2, 3, 2] 1D indices', async () => {
    const t = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 3, 2]);
    const indices = tf.tensor1d([1, 0, 1, 0], 'int32');
    const dy = tf.tensor(
        [
          2, -3, 4,  15, 6,  0.7, 1, 18, 0.01, 0,  12, 13,
          4, 15, 12, -7, 18, 19,  2, 21, 6,    23, 24, 25
        ],
        [2, 3, 4]);
    const axis = 2;

    const gradients = tf.grad(t => tf.gather(t, indices, axis))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(
        await gradients.data(),
        [12, 6, 18.7, 7, 13, 12.01, 8, 16, 40, 20, 48, 30]);
  });

  it('gradient 3D (gather) axis=2 shape=[2, 3, 2] 2D indices', async () => {
    const t = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 3, 2]);
    const indices = tf.tensor2d([1, 0, 1, 0], [2, 2], 'int32');
    const dy = tf.tensor(
        [
          2, -3, 4,  15, 6,  0.7, 1, 18, 0.01, 0,  12, 13,
          4, 15, 12, -7, 18, 19,  2, 21, 6,    23, 24, 25
        ],
        [2, 3, 2, 2]);
    const axis = 2;

    const gradients = tf.grad(t => tf.gather(t, indices, axis))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(
        await gradients.data(),
        [12, 6, 18.7, 7, 13, 12.01, 8, 16, 40, 20, 48, 30]);
  });

  it('gradient 3D (gather) axis=2 shape=[4, 1, 4] 1D indices', async () => {
    const t = tf.tensor3d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [4, 1, 4]);
    const indices = tf.tensor1d([1, 3, 1], 'int32');
    const dy =
        tf.tensor([2, -3, 4, 15, 6, 0.7, 1, 18, 0.01, 0, 4, 15], [4, 1, 3]);
    const axis = 2;

    const gradients = tf.grad(t => tf.gather(t, indices, axis))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(
        await gradients.data(),
        [0, 6, 0, -3, 0, 15.7, 0, 6, 0, 1.01, 0, 18, 0, 15, 0, 4]);
  });

  it('gradient 3D (gather) axis=2 shape=[4, 1, 4] 2D indices', async () => {
    const t = tf.tensor3d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [4, 1, 4]);
    const indices = tf.tensor2d([1, 3, 1], [1, 3], 'int32');
    const dy =
        tf.tensor([2, -3, 4, 15, 6, 0.7, 1, 18, 0.01, 0, 4, 15], [4, 1, 1, 3]);
    const axis = 2;

    const gradients = tf.grad(t => tf.gather(t, indices, axis))(t, dy);

    expect(gradients.shape).toEqual(t.shape);
    expectArraysClose(
        await gradients.data(),
        [0, 6, 0, -3, 0, 15.7, 0, 6, 0, 1.01, 0, 18, 0, 15, 0, 4]);
  });

  it('ensure no memory leak', async () => {
    const numTensorsBefore = tf.memory().numTensors;
    const numDataIdBefore = tf.engine().backend.numDataIds();
    const t = tf.tensor1d([1, 2, 3]);
    const t1 = tf.scalar(1, 'int32');
    const t2 = tf.gather(t, t1, 0);

    expect(t2.shape).toEqual([]);
    expectArraysClose(await t2.data(), [2]);

    t.dispose();
    t1.dispose();
    t2.dispose();

    const numTensorsAfter = tf.memory().numTensors;
    const numDataIdAfter = tf.engine().backend.numDataIds();
    expect(numTensorsAfter).toBe(numTensorsBefore);
    expect(numDataIdAfter).toBe(numDataIdBefore);
  });
});
