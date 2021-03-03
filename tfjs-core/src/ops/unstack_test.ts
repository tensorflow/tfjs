/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

describeWithFlags('unstack', ALL_ENVS, () => {
  it('unstack by default', async () => {
    const x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
    const res = tf.unstack(x);
    expect(res.length).toEqual(2);
    expect(res[0].rank).toEqual(1);
    expect(res[0].shape).toEqual([4]);
    expectArraysClose(await res[0].data(), [1, 2, 3, 4]);
    expect(res[1].rank).toEqual(1);
    expect(res[1].shape).toEqual([4]);
    expectArraysClose(await res[1].data(), [5, 6, 7, 8]);
  });

  it('chain api', async () => {
    const x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
    const res = x.unstack();
    expect(res.length).toEqual(2);
    expect(res[0].rank).toEqual(1);
    expect(res[0].shape).toEqual([4]);
    expectArraysClose(await res[0].data(), [1, 2, 3, 4]);
    expect(res[1].rank).toEqual(1);
    expect(res[1].shape).toEqual([4]);
    expectArraysClose(await res[1].data(), [5, 6, 7, 8]);
  });

  it('unstack with negative integer axis', async () => {
    const x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);

    let res = tf.unstack(x, -1);
    expect(res.length).toEqual(4);
    expect(res[0].rank).toEqual(1);
    expect(res[0].shape).toEqual([2]);
    expectArraysClose(await res[0].data(), [1, 5]);
    expect(res[1].rank).toEqual(1);
    expect(res[1].shape).toEqual([2]);
    expectArraysClose(await res[1].data(), [2, 6]);
    expect(res[2].rank).toEqual(1);
    expect(res[2].shape).toEqual([2]);
    expectArraysClose(await res[2].data(), [3, 7]);
    expect(res[3].rank).toEqual(1);
    expect(res[3].shape).toEqual([2]);
    expectArraysClose(await res[3].data(), [4, 8]);

    res = tf.unstack(x, -2);
    expect(res.length).toEqual(2);
    expect(res[0].rank).toEqual(1);
    expect(res[0].shape).toEqual([4]);
    expectArraysClose(await res[0].data(), [1, 2, 3, 4]);
    expect(res[1].rank).toEqual(1);
    expect(res[1].shape).toEqual([4]);
    expectArraysClose(await res[1].data(), [5, 6, 7, 8]);
  });

  it('unstack into 3 tensors', async () => {
    const x = tf.tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
    const res = tf.unstack(x, 0);
    expect(res.length).toEqual(3);
    expect(res[0].rank).toEqual(1);
    expect(res[0].shape).toEqual([2]);
    expectArraysClose(await res[0].data(), [1, 2]);
    expect(res[1].rank).toEqual(1);
    expect(res[1].shape).toEqual([2]);
    expectArraysClose(await res[1].data(), [3, 4]);
    expect(res[2].rank).toEqual(1);
    expect(res[2].shape).toEqual([2]);
    expectArraysClose(await res[2].data(), [5, 6]);
  });

  it('unstack by axis=1', async () => {
    const x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
    const res = tf.unstack(x, 1);
    expect(res.length).toEqual(4);
    expect(res[0].rank).toEqual(1);
    expect(res[0].shape).toEqual([2]);
    expectArraysClose(await res[0].data(), [1, 5]);
    expect(res[1].rank).toEqual(1);
    expect(res[1].shape).toEqual([2]);
    expectArraysClose(await res[1].data(), [2, 6]);
    expect(res[2].rank).toEqual(1);
    expect(res[2].shape).toEqual([2]);
    expectArraysClose(await res[2].data(), [3, 7]);
    expect(res[3].rank).toEqual(1);
    expect(res[3].shape).toEqual([2]);
    expectArraysClose(await res[3].data(), [4, 8]);
  });

  it('unstack rank 3 tensor', async () => {
    const x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
    const res = tf.unstack(x);
    expect(res.length).toEqual(2);
    expect(res[0].rank).toEqual(2);
    expect(res[0].shape).toEqual([2, 2]);
    expectArraysClose(await res[0].data(), [1, 2, 3, 4]);
    expect(res[1].rank).toEqual(2);
    expect(res[1].shape).toEqual([2, 2]);
    expectArraysClose(await res[1].data(), [5, 6, 7, 8]);
  });

  it('unstack rank 3 tensor with axis=1', async () => {
    const x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
    const res = tf.unstack(x, 1);
    expect(res.length).toEqual(2);
    expect(res[0].rank).toEqual(2);
    expect(res[0].shape).toEqual([2, 2]);
    expectArraysClose(await res[0].data(), [1, 2, 5, 6]);
    expect(res[1].rank).toEqual(2);
    expect(res[1].shape).toEqual([2, 2]);
    expectArraysClose(await res[1].data(), [3, 4, 7, 8]);
  });

  it('unstack rank 3 tensor with axis=2', async () => {
    const x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
    const res = tf.unstack(x, 2);
    expect(res.length).toEqual(2);
    expect(res[0].rank).toEqual(2);
    expect(res[0].shape).toEqual([2, 2]);
    expectArraysClose(await res[0].data(), [1, 3, 5, 7]);
    expect(res[1].rank).toEqual(2);
    expect(res[1].shape).toEqual([2, 2]);
    expectArraysClose(await res[1].data(), [2, 4, 6, 8]);
  });

  it('unstack rank 4 tensor', async () => {
    const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 1]);
    const res = tf.unstack(x);
    expect(res.length).toEqual(2);
    expect(res[0].rank).toEqual(3);
    expect(res[0].shape).toEqual([2, 2, 1]);
    expectArraysClose(await res[0].data(), [1, 2, 3, 4]);
    expect(res[1].rank).toEqual(3);
    expect(res[1].shape).toEqual([2, 2, 1]);
    expectArraysClose(await res[1].data(), [5, 6, 7, 8]);
  });

  it('unstack rank 4 tensor with axis=1', async () => {
    const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 1]);
    const res = tf.unstack(x, 1);
    expect(res.length).toEqual(2);
    expect(res[0].rank).toEqual(3);
    expect(res[0].shape).toEqual([2, 2, 1]);
    expectArraysClose(await res[0].data(), [1, 2, 5, 6]);
    expect(res[1].rank).toEqual(3);
    expect(res[1].shape).toEqual([2, 2, 1]);
    expectArraysClose(await res[1].data(), [3, 4, 7, 8]);
  });

  it('unstack rank 4 tensor with axis=2', async () => {
    const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 1]);
    const res = tf.unstack(x, 2);
    expect(res.length).toEqual(2);
    expect(res[0].rank).toEqual(3);
    expect(res[0].shape).toEqual([2, 2, 1]);
    expectArraysClose(await res[0].data(), [1, 3, 5, 7]);
    expect(res[1].rank).toEqual(3);
    expect(res[1].shape).toEqual([2, 2, 1]);
    expectArraysClose(await res[1].data(), [2, 4, 6, 8]);
  });

  it('unstack rank 4 tensor with axis=3', async () => {
    const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 1]);
    const res = tf.unstack(x, 3);
    expect(res.length).toEqual(1);
    expect(res[0].rank).toEqual(3);
    expect(res[0].shape).toEqual([2, 2, 2]);
    expectArraysClose(await res[0].data(), [1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.unstack({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'unstack' must be a Tensor/);
  });

  it('throws when passed an invalid axis', () => {
    expect(() => {
      const x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
      tf.unstack(x, 3);
    }).toThrowError('Axis = 3 is not in [-2, 2)');
    expect(() => {
      const x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
      tf.unstack(x, 3);
    }).toThrowError('Axis = 3 is not in [-3, 3)');
    expect(() => {
      const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 1]);
      tf.unstack(x, 5);
    }).toThrowError('Axis = 5 is not in [-4, 4)');
  });

  it('accepts a tensor-like object', async () => {
    const x = [[1, 2, 3, 4], [5, 6, 7, 8]];
    const res = tf.unstack(x);
    expect(res.length).toEqual(2);
    expect(res[0].rank).toEqual(1);
    expect(res[0].shape).toEqual([4]);
    expectArraysClose(await res[0].data(), [1, 2, 3, 4]);
    expect(res[1].rank).toEqual(1);
    expect(res[1].shape).toEqual([4]);
    expectArraysClose(await res[1].data(), [5, 6, 7, 8]);
  });

  it('accepts string', async () => {
    const x =
        [['one', 'two', 'three', 'four'], ['five', 'six', 'seven', 'eight']];
    const res = tf.unstack(x);
    expect(res.length).toEqual(2);
    expect(res[0].rank).toEqual(1);
    expect(res[0].shape).toEqual([4]);
    expectArraysClose(await res[0].data(), ['one', 'two', 'three', 'four']);
    expect(res[1].rank).toEqual(1);
    expect(res[1].shape).toEqual([4]);
    expectArraysClose(await res[1].data(), ['five', 'six', 'seven', 'eight']);
  });

  it('grad of unstack axis=0', async () => {
    const x = tf.tensor([[1, 2, 3], [4, 5, 6]]);
    const dx1 = tf.grad(x => tf.unstack(x)[0])(x);
    expect(dx1.shape).toEqual([2, 3]);
    expect(dx1.dtype).toBe('float32');
    expectArraysClose(await dx1.data(), [1, 1, 1, 0, 0, 0]);

    const dx2 = tf.grad(x => tf.unstack(x)[1])(x);
    expect(dx2.shape).toEqual([2, 3]);
    expect(dx2.dtype).toBe('float32');
    expectArraysClose(await dx2.data(), [0, 0, 0, 1, 1, 1]);
  });

  it('gradient with clones', async () => {
    const x = tf.tensor([[1, 2, 3], [4, 5, 6]]);
    const dx1 = tf.grad(x => tf.unstack(x.clone())[0].clone())(x);
    expect(dx1.shape).toEqual([2, 3]);
    expect(dx1.dtype).toBe('float32');
    expectArraysClose(await dx1.data(), [1, 1, 1, 0, 0, 0]);

    const dx2 = tf.grad(x => tf.unstack(x.clone())[1].clone())(x);
    expect(dx2.shape).toEqual([2, 3]);
    expect(dx2.dtype).toBe('float32');
    expectArraysClose(await dx2.data(), [0, 0, 0, 1, 1, 1]);
  });

  it('grad of unstack axis=1', async () => {
    const x = tf.tensor([[1, 2, 3], [4, 5, 6]]);
    const axis = 1;
    const dx1 = tf.grad(x => tf.unstack(x, axis)[0])(x);
    expect(dx1.shape).toEqual([2, 3]);
    expect(dx1.dtype).toBe('float32');
    expectArraysClose(await dx1.data(), [1, 0, 0, 1, 0, 0]);

    const dx2 = tf.grad(x => tf.unstack(x, axis)[1])(x);
    expect(dx2.shape).toEqual([2, 3]);
    expect(dx2.dtype).toBe('float32');
    expectArraysClose(await dx2.data(), [0, 1, 0, 0, 1, 0]);

    const dx3 = tf.grad(x => tf.unstack(x, axis)[2])(x);
    expect(dx3.shape).toEqual([2, 3]);
    expect(dx3.dtype).toBe('float32');
    expectArraysClose(await dx3.data(), [0, 0, 1, 0, 0, 1]);
  });
});
