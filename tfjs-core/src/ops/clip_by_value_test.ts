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

describeWithFlags('clipByValue', ALL_ENVS, () => {
  it('basic', async () => {
    const a = tf.tensor1d([3, -1, 0, 100, -7, 2]);
    const min = -1;
    const max = 50;

    const result = tf.clipByValue(a, min, max);

    expectArraysClose(await result.data(), [3, -1, 0, 50, -1, 2]);
  });

  it('propagates NaNs', async () => {
    const a = tf.tensor1d([3, -1, 0, 100, -7, 2, NaN]);
    const min = -1;
    const max = 50;

    const result = tf.clipByValue(a, min, max);

    expectArraysClose(await result.data(), [3, -1, 0, 50, -1, 2, NaN]);
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

  it('gradient: 1D tensor', async () => {
    const min = -1;
    const max = 2;
    const x = tf.tensor1d([3, -2, 1]);  // Only 1 is not clipped.
    const dy = tf.tensor1d([5, 50, 500]);
    const gradients = tf.grad(x => x.clipByValue(min, max))(x, dy);

    expect(gradients.shape).toEqual(x.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [0, 0, 500]);
  });

  it('gradient: 1D tensor with max or min value', async () => {
    const min = -1;
    const max = 2;
    const x = tf.tensor1d([-1, 1, 2, 3]);
    const dy = tf.tensor1d([1, 10, 100, 1000]);
    const gradients = tf.grad(x => x.clipByValue(min, max))(x, dy);

    expect(gradients.shape).toEqual(x.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [1, 10, 100, 0]);
  });

  it('gradient: scalar', async () => {
    const min = -1;
    const max = 2;
    const x = tf.scalar(-10);  // Clipped.
    const dy = tf.scalar(5);
    const gradients = tf.grad(x => x.clipByValue(min, max))(x, dy);

    expect(gradients.shape).toEqual(x.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [0]);
  });

  it('gradient with clones', async () => {
    const min = -1;
    const max = 2;
    const x = tf.scalar(-10);  // Clipped.
    const dy = tf.scalar(5);
    const gradients =
        tf.grad(x => x.clone().clipByValue(min, max).clone())(x, dy);

    expect(gradients.shape).toEqual(x.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [0]);
  });

  it('gradient with primitive as input', async () => {
    const min = -1;
    const max = 2;
    const x = -10;
    const dy = tf.scalar(5);
    const gradients = tf.grad(x => x.clipByValue(min, max))(x, dy);
    expect(gradients.shape).toEqual([]);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [0]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.clipByValue({} as tf.Tensor, 0, 1))
        .toThrowError(/Argument 'x' passed to 'clipByValue' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const min = -1;
    const max = 50;
    const result = tf.clipByValue([3, -1, 0, 100, -7, 2], min, max);
    expectArraysClose(await result.data(), [3, -1, 0, 50, -1, 2]);
  });

  it('clip(x, eps, 1-eps) never returns 0 or 1', async () => {
    const min = tf.backend().epsilon();
    const max = 0.5;
    const res = await tf.clipByValue([0, 1], min, max).data();
    expect(res[0]).toBeGreaterThan(0);
    expect(res[1]).toBeCloseTo(max);
  });

  it('throws for string tensor', () => {
    expect(() => tf.clipByValue('q', 0, 1))
        .toThrowError(/Argument 'x' passed to 'clipByValue' must be numeric/);
  });

  it('clip int32 tensor', async () => {
    const min = -1;
    const max = 50;
    const tensor = tf.tensor([2, 3, 4], [3], 'int32');
    const result = tf.clipByValue(tensor, min, max);
    expectArraysClose(await result.data(), [2, 3, 4]);
    expect(result.dtype).toEqual('int32');
  });
});
