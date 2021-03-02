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

describeWithFlags('expandDims', ALL_ENVS, () => {
  it('scalar, default axis is 0', async () => {
    const res = tf.scalar(1).expandDims();
    expect(res.shape).toEqual([1]);
    expectArraysClose(await res.data(), [1]);
  });

  it('scalar, axis is out of bounds throws error', () => {
    const f = () => tf.scalar(1).expandDims(1);
    expect(f).toThrowError();
  });

  it('1d, axis=-3', () => {
    expect(() => {
      tf.tensor1d([1, 2, 3]).expandDims(-3);
    }).toThrowError();
  });

  it('1d, axis=-2', async () => {
    const res = tf.tensor1d([1, 2, 3]).expandDims(-2 /* axis */);
    expect(res.shape).toEqual([1, 3]);
    expectArraysClose(await res.data(), [1, 2, 3]);
  });

  it('1d, axis=-1', async () => {
    const res = tf.tensor1d([1, 2, 3]).expandDims(-1 /* axis */);
    expect(res.shape).toEqual([3, 1]);
    expectArraysClose(await res.data(), [1, 2, 3]);
  });

  it('1d, axis=0', async () => {
    const res = tf.tensor1d([1, 2, 3]).expandDims(0 /* axis */);
    expect(res.shape).toEqual([1, 3]);
    expectArraysClose(await res.data(), [1, 2, 3]);
  });

  it('1d, axis=1', async () => {
    const res = tf.tensor1d([1, 2, 3]).expandDims(1 /* axis */);
    expect(res.shape).toEqual([3, 1]);
    expectArraysClose(await res.data(), [1, 2, 3]);
  });

  it('2d, axis=-4', () => {
    expect(() => {
      tf.tensor2d([[1, 2], [3, 4], [5, 6]]).expandDims(-4 /* axis */);
    }).toThrowError();
  });

  it('2d, axis=-3', async () => {
    const res = tf.tensor2d([[1, 2], [3, 4], [5, 6]]).expandDims(-3 /* axis */);
    expect(res.shape).toEqual([1, 3, 2]);
    expectArraysClose(await res.data(), [1, 2, 3, 4, 5, 6]);
  });

  it('2d, axis=-2', async () => {
    const res = tf.tensor2d([[1, 2], [3, 4], [5, 6]]).expandDims(-2 /* axis */);
    expect(res.shape).toEqual([3, 1, 2]);
    expectArraysClose(await res.data(), [1, 2, 3, 4, 5, 6]);
  });

  it('2d, axis=-1', async () => {
    const res = tf.tensor2d([[1, 2], [3, 4], [5, 6]]).expandDims(-1 /* axis */);
    expect(res.shape).toEqual([3, 2, 1]);
    expectArraysClose(await res.data(), [1, 2, 3, 4, 5, 6]);
  });

  it('2d, axis=0', async () => {
    const res = tf.tensor2d([[1, 2], [3, 4], [5, 6]]).expandDims(0 /* axis */);
    expect(res.shape).toEqual([1, 3, 2]);
    expectArraysClose(await res.data(), [1, 2, 3, 4, 5, 6]);
  });

  it('2d, axis=1', async () => {
    const res = tf.tensor2d([[1, 2], [3, 4], [5, 6]]).expandDims(1 /* axis */);
    expect(res.shape).toEqual([3, 1, 2]);
    expectArraysClose(await res.data(), [1, 2, 3, 4, 5, 6]);
  });

  it('2d, axis=2', async () => {
    const res = tf.tensor2d([[1, 2], [3, 4], [5, 6]]).expandDims(2 /* axis */);
    expect(res.shape).toEqual([3, 2, 1]);
    expectArraysClose(await res.data(), [1, 2, 3, 4, 5, 6]);
  });

  it('4d, axis=0', async () => {
    const res = tf.tensor4d([[[[4]]]]).expandDims();
    expect(res.shape).toEqual([1, 1, 1, 1, 1]);
    expectArraysClose(await res.data(), [4]);
  });

  it('1d string tensor', async () => {
    const t = tf.tensor(['hello', 'world']);
    const res = t.expandDims();
    expect(res.shape).toEqual([1, 2]);
    expectArraysClose(await res.data(), ['hello', 'world']);
  });

  it('2d string tensor, axis=1', async () => {
    const t = tf.tensor([['a', 'b'], ['c', 'd']]);
    const res = t.expandDims(1);
    expect(res.shape).toEqual([2, 1, 2]);
    expectArraysClose(await res.data(), ['a', 'b', 'c', 'd']);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.expandDims({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'expandDims' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const res = tf.expandDims(7);
    expect(res.shape).toEqual([1]);
    expectArraysClose(await res.data(), [7]);
  });

  it('works with 0 in shape', async () => {
    const a = tf.tensor2d([], [0, 3]);
    const res = a.expandDims();
    expect(res.shape).toEqual([1, 0, 3]);
    expectArraysClose(await res.data(), []);

    const res2 = a.expandDims(1);
    expect(res2.shape).toEqual([0, 1, 3]);
    expectArraysClose(await res2.data(), []);

    const res3 = a.expandDims(2);
    expect(res3.shape).toEqual([0, 3, 1]);
    expectArraysClose(await res3.data(), []);
  });
  it('ensure no memory leak', async () => {
    const numTensorsBefore = tf.memory().numTensors;
    const numDataIdBefore = tf.engine().backend.numDataIds();

    const t = tf.scalar(1);
    const res = t.expandDims();
    expect(res.shape).toEqual([1]);
    expectArraysClose(await res.data(), [1]);

    res.dispose();
    t.dispose();

    const numTensorsAfter = tf.memory().numTensors;
    const numDataIdAfter = tf.engine().backend.numDataIds();
    expect(numTensorsAfter).toBe(numTensorsBefore);
    expect(numDataIdAfter).toBe(numDataIdBefore);
  });
});
