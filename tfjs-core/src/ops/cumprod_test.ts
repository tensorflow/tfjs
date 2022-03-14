/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '../index';
import { ALL_ENVS, describeWithFlags } from '../jasmine_util';
import { expectArraysClose } from '../test_util';

describeWithFlags('cumprod', ALL_ENVS, () => {
  it('1D standard', async () => {
    const res = tf.tensor1d([1, 2, 3, 4]).cumprod();
    expect(res.shape).toEqual([4]);
    expectArraysClose(await res.data(), [1, 2, 6, 24]);
  });

  it('1D reverse', async () => {
    const reverse = true;
    const exclusive = false;
    const res = tf.tensor1d([1, 2, 3, 4]).cumprod(0, exclusive, reverse);
    expect(res.shape).toEqual([4]);
    expectArraysClose(await res.data(), [24, 24, 12, 4]);
  });

  it('1D exclusive', async () => {
    const exclusive = true;
    const res = tf.tensor1d([1, 2, 3, 4]).cumprod(0, exclusive);
    expect(res.shape).toEqual([4]);
    expectArraysClose(await res.data(), [1, 1, 2, 6]);
  });

  it('1D exclusive reverse', async () => {
    const reverse = true;
    const exclusive = true;
    const res = tf.tensor1d([1, 2, 3, 4]).cumprod(0, exclusive, reverse);
    expect(res.shape).toEqual([4]);
    expectArraysClose(await res.data(), [24, 12, 4, 1]);
  });

  // TODO: once gradients are implemented, create tests something like this.
  // it('gradient: 1D', async () => {
  //   const a = tf.tensor1d([1, 2, 3]);
  //   const dy = tf.tensor1d([4, 5, 6]);
  //   const da = tf.grad((x) => tf.cumprod(x))(a, dy);

  //   expect(da.shape).toEqual([3]);
  //   expectArraysClose(await da.data(), [15, 11, 6]);
  // });

  // it('gradient with clones', async () => {
  //   const a = tf.tensor1d([1, 2, 3]);
  //   const dy = tf.tensor1d([4, 5, 6]);
  //   const da = tf.grad((x) => tf.cumprod(x.clone()).clone())(a, dy);

  //   expect(da.shape).toEqual([3]);
  //   expectArraysClose(await da.data(), [15, 11, 6]);
  // });

  it('2D standard', async () => {
    const res = tf
      .tensor2d([
        [1, 2],
        [3, 4],
      ])
      .cumprod(1);
    expect(res.shape).toEqual([2, 2]);
    expectArraysClose(await res.array(), [[1, 2], [3, 12]]);
  });

  it('2D reverse exclusive', async () => {
    const reverse = true;
    const exclusive = true;
    const res = tf
      .tensor2d([
        [1, 2],
        [3, 4],
      ])
      .cumprod(1, exclusive, reverse);
    expect(res.shape).toEqual([2, 2]);
    expectArraysClose(await res.array(), [[2, 1], [4, 1]]);
  });

  it('2D axis=0', async () => {
    const res = tf
      .tensor2d([
        [1, 2],
        [3, 4],
      ])
      .cumprod();
    expect(res.shape).toEqual([2, 2]);
    expectArraysClose(await res.array(), [[1, 2], [3, 8]]);
  });

  it('3D standard', async () => {
    const res = tf
      .tensor3d([
        [
          [0, 1],
          [2, 3],
        ],
        [
          [4, 5],
          [6, 7],
        ],
      ])
      .cumprod(2);
    expect(res.shape).toEqual([2, 2, 2]);
    expectArraysClose(await res.array(), [
      [
        [0, 0 * 1],
        [2, 2 * 3]
      ],
      [
        [4, 4 * 5],
        [6, 6 * 7]
      ]
    ]);
  });

  it('4d axis=2', async () => {
    const input = tf.add(tf.ones([1, 32, 8, 4]), tf.ones([1, 32, 8, 4]));
    const res = tf.cumprod(input, 2, false, false);

    expect(res.shape).toEqual([1, 32, 8, 4]);

    const earlySlice = tf.slice(res, [0, 0, 0, 0], [1, 1, 8, 1]);
    const lateSlice = tf.slice(res, [0, 31, 0, 0], [1, 1, 8, 1]);
    const expectedDataInEachSlice = [2, 4, 8, 16, 32, 64, 128, 256];
    expectArraysClose(await earlySlice.data(), expectedDataInEachSlice);
    expectArraysClose(await lateSlice.data(), expectedDataInEachSlice);
  });

  it('handle permutation properly', async () => {
    const res = tf.ones([1, 240, 1, 10]).cumprod(1);
    expect(res.shape).toEqual([1, 240, 1, 10]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.cumprod({} as tf.Tensor)).toThrowError(
      /Argument 'x' passed to 'cumprod' must be a Tensor/
    );
  });

  it('accepts a tensor-like object', async () => {
    const res = tf.cumprod([1, 2, 3, 4]);
    expect(res.shape).toEqual([4]);
    expectArraysClose(await res.data(), [1, 2, 6, 24]);
  });

  it('throws error for string tensor', () => {
    expect(() => tf.cumprod(['a', 'b', 'c'])).toThrowError(
      /Argument 'x' passed to 'cumprod' must be numeric tensor/
    );
  });
});
