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

describeWithFlags('avgPool', ALL_ENVS, () => {
  it('x=[1,1,1] f=[1,1] s=1 [0] => [0]', async () => {
    const a = tf.tensor3d([0], [1, 1, 1]);
    const result = tf.avgPool(a, 1, 1, 0);
    expectArraysClose(await result.data(), [0]);
  });

  it('x=[3,3,1] f=[2,2] s=1', async () => {
    // Feed forward.
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 9, 8], [3, 3, 1]);
    const result = tf.avgPool(a, 2, 1, 0);

    expect(result.shape).toEqual([2, 2, 1]);
    expect(result.dtype).toBe('float32');
    expectArraysClose(await result.data(), [3, 4, 6.25, 7]);
  });

  it('input int32 throws error', () => {
    // Feed forward.
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 9, 8], [3, 3, 1], 'int32');
    expect(() => tf.avgPool(a, 2, 1, 0)).toThrowError();
  });

  it('x=[2,3,3,1] f=[2,2], s=1', async () => {
    // Feed forward.
    const a = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 9, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 3, 3, 1]);
    const result = tf.avgPool(a, 2, 1, 0);

    expect(result.shape).toEqual([2, 2, 2, 1]);
    expectArraysClose(await result.data(), [3, 4, 6.25, 7, 3, 4, 6, 7]);
  });

  it('x=[3,3,1] f=[2,2] s=1 propagates NaNs', async () => {
    // Feed forward.
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, NaN, 8], [3, 3, 1]);
    const result = tf.avgPool(a, 2, 1, 0);

    expect(result.shape).toEqual([2, 2, 1]);
    expectArraysClose(await result.data(), [3, 4, NaN, NaN]);
  });

  it('x=[3,3,2] f=[2,2] s=1', async () => {
    // Feed forward.
    const a = tf.tensor3d(
        [1, 99, 2, 88, 3, 77, 4, 66, 5, 55, 6, 44, 7, 33, 9, 22, 8, 11],
        [3, 3, 2]);
    const result = tf.avgPool(a, 2, 1, 0);

    expect(result.shape).toEqual([2, 2, 2]);
    expectArraysClose(await result.data(), [3, 77, 4, 66, 6.25, 44, 7, 33]);
  });

  it('x=[4,4,1] f=[2,2] s=2', async () => {
    // Feed forward.
    const a = tf.tensor3d(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [4, 4, 1]);
    const result = tf.avgPool(a, 2, 2, 0);

    expect(result.shape).toEqual([2, 2, 1]);
    expectArraysClose(await result.data(), [2.5, 4.5, 10.5, 12.5]);
  });

  it('x=[2,2,1] f=[2,2] s=1 p=same', async () => {
    // Feed forward.
    const a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
    const fSize = 2;
    const strides = 1;
    const result = tf.avgPool(a, fSize, strides, 'same');

    expect(result.shape).toEqual([2, 2, 1]);
    expectArraysClose(await result.data(), [2.5, 3, 3.5, 4]);
  });

  it('x=[2,2,3] f=[1,1] s=2 p=1 dimRoundingMode=floor', () => {
    // Feed forward.
    const x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 2, 3]);
    const result = tf.avgPool(x, 1, 2, 1, 'floor');

    expect(result.shape).toEqual([2, 2, 3]);
  });

  it('gradient x=[1,1,1] f=[1,1] s=1 [0] => [0]', async () => {
    const x = tf.tensor3d([0], [1, 1, 1]);
    const dy = tf.tensor3d([0], [1, 1, 1]);
    const dx = tf.grad((x: tf.Tensor3D) => x.avgPool(1, 1, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), [0]);
  });

  it('gradient with clones', async () => {
    const x = tf.tensor3d([0], [1, 1, 1]);
    const dy = tf.tensor3d([0], [1, 1, 1]);
    const dx = tf.grad(
        (x: tf.Tensor3D) => tf.avgPool(x.clone(), 1, 1, 0).clone())(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), [0]);
  });

  it('gradient x=[3,3,1] f=[2,2] s=1', async () => {
    // Feed forward.
    const x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 9, 8], [3, 3, 1]);
    const dy = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
    const avgMultiplier = 1 / (2 * 2);

    const dx = tf.grad((x: tf.Tensor3D) => x.avgPool(2, 1, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), [
      1 * avgMultiplier, 3 * avgMultiplier, 2 * avgMultiplier,
      4 * avgMultiplier, 10 * avgMultiplier, 6 * avgMultiplier,
      3 * avgMultiplier, 7 * avgMultiplier, 4 * avgMultiplier
    ]);
  });

  it('gradient x=[2,3,3,1] f=[2,2], s=1', async () => {
    // Feed forward.
    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 9, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 3, 3, 1]);
    const dy = tf.tensor4d([1, 2, 3, 4, 1, 2, 3, 4], [2, 2, 2, 1]);
    const avgMultiplier = 1 / (2 * 2);

    const dx = tf.grad((x: tf.Tensor4D) => x.avgPool(2, 1, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), [
      1 * avgMultiplier, 3 * avgMultiplier, 2 * avgMultiplier,
      4 * avgMultiplier, 10 * avgMultiplier, 6 * avgMultiplier,
      3 * avgMultiplier, 7 * avgMultiplier, 4 * avgMultiplier,
      1 * avgMultiplier, 3 * avgMultiplier, 2 * avgMultiplier,
      4 * avgMultiplier, 10 * avgMultiplier, 6 * avgMultiplier,
      3 * avgMultiplier, 7 * avgMultiplier, 4 * avgMultiplier
    ]);
  });

  it('throws when dimRoundingMode is set and pad is not a number', () => {
    const x = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);

    const pad = 'valid';
    const dimRoundingMode = 'round';

    expect(() => tf.avgPool(x, 2, 1, pad, dimRoundingMode)).toThrowError();
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.avgPool({} as tf.Tensor3D, 2, 1, 'valid'))
        .toThrowError(/Argument 'x' passed to 'avgPool' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const a = [[[0]]];  // 1x1x1
    const result = tf.avgPool(a, 1, 1, 0);
    expectArraysClose(await result.data(), [0]);
  });
});
