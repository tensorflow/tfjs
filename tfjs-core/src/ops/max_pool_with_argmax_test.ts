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
import {expectArraysClose} from '../test_util';

describeWithFlags('maxPoolWithArgmax', ALL_ENVS, () => {
  it('x=[1,1,1] f=[1,1] s=1 d=1 [0] => [0]', async () => {
    const x = tf.tensor4d([0], [1, 1, 1, 1]);

    const padding = 0;

    const {result, indexes} = tf.maxPoolWithArgmax(x, [1, 1], [1, 1], padding);
    expectArraysClose(await result.data(), [0]);
    expectArraysClose(await indexes.data(), [0]);
  });

  it('x=[2,2,2,1] f=[2,2,2] s=1 p=valid', async () => {
    const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 1]);

    const {result, indexes} = tf.maxPoolWithArgmax(x, 2, 1, 'valid');

    expect(result.shape).toEqual([2, 1, 1, 1]);
    expectArraysClose(await result.data(), [4, 8]);
    expect(indexes.shape).toEqual([2, 1, 1, 1]);
    expectArraysClose(await indexes.data(), [3, 3]);
  });

  it('x=[2,2,2,1] f=[2,2,2] s=1 p=valid includeBatchInIndex=true', async () => {
    const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 1]);

    const {result, indexes} = tf.maxPoolWithArgmax(x, 2, 1, 'valid', true);

    expect(result.shape).toEqual([2, 1, 1, 1]);
    expectArraysClose(await result.data(), [4, 8]);
    expect(indexes.shape).toEqual([2, 1, 1, 1]);
    expectArraysClose(await indexes.data(), [3, 7]);
  });

  it('x=[1,3,3,1] f=[2,2] s=1, p=0', async () => {
    // Feed forward.
    const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 9, 8], [1, 3, 3, 1]);

    const {result, indexes} = tf.maxPoolWithArgmax(x, 2, 1, 0);

    expect(result.shape).toEqual([1, 2, 2, 1]);
    expectArraysClose(await result.data(), [5, 6, 9, 9]);
    expect(indexes.shape).toEqual([1, 2, 2, 1]);
    expectArraysClose(await indexes.data(), [4, 5, 7, 7]);
  });

  it('x=[1,3,3,1] f=[2,2] s=1 p=same', async () => {
    const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 9, 8], [1, 3, 3, 1]);

    const {result, indexes} = tf.maxPoolWithArgmax(x, 2, 1, 'same');

    expect(result.shape).toEqual([1, 3, 3, 1]);
    tf.test_util.expectArraysClose(
        await result.data(), new Float32Array([5, 6, 6, 9, 9, 8, 9, 9, 8]));
    expect(indexes.shape).toEqual([1, 3, 3, 1]);
    expectArraysClose(await indexes.data(), [4, 5, 5, 7, 7, 8, 7, 7, 8]);
  });

  it('x=[2,3,3,1] f=[2,2] s=1', async () => {
    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 9, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 3, 3, 1]);

    const {result, indexes} = tf.maxPoolWithArgmax(x, 2, 1, 0);
    expect(result.shape).toEqual([2, 2, 2, 1]);
    expectArraysClose(await result.data(), [5, 6, 9, 9, 5, 6, 8, 9]);
    expect(indexes.shape).toEqual([2, 2, 2, 1]);
    expectArraysClose(await indexes.data(), [4, 5, 7, 7, 4, 5, 7, 8]);
  });

  it('x=[2,3,3,1] f=[2,2] s=1 includeBatchInIndex=true', async () => {
    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 9, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 3, 3, 1]);

    const {result, indexes} = tf.maxPoolWithArgmax(x, 2, 1, 0, true);
    expect(result.shape).toEqual([2, 2, 2, 1]);
    expectArraysClose(await result.data(), [5, 6, 9, 9, 5, 6, 8, 9]);
    expect(indexes.shape).toEqual([2, 2, 2, 1]);
    expectArraysClose(await indexes.data(), [4, 5, 7, 7, 13, 14, 16, 17]);
  });

  it('[x=[1,3,3,1] f=[2,2] s=1 ignores NaNs', async () => {
    const x = tf.tensor4d([NaN, 1, 2, 3, 4, 5, 6, 7, 9], [1, 3, 3, 1]);

    const {result, indexes} = tf.maxPoolWithArgmax(x, 2, 1, 0);

    expect(result.shape).toEqual([1, 2, 2, 1]);
    expectArraysClose(await result.data(), [4, 5, 7, 9]);
    expect(indexes.shape).toEqual([1, 2, 2, 1]);
    expectArraysClose(await indexes.data(), [4, 5, 7, 8]);
  });

  it('x=[1, 3,3,2] f=[2,2] s=1', async () => {
    // Feed forward.
    const x = tf.tensor4d(
        [1, 99, 2, 88, 3, 77, 4, 66, 5, 55, 6, 44, 7, 33, 9, 22, 8, 11],
        [1, 3, 3, 2]);

    const {result, indexes} = tf.maxPoolWithArgmax(x, 2, 1, 0);

    expect(result.shape).toEqual([1, 2, 2, 2]);
    expectArraysClose(await result.data(), [5, 99, 6, 88, 9, 66, 9, 55]);
    expect(indexes.shape).toEqual([1, 2, 2, 2]);
    expectArraysClose(await indexes.data(), [8, 1, 10, 3, 14, 7, 14, 9]);
  });

  it('x=[1,4,4,1] f=[2,2] s=2', async () => {
    // Feed forward.
    const x = tf.tensor4d(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [1, 4, 4, 1]);

    const {result, indexes} = tf.maxPoolWithArgmax(x, 2, 2, 0);

    expect(result.shape).toEqual([1, 2, 2, 1]);
    expectArraysClose(await result.data(), [5, 7, 13, 15]);
    expect(indexes.shape).toEqual([1, 2, 2, 1]);
    expectArraysClose(await indexes.data(), [5, 7, 13, 15]);
  });

  it('x=[1,2,2,1] f=[2,2] s=1 p=same', async () => {
    // Feed forward.
    const x = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);

    const {result, indexes} = tf.maxPoolWithArgmax(x, 2, 1, 'same');
    expect(result.shape).toEqual([1, 2, 2, 1]);
    expectArraysClose(await result.data(), [4, 4, 4, 4]);
    expect(indexes.shape).toEqual([1, 2, 2, 1]);
    expectArraysClose(await indexes.data(), [3, 3, 3, 3]);
  });

  it('throws when x is not rank 4', () => {
    // tslint:disable-next-line:no-any
    const x: any = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3]);

    expect(() => tf.maxPoolWithArgmax(x, 2, 1, 0)).toThrowError();
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.maxPoolWithArgmax({} as tf.Tensor4D, 2, 1, 'valid'))
        .toThrowError(
            /Argument 'x' passed to 'maxPoolWithArgmax' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const x = [[[[0]]]];  // 1x1x1
    const {result, indexes} = tf.maxPoolWithArgmax(x, 1, 1, 0);
    expectArraysClose(await result.data(), [0]);
    expect(indexes.shape).toEqual([1, 1, 1, 1]);
    expectArraysClose(await indexes.data(), [0]);
  });
});
