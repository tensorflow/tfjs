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
import {expectArraysEqual} from '../test_util';

import {tensor1d} from './tensor1d';

describeWithFlags('unique', ALL_ENVS, () => {
  it('1d tensor with int32', async () => {
    const x = tensor1d([1, 1, 2, 4, 4, 4, 7, 8, 8]);
    const {values, indices} = tf.unique(x);

    expect(indices.dtype).toBe('int32');
    expect(indices.shape).toEqual(x.shape);
    expect(values.shape).toEqual([5]);
    expectArraysEqual(await values.data(), [1, 2, 4, 7, 8]);
    expectArraysEqual(await indices.data(), [0, 0, 1, 2, 2, 2, 3, 4, 4]);
  });

  it('1d tensor with string', async () => {
    const x = tensor1d(['a', 'b', 'b', 'c', 'c']);
    const {values, indices} = tf.unique(x);

    expect(indices.dtype).toBe('int32');
    expect(indices.shape).toEqual(x.shape);
    expect(values.dtype).toEqual('string');
    expect(values.shape).toEqual([3]);
    expectArraysEqual(await values.data(), ['a', 'b', 'c']);
    expectArraysEqual(await indices.data(), [0, 1, 1, 2, 2]);
  });

  it('1d tensor with bool', async () => {
    const x = tensor1d([true, true, false]);
    const {values, indices} = tf.unique(x);

    expect(indices.dtype).toBe('int32');
    expect(indices.shape).toEqual(x.shape);
    expect(values.dtype).toEqual('bool');
    expect(values.shape).toEqual([2]);
    expectArraysEqual(await values.data(), [true, false]);
    expectArraysEqual(await indices.data(), [0, 0, 1]);
  });

  it('1d tensor with NaN and Infinity', async () => {
    const x = tensor1d([NaN, Infinity, NaN, Infinity]);
    const {values, indices} = tf.unique(x);

    expect(indices.dtype).toBe('int32');
    expect(indices.shape).toEqual(x.shape);
    expect(values.shape).toEqual([2]);
    expectArraysEqual(await values.data(), [NaN, Infinity]);
    expectArraysEqual(await indices.data(), [0, 1, 0, 1]);
  });

  it('2d tensor with axis=0', async () => {
    const x = tf.tensor2d([[1, 0, 0], [1, 0, 0], [2, 0, 0]]);
    const {values, indices} = tf.unique(x, 0);

    expect(indices.dtype).toBe('int32');
    expect(indices.shape).toEqual([x.shape[0]]);
    expect(values.shape).toEqual([2, 3]);
    expectArraysEqual(await values.data(), [1, 0, 0, 2, 0, 0]);
    expectArraysEqual(await indices.data(), [0, 0, 1]);
  });

  it('2d tensor with axis=1', async () => {
    const x = tf.tensor2d([[1, 0, 0, 1], [1, 0, 0, 1], [2, 0, 0, 2]]);
    const {values, indices} = tf.unique(x, 1);

    expect(indices.dtype).toBe('int32');
    expect(indices.shape).toEqual([x.shape[1]]);
    expect(values.shape).toEqual([3, 2]);
    expectArraysEqual(await values.data(), [[1, 0], [1, 0], [2, 0]]);
    expectArraysEqual(await indices.data(), [0, 1, 1, 0]);
  });

  it('2d tensor with string', async () => {
    const x = tf.tensor2d([['a', 'b', 'b'], ['a', 'b', 'b'], ['c', 'b', 'b']]);
    const {values, indices} = tf.unique(x, 0);

    expect(indices.dtype).toBe('int32');
    expect(indices.shape).toEqual([x.shape[0]]);
    expect(values.dtype).toEqual('string');
    expect(values.shape).toEqual([2, 3]);
    expectArraysEqual(await values.data(), ['a', 'b', 'b', 'c', 'b', 'b']);
    expectArraysEqual(await indices.data(), [0, 0, 1]);
  });

  it('2d tensor with strings that have comma', async () => {
    const x = tf.tensor2d([['a', 'b,c', 'd'], ['a', 'b', 'c,d']]);
    const {values, indices} = tf.unique(x, 0);

    expect(indices.dtype).toBe('int32');
    expect(indices.shape).toEqual([x.shape[0]]);
    expect(values.dtype).toEqual('string');
    expect(values.shape).toEqual([2, 3]);
    expectArraysEqual(await values.data(), ['a', 'b,c', 'd', 'a', 'b', 'c,d']);
    expectArraysEqual(await indices.data(), [0, 1]);
  });

  it('3d tensor with axis=0', async () => {
    const x =
        tf.tensor3d([[[1, 0], [1, 0]], [[1, 0], [1, 0]], [[1, 1], [1, 1]]]);
    const {values, indices} = tf.unique(x, 0);

    expect(indices.dtype).toBe('int32');
    expect(indices.shape).toEqual([x.shape[0]]);
    expect(values.shape).toEqual([2, 2, 2]);
    expectArraysEqual(await values.data(), [1, 0, 1, 0, 1, 1, 1, 1]);
    expectArraysEqual(await indices.data(), [0, 0, 1]);
  });

  it('3d tensor with axis=1', async () => {
    const x =
        tf.tensor3d([[[1, 0], [1, 0]], [[1, 0], [1, 0]], [[1, 1], [1, 1]]]);
    const {values, indices} = tf.unique(x, 1);

    expect(indices.dtype).toBe('int32');
    expect(indices.shape).toEqual([x.shape[1]]);
    expect(values.shape).toEqual([3, 1, 2]);
    expectArraysEqual(await values.data(), [[[1, 0]], [[1, 0]], [[1, 1]]]);
    expectArraysEqual(await indices.data(), [0, 0]);
  });

  it('3d tensor with axis=2', async () => {
    const x = tf.tensor3d([[[1, 0, 1]], [[1, 0, 1]]]);
    const {values, indices} = tf.unique(x, 2);

    expect(indices.dtype).toBe('int32');
    expect(indices.shape).toEqual([x.shape[2]]);
    expect(values.shape).toEqual([2, 1, 2]);
    expectArraysEqual(await values.data(), [1, 0, 1, 0]);
    expectArraysEqual(await indices.data(), [0, 1, 0]);
  });

  it('3d tensor with string', async () => {
    const x = tf.tensor3d([
      [['a', 'b'], ['a', 'b']], [['a', 'b'], ['a', 'b']],
      [['a', 'a'], ['a', 'a']]
    ]);
    const {values, indices} = tf.unique(x, 0);

    expect(indices.dtype).toBe('int32');
    expect(indices.shape).toEqual([x.shape[0]]);
    expect(values.dtype).toEqual('string');
    expect(values.shape).toEqual([2, 2, 2]);
    expectArraysEqual(
        await values.data(), ['a', 'b', 'a', 'b', 'a', 'a', 'a', 'a']);
    expectArraysEqual(await indices.data(), [0, 0, 1]);
  });
});
