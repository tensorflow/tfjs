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
import {expectArraysClose, expectArraysEqual} from '../test_util';

describeWithFlags('zeros', ALL_ENVS, () => {
  it('1D default dtype', async () => {
    const a: tf.Tensor1D = tf.zeros([3]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    expectArraysClose(await a.data(), [0, 0, 0]);
  });

  it('1D float32 dtype', async () => {
    const a = tf.zeros([3], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    expectArraysClose(await a.data(), [0, 0, 0]);
  });

  it('1D int32 dtype', async () => {
    const a = tf.zeros([3], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3]);
    expectArraysEqual(await a.data(), [0, 0, 0]);
  });

  it('1D bool dtype', async () => {
    const a = tf.zeros([3], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([3]);
    expectArraysEqual(await a.data(), [0, 0, 0]);
  });

  it('2D default dtype', async () => {
    const a = tf.zeros([3, 2]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2]);
    expectArraysClose(await a.data(), [0, 0, 0, 0, 0, 0]);
  });

  it('2D float32 dtype', async () => {
    const a = tf.zeros([3, 2], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2]);
    expectArraysClose(await a.data(), [0, 0, 0, 0, 0, 0]);
  });

  it('2D int32 dtype', async () => {
    const a = tf.zeros([3, 2], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3, 2]);
    expectArraysEqual(await a.data(), [0, 0, 0, 0, 0, 0]);
  });

  it('2D bool dtype', async () => {
    const a = tf.zeros([3, 2], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([3, 2]);
    expectArraysEqual(await a.data(), [0, 0, 0, 0, 0, 0]);
  });

  it('3D default dtype', async () => {
    const a = tf.zeros([2, 2, 2]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 2]);
    expectArraysClose(await a.data(), [0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it('3D float32 dtype', async () => {
    const a = tf.zeros([2, 2, 2], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 2]);
    expectArraysClose(await a.data(), [0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it('3D int32 dtype', async () => {
    const a = tf.zeros([2, 2, 2], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 2]);
    expectArraysEqual(await a.data(), [0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it('3D bool dtype', async () => {
    const a = tf.zeros([2, 2, 2], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([2, 2, 2]);
    expectArraysEqual(await a.data(), [0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it('4D default dtype', async () => {
    const a = tf.zeros([3, 2, 1, 1]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    expectArraysClose(await a.data(), [0, 0, 0, 0, 0, 0]);
  });

  it('4D float32 dtype', async () => {
    const a = tf.zeros([3, 2, 1, 1], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    expectArraysClose(await a.data(), [0, 0, 0, 0, 0, 0]);
  });

  it('4D int32 dtype', async () => {
    const a = tf.zeros([3, 2, 1, 1], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    expectArraysEqual(await a.data(), [0, 0, 0, 0, 0, 0]);
  });

  it('4D bool dtype', async () => {
    const a = tf.zeros([3, 2, 1, 1], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    expectArraysEqual(await a.data(), [0, 0, 0, 0, 0, 0]);
  });
});
