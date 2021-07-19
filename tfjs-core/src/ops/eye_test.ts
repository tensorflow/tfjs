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

describeWithFlags('eye', ALL_ENVS, () => {
  it('1x1', async () => {
    const r = tf.eye(1);
    expectArraysClose(await r.data(), [1]);
    expect(r.shape).toEqual([1, 1]);
    expect(r.dtype).toBe('float32');
  });

  it('2x2', async () => {
    const r = tf.eye(2);
    expect(r.shape).toEqual([2, 2]);
    expect(r.dtype).toBe('float32');
    expectArraysClose(await r.data(), [1, 0, 0, 1]);
  });

  it('3x3', async () => {
    const r = tf.eye(3);
    expect(r.shape).toEqual([3, 3]);
    expect(r.dtype).toBe('float32');
    expectArraysClose(await r.data(), [1, 0, 0, 0, 1, 0, 0, 0, 1]);
  });

  it('3x4', async () => {
    const r = tf.eye(3, 4);
    expect(r.shape).toEqual([3, 4]);
    expect(r.dtype).toBe('float32');
    expectArraysClose(await r.data(), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]);
  });

  it('4x3', async () => {
    const r = tf.eye(4, 3);
    expect(r.shape).toEqual([4, 3]);
    expectArraysClose(await r.data(), [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
  });

  it('with 1D batchShape', async () => {
    const r = tf.eye(2, 2, [3]);
    expect(r.shape).toEqual([3, 2, 2]);
    expectArraysClose(await r.data(), [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]);
  });

  it('with 2D batchShape', async () => {
    const r = tf.eye(2, 2, [2, 3]);
    expect(r.shape).toEqual([2, 3, 2, 2]);
    expectArraysClose(await r.data(), [
      1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1
    ]);
  });

  it('with 3D batchShape', async () => {
    const r = tf.eye(2, 2, [2, 2, 3]);
    expect(r.shape).toEqual([2, 2, 3, 2, 2]);
    expectArraysClose(await r.data(), [
      1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1,
      1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1
    ]);
  });

  it('3x3, int32', async () => {
    const r = tf.eye(3, 3, null, 'int32');
    expect(r.dtype).toBe('int32');
    expect(r.shape).toEqual([3, 3]);
    expectArraysClose(await r.data(), [1, 0, 0, 0, 1, 0, 0, 0, 1]);
  });

  it('3x3, bool', async () => {
    const r = tf.eye(3, 3, null, 'bool');
    expect(r.dtype).toBe('bool');
    expect(r.shape).toEqual([3, 3]);
    expectArraysClose(await r.data(), [1, 0, 0, 0, 1, 0, 0, 0, 1]);
  });
});
