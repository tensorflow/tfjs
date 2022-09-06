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

describeWithFlags('fill', ALL_ENVS, () => {
  it('1D fill', async () => {
    const a = tf.fill([3], 2);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    expectArraysClose(await a.data(), [2, 2, 2]);
  });

  it('1D fill with inf', async () => {
    const a = tf.fill([3], Number.POSITIVE_INFINITY);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    expectArraysClose(await a.data(), [
      Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY,
      Number.POSITIVE_INFINITY
    ]);
  });

  it('1D fill string', async () => {
    const a = tf.fill([3], 'aa');
    expect(a.dtype).toBe('string');
    expect(a.shape).toEqual([3]);
    expectArraysEqual(await a.data(), ['aa', 'aa', 'aa']);
  });

  it('2D fill', async () => {
    const a = tf.fill([3, 2], 2);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2]);
    expectArraysClose(await a.data(), [2, 2, 2, 2, 2, 2]);
  });

  it('2D fill string', async () => {
    const a = tf.fill([3, 2], 'a');
    expect(a.dtype).toBe('string');
    expect(a.shape).toEqual([3, 2]);
    expectArraysEqual(await a.data(), ['a', 'a', 'a', 'a', 'a', 'a']);
  });

  it('3D fill', async () => {
    const a = tf.fill([3, 2, 1], 2);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2, 1]);
    expectArraysClose(await a.data(), [2, 2, 2, 2, 2, 2]);
  });

  it('4D fill', async () => {
    const a = tf.fill([3, 2, 1, 2], 2);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2, 1, 2]);
    expectArraysClose(await a.data(), [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]);
  });

  it('5D fill', async () => {
    const a = tf.fill([2, 1, 2, 1, 2], 2);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 1, 2, 1, 2]);
    expectArraysClose(await a.data(), [2, 2, 2, 2, 2, 2, 2, 2]);
  });
});
