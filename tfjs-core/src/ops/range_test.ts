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

describeWithFlags('range', ALL_ENVS, () => {
  it('start stop', async () => {
    const a = tf.range(0, 3);
    expectArraysEqual(await a.data(), [0, 1, 2]);
    expect(a.shape).toEqual([3]);

    const b = tf.range(3, 8);
    expectArraysEqual(await b.data(), [3, 4, 5, 6, 7]);
    expect(b.shape).toEqual([5]);
  });

  it('start stop negative', async () => {
    const a = tf.range(-2, 3);
    expectArraysEqual(await a.data(), [-2, -1, 0, 1, 2]);
    expect(a.shape).toEqual([5]);

    const b = tf.range(4, -2);
    expectArraysEqual(await b.data(), [4, 3, 2, 1, 0, -1]);
    expect(b.shape).toEqual([6]);
  });

  it('start stop step', async () => {
    const a = tf.range(4, 15, 4);
    expectArraysEqual(await a.data(), [4, 8, 12]);
    expect(a.shape).toEqual([3]);

    const b = tf.range(4, 11, 4);
    expectArraysEqual(await b.data(), [4, 8]);
    expect(b.shape).toEqual([2]);

    const c = tf.range(4, 17, 4);
    expectArraysEqual(await c.data(), [4, 8, 12, 16]);
    expect(c.shape).toEqual([4]);

    const d = tf.range(0, 30, 5);
    expectArraysEqual(await d.data(), [0, 5, 10, 15, 20, 25]);
    expect(d.shape).toEqual([6]);

    const e = tf.range(-3, 9, 2);
    expectArraysEqual(await e.data(), [-3, -1, 1, 3, 5, 7]);
    expect(e.shape).toEqual([6]);

    const f = tf.range(3, 3);
    expectArraysEqual(await f.data(), new Float32Array(0));
    expect(f.shape).toEqual([0]);

    const g = tf.range(3, 3, 1);
    expectArraysEqual(await g.data(), new Float32Array(0));
    expect(g.shape).toEqual([0]);

    const h = tf.range(3, 3, 4);
    expectArraysEqual(await h.data(), new Float32Array(0));
    expect(h.shape).toEqual([0]);

    const i = tf.range(-18, -2, 5);
    expectArraysEqual(await i.data(), [-18, -13, -8, -3]);
    expect(i.shape).toEqual([4]);
  });

  it('start stop large step', async () => {
    const a = tf.range(3, 10, 150);
    expectArraysEqual(await a.data(), [3]);
    expect(a.shape).toEqual([1]);

    const b = tf.range(10, 500, 205);
    expectArraysEqual(await b.data(), [10, 215, 420]);
    expect(b.shape).toEqual([3]);

    const c = tf.range(3, -10, -150);
    expectArraysEqual(await c.data(), [3]);
    expect(c.shape).toEqual([1]);

    const d = tf.range(-10, -500, -205);
    expectArraysEqual(await d.data(), [-10, -215, -420]);
    expect(d.shape).toEqual([3]);
  });

  it('start stop negative step', async () => {
    const a = tf.range(0, -10, -1);
    expectArraysEqual(await a.data(), [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]);
    expect(a.shape).toEqual([10]);

    const b = tf.range(0, -10);
    expectArraysEqual(await b.data(), [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]);
    expect(b.shape).toEqual([10]);

    const c = tf.range(3, -4, -2);
    expectArraysEqual(await c.data(), [3, 1, -1, -3]);
    expect(c.shape).toEqual([4]);

    const d = tf.range(-3, -18, -5);
    expectArraysEqual(await d.data(), [-3, -8, -13]);
    expect(d.shape).toEqual([3]);
  });

  it('start stop incompatible step', async () => {
    const a = tf.range(3, 10, -2);
    expectArraysEqual(await a.data(), new Float32Array(0));
    expect(a.shape).toEqual([0]);

    const b = tf.range(40, 3, 2);
    expectArraysEqual(await b.data(), new Float32Array(0));
    expect(b.shape).toEqual([0]);
  });

  it('zero step', () => {
    expect(() => tf.range(2, 10, 0)).toThrow();
  });

  it('should have default dtype', async () => {
    const a = tf.range(1, 4);
    expectArraysEqual(await a.data(), [1, 2, 3]);
    expect(a.dtype).toEqual('float32');
    expect(a.shape).toEqual([3]);
  });

  it('should have float32 dtype', async () => {
    const a = tf.range(1, 4, undefined, 'float32');
    expectArraysEqual(await a.data(), [1, 2, 3]);
    expect(a.dtype).toEqual('float32');
    expect(a.shape).toEqual([3]);
  });

  it('should have int32 dtype', async () => {
    const a = tf.range(1, 4, undefined, 'int32');
    expectArraysEqual(await a.data(), [1, 2, 3]);
    expect(a.dtype).toEqual('int32');
    expect(a.shape).toEqual([3]);
  });
});
