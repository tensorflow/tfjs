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

describeWithFlags('linspace', ALL_ENVS, () => {
  it('start stop', async () => {
    const a = tf.linspace(1, 10, 10);
    expectArraysEqual(
        await a.data(), [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
    expect(a.shape).toEqual([10]);

    const b = tf.linspace(12, 17, 8);
    expectArraysClose(await b.data(), [
      12., 12.71428571, 13.42857143, 14.14285714, 14.85714286, 15.57142857,
      16.28571429, 17.
    ]);
    expect(b.shape).toEqual([8]);

    const c = tf.linspace(9, 0, 6);
    expectArraysClose(await c.data(), [9., 7.2, 5.4, 3.6, 1.8, 0.]);
    expect(c.shape).toEqual([6]);
  });

  it('negative start stop', async () => {
    const a = tf.linspace(-4, 5, 6);
    expectArraysClose(await a.data(), [-4., -2.2, -0.4, 1.4, 3.2, 5.]);
    expect(a.shape).toEqual([6]);
  });

  it('start negative stop', async () => {
    const a = tf.linspace(4, -5, 6);
    expectArraysClose(await a.data(), [4., 2.2, 0.4, -1.4, -3.2, -5.]);
    expect(a.shape).toEqual([6]);
  });

  it('negative start negative stop', async () => {
    const a = tf.linspace(-4, -5, 6);
    expectArraysClose(await a.data(), [-4., -4.2, -4.4, -4.6, -4.8, -5.]);
    expect(a.shape).toEqual([6]);

    const b = tf.linspace(-9, -4, 5);
    expectArraysClose(await b.data(), [-9., -7.75, -6.5, -5.25, -4.]);
    expect(b.shape).toEqual([5]);
  });

  it('should throw with no samples', () => {
    expect(() => tf.linspace(2, 10, 0)).toThrow();
  });
});
