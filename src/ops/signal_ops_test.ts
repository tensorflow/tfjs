/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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

describeWithFlags('hannWindow', ALL_ENVS, () => {
  it('length=3', async () => {
    const ret = tf.signal.hannWindow(3);
    expectArraysClose(await ret.data(), [0, 1, 0]);
  });

  it('length=7', async () => {
    const ret = tf.signal.hannWindow(7);
    expectArraysClose(await ret.data(), [0, 0.25, 0.75, 1, 0.75, 0.25, 0]);
  });

  it('length=6', async () => {
    const ret = tf.signal.hannWindow(6);
    expectArraysClose(await ret.data(), [0., 0.25, 0.75, 1., 0.75, 0.25]);
  });

  it('length=20', async () => {
    const ret = tf.signal.hannWindow(20);
    expectArraysClose(await ret.data(), [
      0.,  0.02447176, 0.09549153, 0.20610738, 0.34549153,
      0.5, 0.65450853, 0.79389274, 0.9045085,  0.97552824,
      1.,  0.97552824, 0.9045085,  0.7938925,  0.65450835,
      0.5, 0.34549144, 0.20610726, 0.09549153, 0.02447173
    ]);
  });
});

describeWithFlags('hammingWindow', ALL_ENVS, () => {
  it('length=3', async () => {
    const ret = tf.signal.hammingWindow(3);
    expectArraysClose(await ret.data(), [0.08, 1, 0.08]);
  });

  it('length=6', async () => {
    const ret = tf.signal.hammingWindow(6);
    expectArraysClose(await ret.data(), [0.08, 0.31, 0.77, 1., 0.77, 0.31]);
  });

  it('length=7', async () => {
    const ret = tf.signal.hammingWindow(7);
    expectArraysClose(
        await ret.data(), [0.08, 0.31, 0.77, 1, 0.77, 0.31, 0.08]);
  });

  it('length=20', async () => {
    const ret = tf.signal.hammingWindow(20);
    expectArraysClose(await ret.data(), [
      0.08000001, 0.10251403, 0.16785222, 0.2696188,  0.3978522,
      0.54,       0.68214786, 0.8103813,  0.9121479,  0.977486,
      1.,         0.977486,   0.9121478,  0.8103812,  0.6821477,
      0.54,       0.39785212, 0.2696187,  0.16785222, 0.102514
    ]);
  });
});

describeWithFlags('frame', ALL_ENVS, () => {
  it('3 length frames', async () => {
    const input = tf.tensor1d([1, 2, 3, 4, 5]);
    const output = tf.signal.frame(input, 3, 1);
    expect(output.shape).toEqual([3, 3]);
    expectArraysClose(await output.data(), [1, 2, 3, 2, 3, 4, 3, 4, 5]);
  });

  it('3 length frames with step 2', async () => {
    const input = tf.tensor1d([1, 2, 3, 4, 5]);
    const output = tf.signal.frame(input, 3, 2);
    expect(output.shape).toEqual([2, 3]);
    expectArraysClose(await output.data(), [1, 2, 3, 3, 4, 5]);
  });

  it('3 length frames with step 5', async () => {
    const input = tf.tensor1d([1, 2, 3, 4, 5]);
    const output = tf.signal.frame(input, 3, 5);
    expect(output.shape).toEqual([1, 3]);
    expectArraysClose(await output.data(), [1, 2, 3]);
  });

  it('Exceeding frame length', async () => {
    const input = tf.tensor1d([1, 2, 3, 4, 5]);
    const output = tf.signal.frame(input, 6, 1);
    expect(output.shape).toEqual([0, 6]);
    expectArraysClose(await output.data(), []);
  });

  it('Zero frame step', async () => {
    const input = tf.tensor1d([1, 2, 3, 4, 5]);
    const output = tf.signal.frame(input, 6, 0);
    expect(output.shape).toEqual([0, 6]);
    expectArraysClose(await output.data(), []);
  });

  it('Padding with default value', async () => {
    const input = tf.tensor1d([1, 2, 3, 4, 5]);
    const output = tf.signal.frame(input, 3, 3, true);
    expect(output.shape).toEqual([2, 3]);
    expectArraysClose(await output.data(), [1, 2, 3, 4, 5, 0]);
  });

  it('Padding with the given value', async () => {
    const input = tf.tensor1d([1, 2, 3, 4, 5]);
    const output = tf.signal.frame(input, 3, 3, true, 100);
    expect(output.shape).toEqual([2, 3]);
    expectArraysClose(await output.data(), [1, 2, 3, 4, 5, 100]);
  });
});
