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

import * as tf from '../../index';
import {ALL_ENVS, describeWithFlags} from '../../jasmine_util';
import {expectArraysClose} from '../../test_util';

describeWithFlags('frame', ALL_ENVS, () => {
  it('3 length frames', async () => {
    const input = tf.tensor1d([1, 2, 3, 4, 5]);
    const frameLength = 3;
    const frameStep = 1;
    const output = tf.signal.frame(input, frameLength, frameStep);
    expect(output.shape).toEqual([3, 3]);
    expectArraysClose(await output.data(), [1, 2, 3, 2, 3, 4, 3, 4, 5]);
  });

  it('3 length frames with step 2', async () => {
    const input = tf.tensor1d([1, 2, 3, 4, 5]);
    const frameLength = 3;
    const frameStep = 2;
    const output = tf.signal.frame(input, frameLength, frameStep);
    expect(output.shape).toEqual([2, 3]);
    expectArraysClose(await output.data(), [1, 2, 3, 3, 4, 5]);
  });

  it('3 length frames with step 5', async () => {
    const input = tf.tensor1d([1, 2, 3, 4, 5]);
    const frameLength = 3;
    const frameStep = 5;
    const output = tf.signal.frame(input, frameLength, frameStep);
    expect(output.shape).toEqual([1, 3]);
    expectArraysClose(await output.data(), [1, 2, 3]);
  });

  it('Exceeding frame length', async () => {
    const input = tf.tensor1d([1, 2, 3, 4, 5]);
    const frameLength = 6;
    const frameStep = 1;
    const output = tf.signal.frame(input, frameLength, frameStep);
    expect(output.shape).toEqual([0, 6]);
    expectArraysClose(await output.data(), []);
  });

  it('Zero frame step', async () => {
    const input = tf.tensor1d([1, 2, 3, 4, 5]);
    const frameLength = 6;
    const frameStep = 0;
    const output = tf.signal.frame(input, frameLength, frameStep);
    expect(output.shape).toEqual([0, 6]);
    expectArraysClose(await output.data(), []);
  });

  it('Padding with default value', async () => {
    const input = tf.tensor1d([1, 2, 3, 4, 5]);
    const frameLength = 3;
    const frameStep = 3;
    const padEnd = true;
    const output = tf.signal.frame(input, frameLength, frameStep, padEnd);
    expect(output.shape).toEqual([2, 3]);
    expectArraysClose(await output.data(), [1, 2, 3, 4, 5, 0]);
  });

  it('Padding with the given value', async () => {
    const input = tf.tensor1d([1, 2, 3, 4, 5]);
    const frameLength = 3;
    const frameStep = 3;
    const padEnd = true;
    const padValue = 100;
    const output =
        tf.signal.frame(input, frameLength, frameStep, padEnd, padValue);
    expect(output.shape).toEqual([2, 3]);
    expectArraysClose(await output.data(), [1, 2, 3, 4, 5, 100]);
  });

  it('Padding all remaining frames with step=1', async () => {
    const input = tf.tensor1d([1, 2, 3, 4, 5]);
    const frameLength = 4;
    const frameStep = 1;
    const padEnd = true;
    const output = tf.signal.frame(input, frameLength, frameStep, padEnd);
    expect(output.shape).toEqual([5, 4]);
    expectArraysClose(
        await output.data(),
        [1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 0, 4, 5, 0, 0, 5, 0, 0, 0]);
  });

  it('Padding all remaining frames with step=1 and given pad-value',
     async () => {
       const input = tf.tensor1d([1, 2, 3, 4, 5]);
       const frameLength = 4;
       const frameStep = 1;
       const padEnd = true;
       const padValue = 42;
       const output =
           tf.signal.frame(input, frameLength, frameStep, padEnd, padValue);
       expect(output.shape).toEqual([5, 4]);
       expectArraysClose(
           await output.data(),
           [1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 42, 4, 5, 42, 42, 5, 42, 42, 42]);
     });

  it('Padding all remaining frames with step=2', async () => {
    const input = tf.tensor1d([1, 2, 3, 4, 5]);
    const output = tf.signal.frame(input, 4, 2, true);
    expect(output.shape).toEqual([3, 4]);
    expectArraysClose(
        await output.data(), [1, 2, 3, 4, 3, 4, 5, 0, 5, 0, 0, 0]);
  });
});
