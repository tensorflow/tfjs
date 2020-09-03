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

describeWithFlags('stft', ALL_ENVS, () => {
  it('3 length with hann window', async () => {
    const input = tf.tensor1d([1, 1, 1, 1, 1]);
    const frameLength = 3;
    const frameStep = 1;
    const output = tf.signal.stft(input, frameLength, frameStep);
    expect(output.shape).toEqual([3, 3]);
    expectArraysClose(await output.data(), [
      1.0,
      0.0,
      0.0,
      -1.0,
      -1.0,
      0.0,
      1.0,
      0.0,
      0.0,
      -1.0,
      -1.0,
      0.0,
      1.0,
      0.0,
      0.0,
      -1.0,
      -1.0,
      0.0,
    ]);
  });

  it('3 length with hann window (sequencial number)', async () => {
    const input = tf.tensor1d([1, 2, 3, 4, 5]);
    const frameLength = 3;
    const frameStep = 1;
    const output = tf.signal.stft(input, frameLength, frameStep);
    expect(output.shape).toEqual([3, 3]);
    expectArraysClose(await output.data(), [
      2.0, 0.0, 0.0, -2.0, -2.0, 0.0, 3.0, 0.0, 0.0, -3.0, -3.0, 0.0, 4.0, 0.0,
      0.0, -4.0, -4.0, 0.0
    ]);
  });

  it('3 length, 2 step with hann window', async () => {
    const input = tf.tensor1d([1, 1, 1, 1, 1]);
    const frameLength = 3;
    const frameStep = 2;
    const output = tf.signal.stft(input, frameLength, frameStep);
    expect(output.shape).toEqual([2, 3]);
    expectArraysClose(
        await output.data(),
        [1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0]);
  });

  it('3 fftLength, 5 frameLength, 2 step', async () => {
    const input = tf.tensor1d([1, 1, 1, 1, 1, 1]);
    const frameLength = 5;
    const frameStep = 1;
    const fftLength = 3;
    const output = tf.signal.stft(input, frameLength, frameStep, fftLength);
    expect(output.shape[0]).toEqual(2);
    expectArraysClose(
        await output.data(),
        [1.5, 0.0, -0.749999, 0.433, 1.5, 0.0, -0.749999, 0.433]);
  });

  it('5 length with hann window', async () => {
    const input = tf.tensor1d([1, 1, 1, 1, 1]);
    const frameLength = 5;
    const frameStep = 1;
    const output = tf.signal.stft(input, frameLength, frameStep);
    expect(output.shape).toEqual([1, 5]);
    expectArraysClose(
        await output.data(),
        [2.0, 0.0, 0.0, -1.7071068, -1.0, 0.0, 0.0, 0.29289323, 0.0, 0.0]);
  });

  it('5 length with hann window (sequential)', async () => {
    const input = tf.tensor1d([1, 2, 3, 4, 5]);
    const frameLength = 5;
    const frameStep = 1;
    const output = tf.signal.stft(input, frameLength, frameStep);
    expect(output.shape).toEqual([1, 5]);
    expectArraysClose(await output.data(), [
      6.0, 0.0, -0.70710677, -5.1213202, -3.0, 1.0, 0.70710677, 0.87867975, 0.0,
      0.0
    ]);
  });

  it('3 length with hamming window', async () => {
    const input = tf.tensor1d([1, 1, 1, 1, 1]);
    const frameLength = 3;
    const frameStep = 1;
    const fftLength = 3;
    const output = tf.signal.stft(
        input, frameLength, frameStep, fftLength,
        (length) => tf.signal.hammingWindow(length));
    expect(output.shape).toEqual([3, 2]);
    expectArraysClose(await output.data(), [
      1.16, 0.0, -0.46, -0.79674333, 1.16, 0.0, -0.46, -0.79674333, 1.16, 0.0,
      -0.46, -0.79674333
    ]);
  });

  it('3 length, 2 step with hamming window', async () => {
    const input = tf.tensor1d([1, 1, 1, 1, 1]);
    const frameLength = 3;
    const frameStep = 2;
    const fftLength = 3;
    const output = tf.signal.stft(
        input, frameLength, frameStep, fftLength,
        (length) => tf.signal.hammingWindow(length));
    expect(output.shape).toEqual([2, 2]);
    expectArraysClose(
        await output.data(),
        [1.16, 0.0, -0.46, -0.79674333, 1.16, 0.0, -0.46, -0.79674333]);
  });

  it('3 fftLength, 5 frameLength, 2 step with hamming window', async () => {
    const input = tf.tensor1d([1, 1, 1, 1, 1, 1]);
    const frameLength = 5;
    const frameStep = 1;
    const fftLength = 3;
    const output = tf.signal.stft(
        input, frameLength, frameStep, fftLength,
        (length) => tf.signal.hammingWindow(length));
    expect(output.shape).toEqual([2, 2]);
    expectArraysClose(
        await output.data(),
        [1.619999, 0.0, -0.69, 0.39837, 1.619999, 0.0, -0.69, 0.39837]);
  });

  it('5 length with hann window (sequential)', async () => {
    const input = tf.tensor1d([1, 2, 3, 4, 5]);
    const frameLength = 5;
    const frameStep = 1;
    const fftLength = 5;
    const output = tf.signal.stft(
        input, frameLength, frameStep, fftLength,
        (length) => tf.signal.hammingWindow(length));
    expect(output.shape).toEqual([1, 3]);
    expectArraysClose(
        await output.data(),
        [6.72, 0.0, -3.6371822, -1.1404576, 0.4771822, 0.39919350]);
  });

  it('3 length without window function', async () => {
    const input = tf.tensor1d([1, 1, 1, 1, 1]);
    const frameLength = 3;
    const frameStep = 1;
    const fftLength = 3;
    const ident = (length: number) => tf.ones([length]).as1D();
    const output =
        tf.signal.stft(input, frameLength, frameStep, fftLength, ident);
    expect(output.shape).toEqual([3, 2]);
    expectArraysClose(
        await output.data(),
        [3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0]);
  });

  it('3 length, 2 step without window function', async () => {
    const input = tf.tensor1d([1, 1, 1, 1, 1]);
    const frameLength = 3;
    const frameStep = 2;
    const fftLength = 3;
    const ident = (length: number) => tf.ones([length]).as1D();
    const output =
        tf.signal.stft(input, frameLength, frameStep, fftLength, ident);
    expect(output.shape).toEqual([2, 2]);
    expectArraysClose(
        await output.data(), [3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0]);
  });
});
