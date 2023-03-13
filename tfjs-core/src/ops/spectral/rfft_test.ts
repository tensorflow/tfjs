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

describeWithFlags('1D RFFT', ALL_ENVS, () => {
  it('should return the same value with TensorFlow (3 elements)', async () => {
    const t1Real = tf.tensor1d([1, 2, 3]);
    expectArraysClose(
        await tf.spectral.rfft(t1Real).data(),
        [6, 1.1920929e-07, -1.4999999, 8.6602521e-01]);
  });

  it('should calculate from tensor directly', async () => {
    const t1Real = tf.tensor1d([1, 2, 3]);
    expectArraysClose(
        await t1Real.rfft().data(),
        [6, 1.1920929e-07, -1.4999999, 8.6602521e-01]);
  });

  it('should return the same value with TensorFlow (6 elements)', async () => {
    const t1Real = tf.tensor1d([-3, -2, -1, 1, 2, 3]);
    expectArraysClose(await tf.spectral.rfft(t1Real).data(), [
      -5.8859587e-07, 1.1920929e-07, -3.9999995, 6.9282026e+00, -2.9999998,
      1.7320497, -4.0000000, -2.3841858e-07
    ]);
  });

  it('should return the same value without any fftLength', async () => {
    const t1Real = tf.tensor1d([-3, -2, -1, 1, 2, 3]);
    const fftLength = 6;
    expectArraysClose(await tf.spectral.rfft(t1Real, fftLength).data(), [
      -5.8859587e-07, 1.1920929e-07, -3.9999995, 6.9282026e+00, -2.9999998,
      1.7320497, -4.0000000, -2.3841858e-07
    ]);
  });

  it('should return the value with cropped input', async () => {
    const t1Real = tf.tensor1d([-3, -2, -1, 1, 2, 3]);
    const fftLength = 3;
    expectArraysClose(
        await tf.spectral.rfft(t1Real, fftLength).data(),
        [-6, 0.0, -1.5000002, 0.866]);
  });

  it('should return the value with padded input', async () => {
    const t1Real = tf.tensor1d([-3, -2, -1]);
    const fftLength = 4;
    expectArraysClose(
        await tf.spectral.rfft(t1Real, fftLength).data(),
        [-6, 0, -2, 2, -2, 0]);
  });
});

describeWithFlags('2D RFFT', ALL_ENVS, () => {
  it('should return the same value with TensorFlow (2x2 elements)',
     async () => {
       const t1Real = tf.tensor2d([1, 2, 3, 4], [2, 2]);
       expectArraysClose(
           await tf.spectral.rfft(t1Real).data(), [3, 0, -1, 0, 7, 0, -1, 0]);
     });

  it('should return the same value with TensorFlow (2x3 elements)',
     async () => {
       const t1Real = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
       expectArraysClose(await tf.spectral.rfft(t1Real).data(), [
         6, 1.1920929e-07, -1.4999999, 8.6602521e-01, 15, -5.9604645e-08,
         -1.4999998, 8.6602545e-01
       ]);
     });

  it('should return the same value with TensorFlow (2x2x2 elements)',
     async () => {
       const t1Real = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
       expectArraysClose(
           await tf.spectral.rfft(t1Real).data(),
           [3, 0, -1, 0, 7, 0, -1, 0, 11, 0, -1, 0, 15, 0, -1, 0]);
     });

  it('should return the value with cropping', async () => {
    const t1Real = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const fftLength = 2;
    expectArraysClose(
        await tf.spectral.rfft(t1Real, fftLength).data(),
        [3, 0, -1, 0, 9, 0, -1, 0]);
  });

  it('should return the value with padding', async () => {
    const t1Real = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const fftLength = 4;
    expectArraysClose(
        await tf.spectral.rfft(t1Real, fftLength).data(),
        [6, 0, -2, -2, 2, 0, 15, 0, -2, -5, 5, 0]);
  });
});
