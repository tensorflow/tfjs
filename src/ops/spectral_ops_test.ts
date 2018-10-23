/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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
import {describeWithFlags} from '../jasmine_util';
import {ALL_ENVS, BROWSER_CPU_ENVS, expectArraysClose, WEBGL_ENVS} from '../test_util';

describeWithFlags('1D FFT', ALL_ENVS, () => {
  it('should return the same value with TensorFlow (2 elements)', () => {
    const t1Real = tf.tensor1d([1, 2]);
    const t1Imag = tf.tensor1d([1, 1]);
    const t1 = tf.complex(t1Real, t1Imag);
    expectArraysClose(tf.spectral.fft(t1), [3, 2, -1, 0]);
  });

  it('should calculate FFT from Tensor directly', () => {
    const t1Real = tf.tensor1d([1, 2]);
    const t1Imag = tf.tensor1d([1, 1]);
    const t1 = tf.complex(t1Real, t1Imag);
    expectArraysClose(t1.fft(), [3, 2, -1, 0]);
  });

  it('should return the same value as TensorFlow (3 elements)', () => {
    const t1Real = tf.tensor1d([1, 2, 3]);
    const t1Imag = tf.tensor1d([0, 0, 0]);
    const t1 = tf.complex(t1Real, t1Imag);
    expectArraysClose(
        tf.spectral.fft(t1), [6, 0, -1.5, 0.866025, -1.5, -0.866025]);
  });

  it('should return the same value as TensorFlow with imaginary (3 elements)',
     () => {
       const t1Real = tf.tensor1d([1, 2, 3]);
       const t1Imag = tf.tensor1d([1, 2, 3]);
       const t1 = tf.complex(t1Real, t1Imag);
       expectArraysClose(
           tf.spectral.fft(t1),
           [6, 6, -2.3660252, -0.63397473, -0.6339747, -2.3660254]);
     });

  it('should return the same value as TensorFlow (negative 3 elements)', () => {
    const t1Real = tf.tensor1d([-1, -2, -3]);
    const t1Imag = tf.tensor1d([-1, -2, -3]);
    const t1 = tf.complex(t1Real, t1Imag);
    expectArraysClose(
        tf.spectral.fft(t1),
        [-5.9999995, -6, 2.3660252, 0.63397473, 0.6339747, 2.3660254]);
  });

  it('should return the same value with TensorFlow (4 elements)', () => {
    const t1Real = tf.tensor1d([1, 2, 3, 4]);
    const t1Imag = tf.tensor1d([0, 0, 0, 0]);
    const t1 = tf.complex(t1Real, t1Imag);
    expectArraysClose(tf.spectral.fft(t1), [10, 0, -2, 2, -2, 0, -2, -2]);
  });

  it('should return the same value as TensorFlow with imaginary (4 elements)',
     () => {
       const t1Real = tf.tensor1d([1, 2, 3, 4]);
       const t1Imag = tf.tensor1d([1, 2, 3, 4]);
       const t1 = tf.complex(t1Real, t1Imag);
       expectArraysClose(tf.spectral.fft(t1), [10, 10, -4, 0, -2, -2, 0, -4]);
     });
});

describeWithFlags('FFT', WEBGL_ENVS, () => {
  it('2D: should return the same value as TensorFlow', () => {
    const t1Real = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const t1Imag = tf.tensor2d([5, 6, 7, 8], [2, 2]);
    const t1 = tf.complex(t1Real, t1Imag);
    const y = tf.spectral.fft(t1);
    expectArraysClose(y, [3, 11, -1, -1, 7, 15, -1, -1]);
    expect(y.shape).toEqual(t1Real.shape);
  });

  it('3D: should return the same value as TensorFlow', () => {
    const t1Real = tf.tensor3d([1, 2, 3, 4, -1, -2, -3, -4], [2, 2, 2]);
    const t1Imag = tf.tensor3d([5, 6, 7, 8, -5, -6, -7, -8], [2, 2, 2]);
    const t1 = tf.complex(t1Real, t1Imag);
    const y = tf.spectral.fft(t1);
    expectArraysClose(
        y, [3, 11, -1, -1, 7, 15, -1, -1, -3, -11, 1, 1, -7, -15, 1, 1]);
    expect(y.shape).toEqual(t1Real.shape);
  });
});

// TODO: Remove this once we support higher-dimensional FFTs on CPU.
describeWithFlags('FFT CPU', BROWSER_CPU_ENVS, () => {
  it('2D throws', () => {
    const t1Real = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const t1Imag = tf.tensor2d([5, 6, 7, 8], [2, 2]);
    const t1 = tf.complex(t1Real, t1Imag);
    expect(() => tf.spectral.fft(t1)).toThrow();
  });

  it('3D throws', () => {
    const t1Real = tf.tensor3d([1, 2, 3, 4, -1, -2, -3, -4], [2, 2, 2]);
    const t1Imag = tf.tensor3d([5, 6, 7, 8, -5, -6, -7, -8], [2, 2, 2]);
    const t1 = tf.complex(t1Real, t1Imag);
    expect(() => tf.spectral.fft(t1)).toThrow();
  });
});
