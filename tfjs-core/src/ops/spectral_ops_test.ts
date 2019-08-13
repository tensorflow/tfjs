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
import {ALL_ENVS, describeWithFlags} from '../jasmine_util';
import {expectArraysClose} from '../test_util';

describeWithFlags('1D FFT', ALL_ENVS, () => {
  it('should return the same value with TensorFlow (2 elements)', async () => {
    const t1Real = tf.tensor1d([1, 2]);
    const t1Imag = tf.tensor1d([1, 1]);
    const t1 = tf.complex(t1Real, t1Imag);
    expectArraysClose(await tf.spectral.fft(t1).data(), [3, 2, -1, 0]);
  });

  it('should calculate FFT from Tensor directly', async () => {
    const t1Real = tf.tensor1d([1, 2]);
    const t1Imag = tf.tensor1d([1, 1]);
    const t1 = tf.complex(t1Real, t1Imag);
    expectArraysClose(await t1.fft().data(), [3, 2, -1, 0]);
  });

  it('should return the same value as TensorFlow (3 elements)', async () => {
    const t1Real = tf.tensor1d([1, 2, 3]);
    const t1Imag = tf.tensor1d([0, 0, 0]);
    const t1 = tf.complex(t1Real, t1Imag);
    expectArraysClose(
        await tf.spectral.fft(t1).data(),
        [6, 0, -1.5, 0.866025, -1.5, -0.866025]);
  });

  it('should return the same value as TensorFlow with imaginary (3 elements)',
     async () => {
       const t1Real = tf.tensor1d([1, 2, 3]);
       const t1Imag = tf.tensor1d([1, 2, 3]);
       const t1 = tf.complex(t1Real, t1Imag);
       expectArraysClose(
           await tf.spectral.fft(t1).data(),
           [6, 6, -2.3660252, -0.63397473, -0.6339747, -2.3660254]);
     });

  it('should return the same value as TensorFlow (negative 3 elements)',
     async () => {
       const t1Real = tf.tensor1d([-1, -2, -3]);
       const t1Imag = tf.tensor1d([-1, -2, -3]);
       const t1 = tf.complex(t1Real, t1Imag);
       expectArraysClose(
           await tf.spectral.fft(t1).data(),
           [-5.9999995, -6, 2.3660252, 0.63397473, 0.6339747, 2.3660254]);
     });

  it('should return the same value with TensorFlow (4 elements)', async () => {
    const t1Real = tf.tensor1d([1, 2, 3, 4]);
    const t1Imag = tf.tensor1d([0, 0, 0, 0]);
    const t1 = tf.complex(t1Real, t1Imag);
    expectArraysClose(
        await tf.spectral.fft(t1).data(), [10, 0, -2, 2, -2, 0, -2, -2]);
  });

  it('should return the same value as TensorFlow with imaginary (4 elements)',
     async () => {
       const t1Real = tf.tensor1d([1, 2, 3, 4]);
       const t1Imag = tf.tensor1d([1, 2, 3, 4]);
       const t1 = tf.complex(t1Real, t1Imag);
       expectArraysClose(
           await tf.spectral.fft(t1).data(), [10, 10, -4, 0, -2, -2, 0, -4]);
     });
});

describeWithFlags('2D FFT', ALL_ENVS, () => {
  it('2D: should return the same value as TensorFlow', async () => {
    const t1Real = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const t1Imag = tf.tensor2d([5, 6, 7, 8], [2, 2]);
    const t1 = tf.complex(t1Real, t1Imag);
    const y = tf.spectral.fft(t1);
    expectArraysClose(await y.data(), [3, 11, -1, -1, 7, 15, -1, -1]);
    expect(y.shape).toEqual(t1Real.shape);
  });

  it('3D: should return the same value as TensorFlow', async () => {
    const t1Real = tf.tensor3d([1, 2, 3, 4, -1, -2, -3, -4], [2, 2, 2]);
    const t1Imag = tf.tensor3d([5, 6, 7, 8, -5, -6, -7, -8], [2, 2, 2]);
    const t1 = tf.complex(t1Real, t1Imag);
    const y = tf.spectral.fft(t1);
    expectArraysClose(
        await y.data(),
        [3, 11, -1, -1, 7, 15, -1, -1, -3, -11, 1, 1, -7, -15, 1, 1]);
    expect(y.shape).toEqual(t1Real.shape);
  });
});

describeWithFlags('1D IFFT', ALL_ENVS, () => {
  it('should return the same value with TensorFlow (2 elements)', async () => {
    const t1Real = tf.tensor1d([1, 2]);
    const t1Imag = tf.tensor1d([1, 1]);
    const t1 = tf.complex(t1Real, t1Imag);
    expectArraysClose(await tf.spectral.ifft(t1).data(), [1.5, 1, -0.5, 0]);
  });

  it('should calculate FFT from Tensor directly', async () => {
    const t1Real = tf.tensor1d([1, 2]);
    const t1Imag = tf.tensor1d([1, 1]);
    const t1 = tf.complex(t1Real, t1Imag);
    expectArraysClose(await t1.ifft().data(), [1.5, 1, -0.5, 0]);
  });

  it('should return the same value as TensorFlow (3 elements)', async () => {
    const t1Real = tf.tensor1d([1, 2, 3]);
    const t1Imag = tf.tensor1d([0, 0, 0]);
    const t1 = tf.complex(t1Real, t1Imag);
    expectArraysClose(await tf.spectral.ifft(t1).data(), [
      2, -3.9736431e-08, -0.49999997, -.28867507, -0.49999994, 2.8867519e-01
    ]);
  });

  it('should return the same value as TensorFlow with imaginary (3 elements)',
     async () => {
       const t1Real = tf.tensor1d([1, 2, 3]);
       const t1Imag = tf.tensor1d([1, 2, 3]);
       const t1 = tf.complex(t1Real, t1Imag);
       expectArraysClose(
           await tf.spectral.ifft(t1).data(),
           [2, 1.9999999, -0.21132492, -0.78867507, -0.7886752, -0.2113249]);
     });

  it('should return the same value as TensorFlow (negative 3 elements)',
     async () => {
       const t1Real = tf.tensor1d([-1, -2, -3]);
       const t1Imag = tf.tensor1d([-1, -2, -3]);
       const t1 = tf.complex(t1Real, t1Imag);
       expectArraysClose(
           await tf.spectral.ifft(t1).data(),
           [-2, -1.9999999, 0.21132492, 0.78867507, 0.7886752, 0.2113249]);
     });

  it('should return the same value with TensorFlow (4 elements)', async () => {
    const t1Real = tf.tensor1d([1, 2, 3, 4]);
    const t1Imag = tf.tensor1d([0, 0, 0, 0]);
    const t1 = tf.complex(t1Real, t1Imag);
    expectArraysClose(
        await tf.spectral.ifft(t1).data(),
        [2.5, 0, -0.5, -0.5, -0.5, 0, -0.5, 0.5]);
  });

  it('should return the same value as TensorFlow with imaginary (4 elements)',
     async () => {
       const t1Real = tf.tensor1d([1, 2, 3, 4]);
       const t1Imag = tf.tensor1d([1, 2, 3, 4]);
       const t1 = tf.complex(t1Real, t1Imag);
       expectArraysClose(
           await tf.spectral.ifft(t1).data(),
           [2.5, 2.5, 0, -1, -0.5, -0.5, -1, 0]);
     });
});

describeWithFlags('2D IFFT', ALL_ENVS, () => {
  it('2D: should return the same value as TensorFlow', async () => {
    const t1Real = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const t1Imag = tf.tensor2d([5, 6, 7, 8], [2, 2]);
    const t1 = tf.complex(t1Real, t1Imag);
    const y = tf.spectral.ifft(t1);
    expectArraysClose(
        await y.data(), [1.5, 5.5, -0.5, -0.5, 3.5, 7.5, -0.5, -0.5]);
    expect(y.shape).toEqual(t1Real.shape);
  });

  it('3D: should return the same value as TensorFlow', async () => {
    const t1Real = tf.tensor3d([1, 2, 3, 4, -1, -2, -3, -4], [2, 2, 2]);
    const t1Imag = tf.tensor3d([5, 6, 7, 8, -5, -6, -7, -8], [2, 2, 2]);
    const t1 = tf.complex(t1Real, t1Imag);
    const y = tf.spectral.ifft(t1);
    expectArraysClose(await y.data(), [
      1.5, 5.5, -0.5, -0.5, 3.5, 7.5, -0.5, -0.5, -1.5, -5.5, 0.5, 0.5, -3.5,
      -7.5, 0.5, 0.5
    ]);
    expect(y.shape).toEqual(t1Real.shape);
  });
});

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

describeWithFlags('1D IRFFT', ALL_ENVS, () => {
  it('should return the same value with TensorFlow (2 elements)', async () => {
    const t1Real = tf.tensor1d([1, 2]);
    const t1Imag = tf.tensor1d([0, 0]);
    const t1 = tf.complex(t1Real, t1Imag);
    expectArraysClose(await tf.spectral.irfft(t1).data(), [1.5, -0.5]);
  });

  it('should calculate from the tensor directly', async () => {
    const t1Real = tf.tensor1d([1, 2]);
    const t1Imag = tf.tensor1d([0, 0]);
    const t1 = tf.complex(t1Real, t1Imag);
    expectArraysClose(await t1.irfft().data(), [1.5, -0.5]);
  });

  it('should return the same value with TensorFlow (5 elements)', async () => {
    const t1Real = tf.tensor1d([1, 2, 3, 4, 5]);
    const t1Imag = tf.tensor1d([0, 0, 0, 0, 0]);
    const t1 = tf.complex(t1Real, t1Imag);
    expectArraysClose(
        await tf.spectral.irfft(t1).data(),
        [3, -0.8535534, 0, -0.14644662, 0, -0.14644662, 0, -0.8535534]);
  });

  it('should return the same value with TensorFlow (5 elements) with imag',
     async () => {
       const t1Real = tf.tensor1d([1, 2, 3, 4, 5]);
       const t1Imag = tf.tensor1d([1, 2, 3, 4, 5]);
       const t1 = tf.complex(t1Real, t1Imag);
       expectArraysClose(
           await tf.spectral.irfft(t1).data(),
           [3, -2.6642137, 0.5, -0.45710677, 0, 0.16421354, -0.5, 0.95710677]);
     });
});

describeWithFlags('2D IRFFT', ALL_ENVS, () => {
  it('should return the same value with TensorFlow (2x2 elements)',
     async () => {
       const t1Real = tf.tensor2d([1, 2, 3, 4], [2, 2]);
       const t1Imag = tf.tensor2d([0, 0, 0, 0], [2, 2]);
       const t1 = tf.complex(t1Real, t1Imag);
       expectArraysClose(
           await tf.spectral.irfft(t1).data(), [1.5, -0.5, 3.5, -0.5]);
     });

  it('should return the same value with TensorFlow (2x3 elements)',
     async () => {
       const t1Real = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
       const t1Imag = tf.tensor2d([0, 0, 0, 0, 0, 0], [2, 3]);
       const t1 = tf.complex(t1Real, t1Imag);
       expectArraysClose(
           await tf.spectral.irfft(t1).data(),
           [2, -0.5, 0, -0.5, 5, -0.5, 0, -0.5]);
     });

  it('should return the same value with TensorFlow (2x3 elements) with imag',
     async () => {
       const t1Real = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
       const t1Imag = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
       const t1 = tf.complex(t1Real, t1Imag);
       expectArraysClose(
           await tf.spectral.irfft(t1).data(), [2, -1.5, 0, 0.5, 5, -3, 0, 2]);
     });
});

describeWithFlags('3D IRFFT', ALL_ENVS, () => {
  it('should return the same value with TensorFlow (2x2x2 elements)',
     async () => {
       const t1Real = tf.tensor3d([1, 2, 3, 4, 1, 2, 3, 4], [2, 2, 2]);
       const t1Imag = tf.tensor3d([0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2]);
       const t1 = tf.complex(t1Real, t1Imag);
       expectArraysClose(
           await tf.spectral.irfft(t1).data(),
           [1.5, -0.5, 3.5, -0.5, 1.5, -0.5, 3.5, -0.5]);
     });

  it('should return the same value with TensorFlow (2x2x3 elements)',
     async () => {
       const t1Real =
           tf.tensor3d([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6], [2, 2, 3]);
       const t1Imag =
           tf.tensor3d([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 3]);
       const t1 = tf.complex(t1Real, t1Imag);
       expectArraysClose(await tf.spectral.irfft(t1).data(), [
         2, -0.5, 0, -0.5, 5, -0.5, 0, -0.5, 2, -0.5, 0, -0.5, 5, -0.5, 0, -0.5
       ]);
     });

  it('should return the same value with TensorFlow (2x2x3 elements) with imag',
     async () => {
       const t1Real =
           tf.tensor3d([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6], [2, 2, 3]);
       const t1Imag =
           tf.tensor3d([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6], [2, 2, 3]);
       const t1 = tf.complex(t1Real, t1Imag);
       expectArraysClose(
           await tf.spectral.irfft(t1).data(),
           [2, -1.5, 0, 0.5, 5, -3, 0, 2, 2, -1.5, 0, 0.5, 5, -3, 0, 2]);
     });
});
