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

describeWithFlags('FFT2D', ALL_ENVS, () => {
  it('should calculate FFT2D from Tensor directly', async () => {
    const t1Real = tf.tensor2d([[1]]);
    const t1Imag = tf.tensor2d([[0]]);
    const t1 = tf.complex(t1Real, t1Imag);
    expectArraysClose(await t1.fft2d().data(), [1, 0]);
  });

  it('should return the same value with TensorFlow (1x1)', async () => {
    const t1Real = tf.tensor2d([[1]]);
    const t1Imag = tf.tensor2d([[0]]);
    const t1 = tf.complex(t1Real, t1Imag);
    expectArraysClose(await tf.spectral.fft2d(t1).data(), [1, 0]);
  });

  it('should return the same value with TensorFlow (NxN)', async () => {
    const t1Real = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3]);
    const t1Imag = tf.tensor2d([0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3]);
    const t1 = tf.complex(t1Real, t1Imag);
    expectArraysClose(await tf.spectral.fft2d(t1).data(), [
      45, 0, 10.499999046325684, 28.578838348388672, -4.499999046325684,
      -2.5980780124664307, 1.4999985694885254, 33.77499008178711,
      -21.999998092651367, 10.392301559448242, 0.000001430511474609375,
      -3.4641010761260986, -13.499999046325684, -7.7942304611206055,
      0.000001430511474609375, -10.392304420471191, -7.979140264069429e-7,
      5.069260282652976e-7
    ]);
  });

  it('should return the same value as TensorFlow with imaginary (NxN)',
     async () => {
       const t1Real = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3]);
       const t1Imag = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3]);
       const t1 = tf.complex(t1Real, t1Imag);
       expectArraysClose(await tf.spectral.fft2d(t1).data(), [
         45, 45, -18.078838348388672, 39.07883834838867, -1.901921272277832,
         -7.098076820373535, -32.27499008178711, 35.27499008178711,
         -32.392303466796875, -11.607696533203125, 3.464102268218994,
         -3.464099407196045, -5.705768585205078, -21.29422950744629,
         10.392306327819824, -10.392304420471191, -0.0000011717994539139909,
         -1.717786517474451e-7
       ]);
     });

  it('should return the same value as TensorFlow with negative (NxN)',
     async () => {
       const t1Real = tf.tensor2d([-1, -2, -3, -4, -5, -6, -7, -8, -9], [3, 3]);
       const t1Imag = tf.tensor2d([-1, -2, -3, -4, -5, -6, -7, -8, -9], [3, 3]);
       const t1 = tf.complex(t1Real, t1Imag);
       expectArraysClose(await tf.spectral.fft2d(t1).data(), [
         -45, -45, 18.078838348388672, -39.07883834838867, 1.901921272277832,
         7.098076820373535, 32.27499008178711, -35.27499008178711,
         32.392303466796875, 11.607696533203125, -3.464102268218994,
         3.464099407196045, 5.705768585205078, 21.29422950744629,
         -10.392306327819824, 10.392304420471191, 0.0000011717994539139909,
         1.717786517474451e-7
       ]);
     });

  it('should return the same value as TensorFlow (NxM)', async () => {
    const t1Real = tf.tensor2d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        [4, 5]);
    const t1Imag = tf.tensor2d(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [4, 5]);
    const t1 = tf.complex(t1Real, t1Imag);
    expectArraysClose(await tf.spectral.fft2d(t1).data(), [
      210,
      0,
      -9.999992370605469,
      13.763818740844727,
      -9.999995231628418,
      3.249201774597168,
      -9.999984741210938,
      -3.2491941452026367,
      -10.000024795532227,
      -13.763805389404297,
      -49.999996185302734,
      49.9999885559082,
      -8.910799351724563e-7,
      9.16725582555955e-7,
      -0.0000015264142803061986,
      -7.435900215568836e-7,
      -0.0000049903724175237585,
      0.0000028098704660806106,
      -0.0000016918011169764213,
      -0.000007075884241203312,
      -50,
      0.00001573610097693745,
      -0.0000025089843802561518,
      5.165608740753669e-7,
      -0.0000010957016911561368,
      -4.371137265479774e-7,
      -0.0000036726705729961395,
      -0.000001986833467526594,
      0.000004893169261777075,
      -0.000005205487013881793,
      -49.999916076660156,
      -50,
      -0.00000455239251095918,
      0.0000019477663499856135,
      -0.0000028207703053340083,
      -1.9583012544899248e-8,
      -0.000004808577614312526,
      -0.000004510406142799184,
      0.000007294238912436413,
      -0.0000018285909391124733
    ]);
  });

  it('should return the same value as TensorFlow with imaginary (NxM)',
     async () => {
       const t1Real = tf.tensor2d(
           [
             1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
             11, 12, 13, 14, 15, 16, 17, 18, 19, 20
           ],
           [4, 5]);
       const t1Imag = tf.tensor2d(
           [
             1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
             11, 12, 13, 14, 15, 16, 17, 18, 19, 20
           ],
           [4, 5]);
       const t1 = tf.complex(t1Real, t1Imag);
       expectArraysClose(await tf.spectral.fft2d(t1).data(), [
         210,
         210,
         -23.763811111450195,
         3.763826370239258,
         -13.249197006225586,
         -6.75079345703125,
         -6.750790119171143,
         -13.249177932739258,
         3.763780117034912,
         -23.763832092285156,
         -99.99998474121094,
         -0.00000762939453125,
         -0.0000020462241536733927,
         2.640642833284801e-7,
         -7.82824258749315e-7,
         -0.0000022700041881762445,
         -0.000007800243110978045,
         -0.000002299711240993929,
         0.0000053840831242268905,
         -0.000008767685358179733,
         -50.00001525878906,
         -49.99998474121094,
         -0.0000020718709947686875,
         -0.0000015155864048210788,
         -0.0000011354250091244467,
         -0.000001771233883118839,
         -0.0000018050462813334889,
         -0.0000055402947509719525,
         0.000010098656275658868,
         -7.891551945249375e-7,
         0.00008392333984375,
         -99.99991607666016,
         -0.000006738577212672681,
         -0.0000028430440579541028,
         -0.000002801187292789109,
         -0.000002840353545252583,
         -5.975243766442873e-8,
         -0.00000943819304666249,
         0.000009122829396801535,
         0.0000054656475185765885
       ]);
     });

  it('should return the same value as TensorFlow with negative (NxM)',
     async () => {
       const t1Real = tf.tensor2d(
           [
             -1,  -2,  -3,  -4,  -5,  -6,  -7,  -8,  -9,  -10,
             -11, -12, -13, -14, -15, -16, -17, -18, -19, -20
           ],
           [4, 5]);
       const t1Imag = tf.tensor2d(
           [
             -1,  -2,  -3,  -4,  -5,  -6,  -7,  -8,  -9,  -10,
             -11, -12, -13, -14, -15, -16, -17, -18, -19, -20
           ],
           [4, 5]);
       const t1 = tf.complex(t1Real, t1Imag);
       expectArraysClose(await tf.spectral.fft2d(t1).data(), [
         -210,
         -210,
         23.763811111450195,
         -3.763826370239258,
         13.249197006225586,
         6.75079345703125,
         6.750790119171143,
         13.249177932739258,
         -3.763780117034912,
         23.763832092285156,
         99.99998474121094,
         0.00000762939453125,
         0.0000020462241536733927,
         -2.640642833284801e-7,
         7.82824258749315e-7,
         0.0000022700041881762445,
         0.000007800243110978045,
         0.000002299711240993929,
         -0.0000053840831242268905,
         0.000008767685358179733,
         50.00001525878906,
         49.99998474121094,
         0.0000020718709947686875,
         0.0000015155864048210788,
         0.0000011354250091244467,
         0.000001771233883118839,
         0.0000018050462813334889,
         0.0000055402947509719525,
         -0.000010098656275658868,
         7.891551945249375e-7,
         -0.00008392333984375,
         99.99991607666016,
         0.000006738577212672681,
         0.0000028430440579541028,
         0.000002801187292789109,
         0.000002840353545252583,
         5.975243766442873e-8,
         0.00000943819304666249,
         -0.000009122829396801535,
         -0.0000054656475185765885
       ]);
     });

  it('should work with batches', async () => {
    const t1Real = tf.tensor3d([
      [
        [-1, -2, -3, -4, -5], [-6, -7, -8, -9, -10], [-11, -12, -13, -14, -15],
        [-16, -17, -18, -19, -20]
      ],
      [
        [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20]
      ],
      [
        [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20]
      ],
    ]);
    const t1Imag = tf.tensor3d([
      [
        [-1, -2, -3, -4, -5], [-6, -7, -8, -9, -10], [-11, -12, -13, -14, -15],
        [-16, -17, -18, -19, -20]
      ],
      [
        [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20]
      ],
      [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    ]);

    const t1 = tf.complex(t1Real, t1Imag);
    expectArraysClose(await tf.spectral.fft2d(t1).data(), [
      [
        -210,
        -210,
        23.763811111450195,
        -3.763826370239258,
        13.249197006225586,
        6.75079345703125,
        6.750790119171143,
        13.249177932739258,
        -3.763780117034912,
        23.763832092285156,
        99.99998474121094,
        0.00000762939453125,
        0.0000020462241536733927,
        -2.640642833284801e-7,
        7.82824258749315e-7,
        0.0000022700041881762445,
        0.000007800243110978045,
        0.000002299711240993929,
        -0.0000053840831242268905,
        0.000008767685358179733,
        50.00001525878906,
        49.99998474121094,
        0.0000020718709947686875,
        0.0000015155864048210788,
        0.0000011354250091244467,
        0.000001771233883118839,
        0.0000018050462813334889,
        0.0000055402947509719525,
        -0.000010098656275658868,
        7.891551945249375e-7,
        -0.00008392333984375,
        99.99991607666016,
        0.000006738577212672681,
        0.0000028430440579541028,
        0.000002801187292789109,
        0.000002840353545252583,
        5.975243766442873e-8,
        0.00000943819304666249,
        -0.000009122829396801535,
        -0.0000054656475185765885
      ],
      [
        210,
        210,
        -23.763811111450195,
        3.763826370239258,
        -13.249197006225586,
        -6.75079345703125,
        -6.750790119171143,
        -13.249177932739258,
        3.763780117034912,
        -23.763832092285156,
        -99.99998474121094,
        -0.00000762939453125,
        -0.0000020462241536733927,
        2.640642833284801e-7,
        -7.82824258749315e-7,
        -0.0000022700041881762445,
        -0.000007800243110978045,
        -0.000002299711240993929,
        0.0000053840831242268905,
        -0.000008767685358179733,
        -50.00001525878906,
        -49.99998474121094,
        -0.0000020718709947686875,
        -0.0000015155864048210788,
        -0.0000011354250091244467,
        -0.000001771233883118839,
        -0.0000018050462813334889,
        -0.0000055402947509719525,
        0.000010098656275658868,
        -7.891551945249375e-7,
        0.00008392333984375,
        -99.99991607666016,
        -0.000006738577212672681,
        -0.0000028430440579541028,
        -0.000002801187292789109,
        -0.000002840353545252583,
        -5.975243766442873e-8,
        -0.00000943819304666249,
        0.000009122829396801535,
        0.0000054656475185765885
      ],
      [
        210,
        0,
        -9.999992370605469,
        13.763818740844727,
        -9.999995231628418,
        3.249201774597168,
        -9.999984741210938,
        -3.2491941452026367,
        -10.000024795532227,
        -13.763805389404297,
        -49.999996185302734,
        49.9999885559082,
        -8.910799351724563e-7,
        9.16725582555955e-7,
        -0.0000015264142803061986,
        -7.435900215568836e-7,
        -0.0000049903724175237585,
        0.0000028098704660806106,
        -0.0000016918011169764213,
        -0.000007075884241203312,
        -50,
        0.00001573610097693745,
        -0.0000025089843802561518,
        5.165608740753669e-7,
        -0.0000010957016911561368,
        -4.371137265479774e-7,
        -0.0000036726705729961395,
        -0.000001986833467526594,
        0.000004893169261777075,
        -0.000005205487013881793,
        -49.999916076660156,
        -50,
        -0.00000455239251095918,
        0.0000019477663499856135,
        -0.0000028207703053340083,
        -1.9583012544899248e-8,
        -0.000004808577614312526,
        -0.000004510406142799184,
        0.000007294238912436413,
        -0.0000018285909391124733
      ]
    ]);
  });
});
