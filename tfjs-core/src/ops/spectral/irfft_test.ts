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
       const result = tf.spectral.irfft(t1);
       expect(result.shape).toEqual([2, 2, 2]);

       expectArraysClose(
           await result.data(), [1.5, -0.5, 3.5, -0.5, 1.5, -0.5, 3.5, -0.5]);
     });

  it('should return the same value with TensorFlow (2x2x3 elements)',
     async () => {
       const t1Real =
           tf.tensor3d([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6], [2, 2, 3]);
       const t1Imag =
           tf.tensor3d([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 3]);
       const t1 = tf.complex(t1Real, t1Imag);
       const result = tf.spectral.irfft(t1);
       expect(result.shape).toEqual([2, 2, 4]);
       expectArraysClose(await result.data(), [
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
       const result = tf.spectral.irfft(t1);
       expect(result.shape).toEqual([2, 2, 4]);
       expectArraysClose(
           await result.data(),
           [2, -1.5, 0, 0.5, 5, -3, 0, 2, 2, -1.5, 0, 0.5, 5, -3, 0, 2]);
     });
});
