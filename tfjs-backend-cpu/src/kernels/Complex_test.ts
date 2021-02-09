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

import * as tf from '@tensorflow/tfjs-core';
import {test_util} from '@tensorflow/tfjs-core';

const {expectArraysClose} = test_util;
// tslint:disable-next-line: no-imports-from-dist
import {describeWithFlags, ALL_ENVS} from '@tensorflow/tfjs-core/dist/jasmine_util';

describeWithFlags('Complex.', ALL_ENVS, () => {
  it('memory usage.', async () => {
    let numTensors = tf.memory().numTensors;
    let numDataIds = tf.engine().backend.numDataIds();
    const startTensors = numTensors;
    const startDataIds = numDataIds;

    const real1 = tf.tensor1d([1]);
    const imag1 = tf.tensor1d([2]);

    // 2 new Tensors: real1, imag1.
    expect(tf.memory().numTensors).toBe(numTensors + 2);
    expect(tf.engine().backend.numDataIds()).toBe(numDataIds + 2);
    numTensors = tf.memory().numTensors;
    numDataIds = tf.engine().backend.numDataIds();

    const complex1 = tf.complex(real1, imag1);

    // 1 new Tensor and 3 new TensorData for complex, real and imag.
    expect(tf.memory().numTensors).toBe(numTensors + 1);
    expect(tf.engine().backend.numDataIds()).toBe(numDataIds + 3);
    numTensors = tf.memory().numTensors;
    numDataIds = tf.engine().backend.numDataIds();

    const real2 = tf.tensor1d([3]);
    const imag2 = tf.tensor1d([4]);

    // 2 new Tensors: real2, imag2.
    expect(tf.memory().numTensors).toBe(numTensors + 2);
    expect(tf.engine().backend.numDataIds()).toBe(numDataIds + 2);
    numTensors = tf.memory().numTensors;
    numDataIds = tf.engine().backend.numDataIds();

    const complex2 = tf.complex(real2, imag2);

    // 1 new Tensor and 3 new TensorData for complex, real and imag.
    expect(tf.memory().numTensors).toBe(numTensors + 1);
    expect(tf.engine().backend.numDataIds()).toBe(numDataIds + 3);
    numTensors = tf.memory().numTensors;
    numDataIds = tf.engine().backend.numDataIds();

    const result = complex1.add(complex2);

    // A complex tensor is created, which is composed of real and imag parts.
    // They should not increase tensor count, only complex tensor does.
    // 3 new tensorData is created for complex, real and imag.
    expect(tf.memory().numTensors).toBe(numTensors + 1);
    expect(tf.engine().backend.numDataIds()).toBe(numDataIds + 3);
    numTensors = tf.memory().numTensors;
    numDataIds = tf.engine().backend.numDataIds();

    expect(result.dtype).toBe('complex64');
    expect(result.shape).toEqual([1]);
    expectArraysClose(await result.data(), [4, 6]);

    const real = tf.real(result);

    // A new tensor is created. A new tensorData is created.
    expect(tf.memory().numTensors).toBe(numTensors + 1);
    expect(tf.engine().backend.numDataIds()).toBe(numDataIds + 1);
    numTensors = tf.memory().numTensors;
    numDataIds = tf.engine().backend.numDataIds();

    expectArraysClose(await real.data(), [4]);

    const imag = tf.imag(result);

    // A new tensor is created. A new tensorData is created.
    expect(tf.memory().numTensors).toBe(numTensors + 1);
    expect(tf.engine().backend.numDataIds()).toBe(numDataIds + 1);
    numTensors = tf.memory().numTensors;
    numDataIds = tf.engine().backend.numDataIds();

    expectArraysClose(await imag.data(), [6]);

    // After disposing, there should be no tensors.
    real1.dispose();
    imag1.dispose();
    real2.dispose();
    imag2.dispose();
    complex1.dispose();
    complex2.dispose();
    result.dispose();
    real.dispose();
    imag.dispose();
    expect(tf.memory().numTensors).toBe(startTensors);
    expect(tf.engine().backend.numDataIds()).toBe(startDataIds);
  });

  it('Creating tf.real, tf.imag from complex.', async () => {
    let numTensors = tf.memory().numTensors;
    let numDataIds = tf.engine().backend.numDataIds();

    const startTensors = numTensors;
    const startDataIds = numDataIds;

    const complex = tf.complex([3, 30], [4, 40]);

    // 1 new tensor, 3 new data buckets.
    expect(tf.memory().numTensors).toBe(numTensors + 1);
    expect(tf.engine().backend.numDataIds()).toBe(numDataIds + 3);
    numTensors = tf.memory().numTensors;
    numDataIds = tf.engine().backend.numDataIds();

    const real = tf.real(complex);
    const imag = tf.imag(complex);

    // 2 new tensors, 2 new data buckets.
    expect(tf.memory().numTensors).toBe(numTensors + 2);
    expect(tf.engine().backend.numDataIds()).toBe(numDataIds + 2);

    numTensors = tf.memory().numTensors;
    numDataIds = tf.engine().backend.numDataIds();

    complex.dispose();

    // 1 fewer tensor, 3 fewer data buckets.
    expect(tf.memory().numTensors).toBe(numTensors - 1);
    expect(tf.engine().backend.numDataIds()).toBe(numDataIds - 3);

    numTensors = tf.memory().numTensors;
    numDataIds = tf.engine().backend.numDataIds();

    real.dispose();
    imag.dispose();

    // Zero net tensors / data buckets.
    expect(tf.memory().numTensors).toBe(startTensors);
    expect(tf.engine().backend.numDataIds()).toBe(startDataIds);
  });

  it('tf.complex disposing underlying tensors', async () => {
    const numTensors = tf.memory().numTensors;
    const numDataIds = tf.engine().backend.numDataIds();

    const real = tf.tensor1d([3, 30]);
    const imag = tf.tensor1d([4, 40]);
    expect(tf.memory().numTensors).toEqual(numTensors + 2);
    expect(tf.engine().backend.numDataIds()).toEqual(numDataIds + 2);

    const complex = tf.complex(real, imag);

    // 1 new tensor is created for complex. real and imag tensorData is created.
    expect(tf.memory().numTensors).toEqual(numTensors + 3);
    expect(tf.engine().backend.numDataIds()).toEqual(numDataIds + 5);

    real.dispose();
    imag.dispose();

    expect(tf.memory().numTensors).toEqual(numTensors + 1);
    expect(tf.engine().backend.numDataIds()).toEqual(numDataIds + 3);

    expect(complex.dtype).toBe('complex64');
    expect(complex.shape).toEqual(real.shape);
    expectArraysClose(await complex.data(), [3, 4, 30, 40]);

    complex.dispose();

    expect(tf.memory().numTensors).toEqual(numTensors);
    expect(tf.engine().backend.numDataIds()).toEqual(numDataIds);
  });

  it('reshape', async () => {
    const memoryBefore = tf.memory();
    const numDataIdsBefore = tf.engine().backend.numDataIds();

    const a = tf.complex([[1, 3, 5], [7, 9, 11]], [[2, 4, 6], [8, 10, 12]]);

    // 1 new tensor, the complex64 tensor
    expect(tf.memory().numTensors).toBe(memoryBefore.numTensors + 1);
    // 1 new tensor and 2 underlying tensors for real and imag.
    expect(tf.engine().backend.numDataIds()).toBe(numDataIdsBefore + 3);

    const b = a.reshape([6]);

    // 1 new tensor from the reshape.
    expect(tf.memory().numTensors).toBe(memoryBefore.numTensors + 2);
    // No new tensor data.
    expect(tf.engine().backend.numDataIds()).toBe(numDataIdsBefore + 3);

    expect(b.dtype).toBe('complex64');
    expect(b.shape).toEqual([6]);
    expectArraysClose(await a.data(), await b.data());

    b.dispose();
    // 1 complex tensor should be disposed.
    expect(tf.memory().numTensors).toBe(memoryBefore.numTensors + 1);
    // Tensor data not deleted yet.
    expect(tf.engine().backend.numDataIds()).toBe(numDataIdsBefore + 3);

    a.dispose();

    // All the tensors should now be disposed.
    expect(tf.memory().numTensors).toBe(memoryBefore.numTensors);
    expect(tf.engine().backend.numDataIds()).toBe(numDataIdsBefore);
  });

  it('clone', async () => {
    const memoryBefore = tf.memory();
    const numDataIdsBefore = tf.engine().backend.numDataIds();

    const a = tf.complex([[1, 3, 5], [7, 9, 11]], [[2, 4, 6], [8, 10, 12]]);

    // 1 new tensor, the complex64 tensor
    expect(tf.memory().numTensors).toBe(memoryBefore.numTensors + 1);
    // 1 new tensor and 2 underlying tensors for real and imag.
    expect(tf.engine().backend.numDataIds()).toBe(numDataIdsBefore + 3);

    const b = a.clone();

    // 1 new tensor from the clone.
    expect(tf.memory().numTensors).toBe(memoryBefore.numTensors + 2);
    // No new tensor data.
    expect(tf.engine().backend.numDataIds()).toBe(numDataIdsBefore + 3);

    expect(b.dtype).toBe('complex64');
    expectArraysClose(await a.data(), await b.data());

    b.dispose();

    // 1 complex tensor should be disposed.
    expect(tf.memory().numTensors).toBe(memoryBefore.numTensors + 1);
    // Tensor data not deleted yet.
    expect(tf.engine().backend.numDataIds()).toBe(numDataIdsBefore + 3);

    a.dispose();

    // All the tensors should now be disposed.
    expect(tf.memory().numTensors).toBe(memoryBefore.numTensors);
    expect(tf.engine().backend.numDataIds()).toBe(numDataIdsBefore);
  });

  it('tidy should not have mem leak', async () => {
    const numTensors = tf.memory().numTensors;
    const numDataIds = tf.engine().backend.numDataIds();
    const complex = tf.tidy(() => {
      const real = tf.tensor1d([3, 30]);
      const realReshape = tf.reshape(real, [2]);
      const imag = tf.tensor1d([4, 40]);
      const imagReshape = tf.reshape(imag, [2]);
      expect(tf.memory().numTensors).toEqual(numTensors + 4);
      expect(tf.engine().backend.numDataIds()).toEqual(numDataIds + 2);

      const complex = tf.complex(realReshape, imagReshape);

      // 1 new tensor is created for complex. real and imag data buckets
      // created.
      expect(tf.memory().numTensors).toEqual(numTensors + 5);
      expect(tf.engine().backend.numDataIds()).toEqual(numDataIds + 5);

      return complex;
    });

    complex.dispose();

    expect(tf.memory().numTensors).toEqual(numTensors);
    expect(tf.engine().backend.numDataIds()).toEqual(numDataIds);
  });
});
