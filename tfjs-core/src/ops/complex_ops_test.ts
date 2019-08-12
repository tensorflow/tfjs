/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import {ALL_ENVS, BROWSER_ENVS, describeWithFlags} from '../jasmine_util';
import {expectArraysClose} from '../test_util';

describeWithFlags('complex64', ALL_ENVS, () => {
  it('tf.complex', async () => {
    const real = tf.tensor1d([3, 30]);
    const imag = tf.tensor1d([4, 40]);
    const complex = tf.complex(real, imag);

    expect(complex.dtype).toBe('complex64');
    expect(complex.shape).toEqual(real.shape);
    expectArraysClose(await complex.data(), [3, 4, 30, 40]);
  });

  it('tf.real', async () => {
    const complex = tf.complex([3, 30], [4, 40]);
    const real = tf.real(complex);

    expect(real.dtype).toBe('float32');
    expect(real.shape).toEqual([2]);
    expectArraysClose(await real.data(), [3, 30]);
  });

  it('tf.imag', async () => {
    const complex = tf.complex([3, 30], [4, 40]);
    const imag = tf.imag(complex);

    expect(imag.dtype).toBe('float32');
    expect(imag.shape).toEqual([2]);
    expectArraysClose(await imag.data(), [4, 40]);
  });

  it('throws when shapes dont match', () => {
    const real = tf.tensor1d([3, 30]);
    const imag = tf.tensor1d([4, 40, 50]);

    const re =
        /real and imag shapes, 2 and 3, must match in call to tf.complex\(\)/;
    expect(() => tf.complex(real, imag)).toThrowError(re);
  });
});

const BYTES_PER_COMPLEX_ELEMENT = 4 * 2;
describeWithFlags('complex64 memory', BROWSER_ENVS, () => {
  it('usage', async () => {
    let numTensors = tf.memory().numTensors;
    let numBuffers = tf.memory().numDataBuffers;
    const startTensors = numTensors;

    const real1 = tf.tensor1d([1]);
    const imag1 = tf.tensor1d([2]);
    const complex1 = tf.complex(real1, imag1);

    // 5 new Tensors: real1, imag1, complex1, and two internal clones.
    expect(tf.memory().numTensors).toBe(numTensors + 5);
    // Only 3 new data buckets are actually created.
    expect(tf.memory().numDataBuffers).toBe(numBuffers + 3);
    numTensors = tf.memory().numTensors;
    numBuffers = tf.memory().numDataBuffers;

    const real2 = tf.tensor1d([3]);
    const imag2 = tf.tensor1d([4]);
    const complex2 = tf.complex(real2, imag2);

    // 5 new Tensors: real1, imag1, complex1, and two internal clones.
    expect(tf.memory().numTensors).toBe(numTensors + 5);
    // Only 3 new data buckets are actually created.
    expect(tf.memory().numDataBuffers).toBe(numBuffers + 3);
    numTensors = tf.memory().numTensors;
    numBuffers = tf.memory().numDataBuffers;

    const result = complex1.add(complex2);

    // A complex tensor is created, which is composed of 2 underlying tensors.
    expect(tf.memory().numTensors).toBe(numTensors + 3);
    numTensors = tf.memory().numTensors;

    expect(result.dtype).toBe('complex64');
    expect(result.shape).toEqual([1]);
    expectArraysClose(await result.data(), [4, 6]);

    const real = tf.real(result);

    expect(tf.memory().numTensors).toBe(numTensors + 1);
    numTensors = tf.memory().numTensors;

    expectArraysClose(await real.data(), [4]);

    const imag = tf.imag(result);

    expect(tf.memory().numTensors).toBe(numTensors + 1);
    numTensors = tf.memory().numTensors;

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
  });

  it('tf.complex disposing underlying tensors', async () => {
    const numTensors = tf.memory().numTensors;

    const real = tf.tensor1d([3, 30]);
    const imag = tf.tensor1d([4, 40]);
    expect(tf.memory().numTensors).toEqual(numTensors + 2);

    const complex = tf.complex(real, imag);

    // real and imag are cloned.
    expect(tf.memory().numTensors).toEqual(numTensors + 5);

    real.dispose();
    imag.dispose();

    // A copy of real and imag still exist, the one owned by the complex tensor.
    expect(tf.memory().numTensors).toEqual(numTensors + 3);

    expect(complex.dtype).toBe('complex64');
    expect(complex.shape).toEqual(real.shape);
    expectArraysClose(await complex.data(), [3, 4, 30, 40]);

    complex.dispose();

    expect(tf.memory().numTensors).toEqual(numTensors);
  });

  it('reshape', async () => {
    const memoryBefore = tf.memory();

    const a = tf.complex([[1, 3, 5], [7, 9, 11]], [[2, 4, 6], [8, 10, 12]]);

    // 3 new tensors, the complex64 tensor and the 2 underlying float32 tensors.
    expect(tf.memory().numTensors).toBe(memoryBefore.numTensors + 3);
    // Bytes should be counted once.
    expect(tf.memory().numBytes)
        .toBe(memoryBefore.numBytes + 6 * BYTES_PER_COMPLEX_ELEMENT);

    const b = a.reshape([6]);
    // 1 new tensor from the reshape.
    expect(tf.memory().numTensors).toBe(memoryBefore.numTensors + 4);
    // No new bytes from a reshape.
    expect(tf.memory().numBytes)
        .toBe(memoryBefore.numBytes + 6 * BYTES_PER_COMPLEX_ELEMENT);

    expect(b.dtype).toBe('complex64');
    expect(b.shape).toEqual([6]);
    expectArraysClose(await a.data(), await b.data());

    b.dispose();
    // 1 complex tensor should be disposed.
    expect(tf.memory().numTensors).toBe(memoryBefore.numTensors + 3);
    // Byte count should not change because the refcounts are all 1.
    expect(tf.memory().numBytes)
        .toBe(memoryBefore.numBytes + 6 * BYTES_PER_COMPLEX_ELEMENT);

    a.dispose();
    // All the tensors should now be disposed.
    expect(tf.memory().numTensors).toBe(memoryBefore.numTensors);
    // The underlying memory should now be released.
    expect(tf.memory().numBytes).toBe(memoryBefore.numBytes);
  });

  it('clone', async () => {
    const memoryBefore = tf.memory();

    const a = tf.complex([[1, 3, 5], [7, 9, 11]], [[2, 4, 6], [8, 10, 12]]);

    // 3 new tensors, the complex64 tensor and the 2 underlying float32 tensors.
    expect(tf.memory().numTensors).toBe(memoryBefore.numTensors + 3);
    // Bytes should be counted once
    expect(tf.memory().numBytes)
        .toBe(memoryBefore.numBytes + 6 * BYTES_PER_COMPLEX_ELEMENT);

    const b = a.clone();
    // 1 new tensor from the clone.
    expect(tf.memory().numTensors).toBe(memoryBefore.numTensors + 4);
    // No new bytes from a clone.
    expect(tf.memory().numBytes)
        .toBe(memoryBefore.numBytes + 6 * BYTES_PER_COMPLEX_ELEMENT);

    expect(b.dtype).toBe('complex64');
    expectArraysClose(await a.data(), await b.data());

    b.dispose();
    // 1 complex tensor should be disposed.
    expect(tf.memory().numTensors).toBe(memoryBefore.numTensors + 3);
    // Byte count should not change because the refcounts are all 1.
    expect(tf.memory().numBytes)
        .toBe(memoryBefore.numBytes + 6 * BYTES_PER_COMPLEX_ELEMENT);

    a.dispose();
    // All the tensors should now be disposed.
    expect(tf.memory().numTensors).toBe(memoryBefore.numTensors);
    // The underlying memory should now be released.
    expect(tf.memory().numBytes).toBe(memoryBefore.numBytes);
  });
});
