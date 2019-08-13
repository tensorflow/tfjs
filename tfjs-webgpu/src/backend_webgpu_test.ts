/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {WebGPUBackend, WebGPUMemoryInfo} from './backend_webgpu';
import {describeWebGPU} from './test_util';

describeWebGPU('backend webgpu', () => {
  it('should not leak memory in delayed mode', async () => {
    const savedFlag = tf.ENV.get('WEBGPU_IMMEDIATE_EXECUTION_ENABLED');
    tf.ENV.set('WEBGPU_IMMEDIATE_EXECUTION_ENABLED', false);
    const a = tf.tensor2d([2, 4, 6, 8], [2, 2]);
    const b = tf.tensor2d([0.5, 0.5, 0.5, 0.5], [2, 2]);

    const c = tf.mul(a, b);

    const startNumBytes = tf.memory().numBytes;
    const startNumTensors = tf.memory().numTensors;
    const startNumBytesInGPU = (tf.memory() as WebGPUMemoryInfo).numBytesInGPU;

    const f = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const d = tf.matMul(c, f);

    const dData = await d.data();
    const endNumBytes = tf.memory().numBytes;
    const endNumTensors = tf.memory().numTensors;
    const endNumBytesInGPU = (tf.memory() as WebGPUMemoryInfo).numBytesInGPU;

    expect(endNumBytes - startNumBytes).toEqual(48);
    expect(endNumTensors - startNumTensors).toEqual(2);
    expect(endNumBytesInGPU - startNumBytesInGPU).toEqual(48);

    tf.test_util.expectArraysClose(
        dData, new Float32Array([9, 12, 15, 19, 26, 33]));
    tf.ENV.set('WEBGPU_IMMEDIATE_EXECUTION_ENABLED', savedFlag);
  });

  it('should not leak memory in immediate mode', async () => {
    const savedFlag = tf.ENV.get('WEBGPU_IMMEDIATE_EXECUTION_ENABLED');
    tf.ENV.set('WEBGPU_IMMEDIATE_EXECUTION_ENABLED', true);
    const a = tf.tensor2d([2, 4, 6, 8], [2, 2]);
    const b = tf.tensor2d([0.5, 0.5, 0.5, 0.5], [2, 2]);

    const c = tf.mul(a, b);

    const startNumBytes = tf.memory().numBytes;
    const startNumTensors = tf.memory().numTensors;
    const startNumBytesInGPU = (tf.memory() as WebGPUMemoryInfo).numBytesInGPU;

    const f = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const d = tf.matMul(c, f);

    const dData = await d.data();
    const endNumBytes = tf.memory().numBytes;
    const endNumTensors = tf.memory().numTensors;
    const endNumBytesInGPU = (tf.memory() as WebGPUMemoryInfo).numBytesInGPU;

    expect(endNumBytes - startNumBytes).toEqual(48);
    expect(endNumTensors - startNumTensors).toEqual(2);
    expect(endNumBytesInGPU - startNumBytesInGPU).toEqual(48);

    tf.test_util.expectArraysClose(
        dData, new Float32Array([9, 12, 15, 19, 26, 33]));
    tf.ENV.set('WEBGPU_IMMEDIATE_EXECUTION_ENABLED', savedFlag);
  });

  it('should recycle buffers in immediate mode', () => {
    const savedFlag = tf.ENV.get('WEBGPU_IMMEDIATE_EXECUTION_ENABLED');
    tf.ENV.set('WEBGPU_IMMEDIATE_EXECUTION_ENABLED', true);
    const backend = tf.backend() as WebGPUBackend;
    const bufferManager = backend.getBufferManager();
    bufferManager.reset();

    const a = tf.tensor2d([2, 4, 6, 8], [2, 2]);
    const b = tf.tensor2d([0.5, 0.5, 0.5, 0.5], [2, 2]);

    const c = tf.mul(a, b);
    const freeBuffersAfterFirstMul = bufferManager.getNumFreeBuffers();
    const usedBuffersAfterFirstMul = bufferManager.getNumUsedBuffers();

    const f = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    tf.matMul(c, f);
    const freeBuffersAfterFirstMatMul = bufferManager.getNumFreeBuffers();
    const usedBuffersAfterFirstMatMul = bufferManager.getNumUsedBuffers();
    expect(freeBuffersAfterFirstMatMul - freeBuffersAfterFirstMul)
        .toEqual(1);  // from released uniform
    expect(usedBuffersAfterFirstMatMul - usedBuffersAfterFirstMul).toEqual(2);

    const a2 = tf.tensor2d([2, 4, 6, 8], [2, 2]);
    const b2 = tf.tensor2d([0.5, 0.5, 0.5, 0.5], [2, 2]);

    const c2 = tf.mul(a2, b2);
    const freeBuffersAfterSecondMul = bufferManager.getNumFreeBuffers();
    const usedBuffersAfterSecondMul = bufferManager.getNumUsedBuffers();
    expect(freeBuffersAfterSecondMul - freeBuffersAfterFirstMatMul)
        .toEqual(0);  // released a uniform buffer and reused a buffer
    expect(usedBuffersAfterSecondMul - usedBuffersAfterFirstMatMul).toEqual(3);

    const f2 = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    tf.matMul(c2, f2);
    const freeBuffersAfterSecondMatMul = bufferManager.getNumFreeBuffers();
    const usedBuffersAfterSecondMatMul = bufferManager.getNumUsedBuffers();
    expect(freeBuffersAfterSecondMatMul - freeBuffersAfterSecondMul).toEqual(0);
    expect(usedBuffersAfterSecondMatMul - usedBuffersAfterSecondMul).toEqual(2);
    tf.ENV.set('WEBGPU_IMMEDIATE_EXECUTION_ENABLED', savedFlag);
  });

  it('should recycle buffers in delayed mode', () => {
    const savedFlag = tf.ENV.get('WEBGPU_IMMEDIATE_EXECUTION_ENABLED');
    tf.ENV.set('WEBGPU_IMMEDIATE_EXECUTION_ENABLED', false);
    const backend = tf.backend() as WebGPUBackend;
    const bufferManager = backend.getBufferManager();
    bufferManager.reset();

    const a = tf.tensor2d([2, 4, 6, 8], [2, 2]);
    const b = tf.tensor2d([0.5, 0.5, 0.5, 0.5], [2, 2]);

    const c = tf.mul(a, b);
    const freeBuffersAfterFirstMul = bufferManager.getNumFreeBuffers();
    const usedBuffersAfterFirstMul = bufferManager.getNumUsedBuffers();

    const f = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    tf.matMul(c, f);
    const freeBuffersAfterFirstMatMul = bufferManager.getNumFreeBuffers();
    const usedBuffersAfterFirstMatMul = bufferManager.getNumUsedBuffers();
    expect(freeBuffersAfterFirstMatMul - freeBuffersAfterFirstMul)
        .toEqual(1);  // from released uniform
    expect(usedBuffersAfterFirstMatMul - usedBuffersAfterFirstMul).toEqual(2);

    const a2 = tf.tensor2d([2, 4, 6, 8], [2, 2]);
    const b2 = tf.tensor2d([0.5, 0.5, 0.5, 0.5], [2, 2]);

    const c2 = tf.mul(a2, b2);
    const freeBuffersAfterSecondMul = bufferManager.getNumFreeBuffers();
    const usedBuffersAfterSecondMul = bufferManager.getNumUsedBuffers();
    expect(freeBuffersAfterSecondMul - freeBuffersAfterFirstMatMul)
        .toEqual(0);  // released a uniform buffer and reused a buffer
    expect(usedBuffersAfterSecondMul - usedBuffersAfterFirstMatMul).toEqual(3);

    const f2 = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    tf.matMul(c2, f2);
    const freeBuffersAfterSecondMatMul = bufferManager.getNumFreeBuffers();
    const usedBuffersAfterSecondMatMul = bufferManager.getNumUsedBuffers();
    expect(freeBuffersAfterSecondMatMul - freeBuffersAfterSecondMul).toEqual(0);
    expect(usedBuffersAfterSecondMatMul - usedBuffersAfterSecondMul).toEqual(2);
    tf.ENV.set('WEBGPU_IMMEDIATE_EXECUTION_ENABLED', savedFlag);
  });

  it('readSync should throw if tensors are on the GPU', async () => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const b = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);

    const c = tf.matMul(a, b);
    expect(() => c.dataSync())
        .toThrowError(
            'WebGPU readSync is only available for CPU-resident tensors.');

    await c.data();
    // Now that data has been downloaded to the CPU, dataSync should work.
    expect(() => c.dataSync()).not.toThrow();
  });
});