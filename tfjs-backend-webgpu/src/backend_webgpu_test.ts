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

import {GPUData, test_util} from '@tensorflow/tfjs-core';
const {expectArraysEqual, expectArraysClose} = test_util;

import {WebGPUBackend, WebGPUMemoryInfo} from './backend_webgpu';
import {describeWebGPU} from './test_util';
import * as webgpu_util from './webgpu_util';

describeWebGPU('backend webgpu cpu forwarding turned on', () => {
  let cpuForwardFlagSaved: boolean;

  beforeAll(() => {
    cpuForwardFlagSaved = tf.env().getBool('WEBGPU_CPU_FORWARD');

    tf.env().set('WEBGPU_CPU_FORWARD', true);
  });

  afterAll(() => {
    tf.env().set('WEBGPU_CPU_FORWARD', cpuForwardFlagSaved);
  });

  it('should not allocate GPU memory when CPU forwarding', async () => {
    const a = tf.tensor2d([2, 4, 6, 8], [2, 2]);
    const b = tf.tensor2d([0.5, 0.5, 0.5, 0.5], [2, 2]);

    const c = tf.mul(a, b);

    const startNumBytes = tf.memory().numBytes;
    const startNumTensors = tf.memory().numTensors;
    const startNumBytesInGPU = (tf.memory() as WebGPUMemoryInfo).numBytesInGPU;

    expect(startNumBytes).toEqual(48);
    expect(startNumTensors).toEqual(3);
    expect(startNumBytesInGPU).toEqual(0);

    const f = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const d = tf.matMul(c, f);

    const dData = await d.data();
    const endNumBytes = tf.memory().numBytes;
    const endNumTensors = tf.memory().numTensors;
    const endNumBytesInGPU = (tf.memory() as WebGPUMemoryInfo).numBytesInGPU;

    expect(endNumBytes - startNumBytes).toEqual(48);
    expect(endNumTensors - startNumTensors).toEqual(2);
    expect(endNumBytesInGPU - startNumBytesInGPU).toEqual(40);

    expectArraysClose(dData, new Float32Array([9, 12, 15, 19, 26, 33]));
  });
});

describeWebGPU('backend webgpu', () => {
  it('should not leak memory in delayed mode', async () => {
    const savedFlag = tf.env().get('WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE');
    tf.env().set('WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE', 15);
    const a = tf.tensor2d([2, 4, 6, 8], [2, 2]);
    const b = tf.tensor2d([0.5, 0.5, 0.5, 0.5], [2, 2]);

    const c = tf.mul(a, b);

    const startNumBytes = tf.memory().numBytes;
    const startNumTensors = tf.memory().numTensors;

    const f = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const d = tf.matMul(c, f);

    const dData = await d.data();
    const endNumBytes = tf.memory().numBytes;
    const endNumTensors = tf.memory().numTensors;

    expect(endNumBytes - startNumBytes).toEqual(48);
    expect(endNumTensors - startNumTensors).toEqual(2);

    expectArraysClose(dData, new Float32Array([9, 12, 15, 19, 26, 33]));
    tf.env().set('WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE', savedFlag);
  });

  it('should not leak memory in immediate mode', async () => {
    const savedFlag = tf.env().get('WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE');
    tf.env().set('WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE', 1);
    const a = tf.tensor2d([2, 4, 6, 8], [2, 2]);
    const b = tf.tensor2d([0.5, 0.5, 0.5, 0.5], [2, 2]);

    const c = tf.mul(a, b);

    const startNumBytes = tf.memory().numBytes;
    const startNumTensors = tf.memory().numTensors;

    const f = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const d = tf.matMul(c, f);

    const dData = await d.data();
    const endNumBytes = tf.memory().numBytes;
    const endNumTensors = tf.memory().numTensors;

    expect(endNumBytes - startNumBytes).toEqual(48);
    expect(endNumTensors - startNumTensors).toEqual(2);

    expectArraysClose(dData, new Float32Array([9, 12, 15, 19, 26, 33]));
    tf.env().set('WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE', savedFlag);
  });

  it('should recycle buffers in immediate mode', () => {
    const savedFlag = tf.env().get('WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE');
    tf.env().set('WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE', 1);
    const backend = tf.backend() as WebGPUBackend;
    const bufferManager = backend.bufferManager;
    bufferManager.dispose();

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
    expect(usedBuffersAfterFirstMatMul - usedBuffersAfterFirstMul).toEqual(3);

    const a2 = tf.tensor2d([2, 4, 6, 8], [2, 2]);
    const b2 = tf.tensor2d([0.5, 0.5, 0.5, 0.5], [2, 2]);

    const c2 = tf.mul(a2, b2);
    const freeBuffersAfterSecondMul = bufferManager.getNumFreeBuffers();
    const usedBuffersAfterSecondMul = bufferManager.getNumUsedBuffers();
    expect(freeBuffersAfterSecondMul - freeBuffersAfterFirstMatMul)
        .toEqual(0);  // released a uniform buffer and reused a buffer
    expect(usedBuffersAfterSecondMul - usedBuffersAfterFirstMatMul).toEqual(5);

    const f2 = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    tf.matMul(c2, f2);
    const freeBuffersAfterSecondMatMul = bufferManager.getNumFreeBuffers();
    const usedBuffersAfterSecondMatMul = bufferManager.getNumUsedBuffers();
    expect(freeBuffersAfterSecondMatMul - freeBuffersAfterSecondMul).toEqual(0);
    expect(usedBuffersAfterSecondMatMul - usedBuffersAfterSecondMul).toEqual(3);
    tf.env().set('WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE', savedFlag);
  });

  it('should not recycle buffers in delayed mode', async () => {
    const savedFlag = tf.env().get('WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE');
    tf.env().set('WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE', 15);
    const backend = tf.backend() as WebGPUBackend;
    const bufferManager = backend.bufferManager;
    bufferManager.dispose();

    const a = tf.tensor2d([2, 4, 6, 8], [2, 2]);
    const b = tf.tensor2d([0.5, 0.5, 0.5, 0.5], [2, 2]);

    const c = tf.mul(a, b);
    const freeBuffersAfterFirstMul = bufferManager.getNumFreeBuffers();
    const usedBuffersAfterFirstMul = bufferManager.getNumUsedBuffers();

    const f = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    tf.matMul(c, f);
    const freeBuffersAfterFirstMatMul = bufferManager.getNumFreeBuffers();
    const usedBuffersAfterFirstMatMul = bufferManager.getNumUsedBuffers();
    expect(freeBuffersAfterFirstMatMul - freeBuffersAfterFirstMul).toEqual(0);
    expect(usedBuffersAfterFirstMatMul - usedBuffersAfterFirstMul).toEqual(4);

    const a2 = tf.tensor2d([2, 4, 6, 8], [2, 2]);
    const b2 = tf.tensor2d([0.5, 0.5, 0.5, 0.5], [2, 2]);

    const c2 = tf.mul(a2, b2);
    const freeBuffersAfterSecondMul = bufferManager.getNumFreeBuffers();
    const usedBuffersAfterSecondMul = bufferManager.getNumUsedBuffers();
    expect(freeBuffersAfterSecondMul - freeBuffersAfterFirstMatMul).toEqual(0);
    expect(usedBuffersAfterSecondMul - usedBuffersAfterFirstMatMul).toEqual(6);

    const f2 = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const c3 = tf.matMul(c2, f2);
    const freeBuffersAfterSecondMatMul = bufferManager.getNumFreeBuffers();
    const usedBuffersAfterSecondMatMul = bufferManager.getNumUsedBuffers();
    expect(freeBuffersAfterSecondMatMul - freeBuffersAfterSecondMul).toEqual(0);
    expect(usedBuffersAfterSecondMatMul - usedBuffersAfterSecondMul).toEqual(4);

    // Tests happen within a tidy so we need to read a tensor at the end of a
    // test in delayed mode in order to force flush the disposal queue.
    await c3.data();
    tf.env().set('WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE', savedFlag);
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

describeWebGPU('backendWebGPU', () => {
  let prevBackend: string;

  beforeAll(() => {
    prevBackend = tf.getBackend();
  });

  afterEach(() => {
    tf.setBackend(prevBackend);
    tf.removeBackend('test-storage');
  });

  it('lazily upload', async () => {
    const adapter = await navigator.gpu.requestAdapter({});
    const device = await adapter.requestDevice({});
    const backend = new WebGPUBackend(device);
    tf.registerBackend('test-storage', () => backend);
    tf.setBackend('test-storage');

    const bufferManager = backend.bufferManager;
    const t = tf.tensor1d([1, 2, 3], 'float32');

    expect(bufferManager.getNumUsedBuffers()).toBe(0);

    backend.uploadToGPU(t.dataId);
    expect(bufferManager.getNumUsedBuffers())
        .toBe(
            2);  // One is the storage buffer, the other is the staging buffer.
  });
});

describeWebGPU('keeping data on gpu ', () => {
  let flag: boolean;

  beforeAll(() => {
    flag = tf.env().getBool('WEBGPU_CPU_FORWARD');
    tf.env().set('WEBGPU_CPU_FORWARD', false);
  });

  afterAll(() => {
    tf.env().set('WEBGPU_CPU_FORWARD', flag);
  });

  it('has a valid buffer for dtype=float32.', async () => {
    const webGPUBackend = (tf.backend() as WebGPUBackend);
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    const size = 48;
    const a = tf.tensor(data, [1, 3, 4]);
    const b = tf.add(a, 0);
    const res = b.dataToGPU();
    expectArraysEqual(res.bufSize, size);
    if (res.tensorRef.dtype !== 'float32') {
      throw new Error(
          `Unexpected type. Actual: ${res.tensorRef.dtype}. ` +
          `Expected: float32`);
    }
    const resData = await webGPUBackend.getBufferData(res.buffer, res.bufSize);
    const values = webgpu_util.ArrayBufferToTypedArray(
        resData as ArrayBuffer, res.tensorRef.dtype);
    expectArraysEqual(values, data);
  });

  it('has a valid buffer for dtype=int32.', async () => {
    const webGPUBackend = (tf.backend() as WebGPUBackend);
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    const size = 48;
    const a = tf.tensor(data, [1, 3, 4], 'int32');
    const b = tf.tensor([0], [1], 'int32');
    const c = tf.add(a, b);
    const res = c.dataToGPU();
    expectArraysEqual(res.bufSize, size);
    if (res.tensorRef.dtype !== 'int32') {
      throw new Error(
          `Unexpected type. Actual: ${res.tensorRef.dtype}. ` +
          `Expected: float32`);
    }
    const resData = await webGPUBackend.getBufferData(res.buffer, res.bufSize);
    const values = webgpu_util.ArrayBufferToTypedArray(
        resData as ArrayBuffer, res.tensorRef.dtype);
    expectArraysEqual(values, data);
  });

  it('has no memory leak.', () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

    const a = tf.tensor(data, [1, 3, 4]);
    const b = tf.add(a, 0);

    const webGPUBackend = tf.backend() as WebGPUBackend;
    const startTensor = tf.memory().numTensors;
    const startDataBuckets = webGPUBackend.numDataIds();

    const res = b.dataToGPU();
    res.tensorRef.dispose();

    const endTensor = tf.memory().numTensors;
    const endDataBuckets = webGPUBackend.numDataIds();

    expect(endTensor).toEqual(startTensor);
    expect(endDataBuckets).toEqual(startDataBuckets);
  });

  it('can be used in tidy.', async () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

    const webGPUBackend = tf.backend() as WebGPUBackend;
    const startTensor = tf.memory().numTensors;
    const startDataBuckets = webGPUBackend.numDataIds();

    const result = tf.tidy(() => {
      const a = tf.tensor(data, [1, 3, 4]);
      const b = tf.add(a, 0);
      return b.dataToGPU() as {} as tf.Tensor;
    });

    const endTensor = tf.memory().numTensors;
    const endDataBuckets = webGPUBackend.numDataIds();

    expect(endTensor).toEqual(startTensor + 1);
    expect(endDataBuckets).toEqual(startDataBuckets + 1);

    const res = result as {} as GPUData;
    const resData = await webGPUBackend.getBufferData(res.buffer, res.bufSize);
    const values = webgpu_util.ArrayBufferToTypedArray(
        resData as ArrayBuffer, res.tensorRef.dtype);
    expectArraysEqual(values, data);
  });

  it('tidy has no memory leak.', () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

    const webGPUBackend = tf.backend() as WebGPUBackend;
    const startTensor = tf.memory().numTensors;
    const startDataBuckets = webGPUBackend.numDataIds();

    tf.tidy(() => {
      const a = tf.tensor(data, [1, 3, 4]);
      const b = tf.add(a, 0);
      b.dataToGPU();
      return b;
    });

    const endTensor = tf.memory().numTensors;
    const endDataBuckets = webGPUBackend.numDataIds();

    expect(endTensor).toEqual(startTensor + 1);
    expect(endDataBuckets).toEqual(startDataBuckets + 1);
  });
});
