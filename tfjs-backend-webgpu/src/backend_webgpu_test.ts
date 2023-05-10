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
    expect(endNumBytesInGPU - startNumBytesInGPU).toEqual(64);

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
    expect(usedBuffersAfterFirstMatMul - usedBuffersAfterFirstMul).toEqual(3);

    const a2 = tf.tensor2d([2, 4, 6, 8], [2, 2]);
    const b2 = tf.tensor2d([0.5, 0.5, 0.5, 0.5], [2, 2]);

    const c2 = tf.mul(a2, b2);
    const freeBuffersAfterSecondMul = bufferManager.getNumFreeBuffers();
    const usedBuffersAfterSecondMul = bufferManager.getNumUsedBuffers();
    expect(freeBuffersAfterSecondMul - freeBuffersAfterFirstMatMul).toEqual(0);
    expect(usedBuffersAfterSecondMul - usedBuffersAfterFirstMatMul).toEqual(4);

    const f2 = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const c3 = tf.matMul(c2, f2);
    const freeBuffersAfterSecondMatMul = bufferManager.getNumFreeBuffers();
    const usedBuffersAfterSecondMatMul = bufferManager.getNumUsedBuffers();
    expect(freeBuffersAfterSecondMatMul - freeBuffersAfterSecondMul).toEqual(0);
    expect(usedBuffersAfterSecondMatMul - usedBuffersAfterSecondMul).toEqual(3);

    // Tests happen within a tidy so we need to read a tensor at the end of a
    // test in delayed mode in order to force flush the disposal queue.
    await c3.data();
    tf.env().set('WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE', savedFlag);
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
    expect(bufferManager.getNumUsedBuffers()).toBe(1);
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
    expectArraysEqual(res.buffer.size, size);
    if (res.tensorRef.dtype !== 'float32') {
      throw new Error(
          `Unexpected type. Actual: ${res.tensorRef.dtype}. ` +
          `Expected: float32`);
    }
    const resData = await webGPUBackend.getBufferData(res.buffer);
    const values = tf.util.convertBackendValuesAndArrayBuffer(
        resData, res.tensorRef.dtype);
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
    expectArraysEqual(res.buffer.size, size);
    if (res.tensorRef.dtype !== 'int32') {
      throw new Error(
          `Unexpected type. Actual: ${res.tensorRef.dtype}. ` +
          `Expected: float32`);
    }
    const resData = await webGPUBackend.getBufferData(res.buffer);
    const values = tf.util.convertBackendValuesAndArrayBuffer(
        resData, res.tensorRef.dtype);
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
      return b.dataToGPU() as unknown as tf.Tensor;
    });

    const endTensor = tf.memory().numTensors;
    const endDataBuckets = webGPUBackend.numDataIds();

    expect(endTensor).toEqual(startTensor + 1);
    expect(endDataBuckets).toEqual(startDataBuckets + 1);

    const res = result as unknown as GPUData;
    const resData = await webGPUBackend.getBufferData(res.buffer);
    const values = tf.util.convertBackendValuesAndArrayBuffer(
        resData, res.tensorRef.dtype);
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

async function parallelCompilationCommon(webGPUBackend: WebGPUBackend) {
  const startNumBytes = (tf.memory() as WebGPUMemoryInfo).numBytesInGPU;
  const startTensor = tf.memory().numTensors;
  const startDataBuckets = webGPUBackend.numDataIds();

  const a1 = tf.tensor1d([1, 1, 1]);
  const b1 = tf.tensor1d([1, 1, 1]);

  // Parallel compile.
  tf.env().set('WEBGPU_ENGINE_COMPILE_ONLY', true);
  const c1 = tf.add(a1, b1);
  await webGPUBackend.checkCompileCompletionAsync();

  // Actual inference.
  tf.env().set('WEBGPU_ENGINE_COMPILE_ONLY', false);
  const c2 = tf.add(a1, b1);
  expectArraysEqual(await c2.data(), [2, 2, 2]);

  tf.dispose([a1, b1, c1, c2]);
  const endNumBytes = (tf.memory() as WebGPUMemoryInfo).numBytesInGPU;
  const endTensor = tf.memory().numTensors;
  const endDataBuckets = webGPUBackend.numDataIds();

  // We only check numBytesInGPU. For parallel compilation,
  // numBytesInGPUAllocated will be more because of the two pass
  // uploadToGPU, but they will all be freed, resulting in endNumbytes equal
  // to startNumBytes.
  expect(startNumBytes).toEqual(endNumBytes);
  expect(startTensor).toEqual(endTensor);
  expect(endDataBuckets).toEqual(startDataBuckets);
}

describeWebGPU('parallel compilation', () => {
  let prevBackend: string;
  let savedWebGPUCPUForward: boolean;
  let savedEngineCompileOnly: boolean;
  let webGPUBackend: WebGPUBackend;
  const customWebGPUBackendName = 'test-parallel';

  beforeAll(() => {
    prevBackend = tf.getBackend();
  });

  beforeEach(async () => {
    const adapter = await navigator.gpu.requestAdapter({});
    const device = await adapter.requestDevice({});
    webGPUBackend = new WebGPUBackend(device);

    tf.copyRegisteredKernels('webgpu', customWebGPUBackendName);
    tf.registerBackend(customWebGPUBackendName, () => webGPUBackend);
    tf.setBackend('test-parallel');

    savedWebGPUCPUForward = tf.env().get('WEBGPU_CPU_FORWARD') as boolean;
    savedEngineCompileOnly =
        tf.env().get('WEBGPU_ENGINE_COMPILE_ONLY') as boolean;
    tf.env().set('WEBGPU_CPU_FORWARD', false);
    await tf.ready();
  });

  afterEach(() => {
    tf.env().set('WEBGPU_CPU_FORWARD', savedWebGPUCPUForward);
    tf.env().set('WEBGPU_ENGINE_COMPILE_ONLY', savedEngineCompileOnly);
    tf.setBackend(prevBackend);
    tf.removeBackend(customWebGPUBackendName);
  });

  it('should work if pipeline cache not exist.', async () => {
    await parallelCompilationCommon(webGPUBackend);
  });

  it('should work if pipeline cache exists.', async () => {
    // This will create pipeline cache.
    const a0 = tf.tensor1d([1, 1, 1]);
    const b0 = tf.tensor1d([1, 1, 1]);
    const c0 = tf.add(a0, b0);
    const data = await c0.data();
    expectArraysClose(data, [2, 2, 2]);

    await parallelCompilationCommon(webGPUBackend);
  });

  it('should work when running parallel compile again', async () => {
    // This will create pipeline cache.
    const a0 = tf.tensor1d([1, 1, 1]);
    const b0 = tf.tensor1d([1, 1, 1]);
    const c0 = tf.add(a0, b0);
    const data = await c0.data();
    expectArraysClose(data, [2, 2, 2]);

    await parallelCompilationCommon(webGPUBackend);
    await parallelCompilationCommon(webGPUBackend);
  });

  it('should not work if not call checkCompileCompletionAsync', async () => {
    const a1 = tf.tensor1d([1, 1, 1]);
    const b1 = tf.tensor1d([1, 1, 1]);

    // Parallel compile but not call await (tf.backend() as
    // WebGPUBackend).checkCompileCompletionAsync().
    tf.env().set('WEBGPU_ENGINE_COMPILE_ONLY', true);
    tf.add(a1, b1);

    // Actual inference.
    tf.env().set('WEBGPU_ENGINE_COMPILE_ONLY', false);
    expect(() => tf.add(a1, b1))
        .toThrowError(
            'Please call checkCompileCompletionAsync to ensure parallel compilation is done!');
  });

  it('read data is invalid if parallel compilation is true', async () => {
    const a1 = tf.tensor1d([1, 1, 1]);
    const b1 = tf.tensor1d([1, 1, 1]);

    // Parallel compile.
    tf.env().set('WEBGPU_ENGINE_COMPILE_ONLY', true);
    const c1 = tf.add(a1, b1);
    await (tf.backend() as WebGPUBackend).checkCompileCompletionAsync();
    // Read data is invalid.
    expectArraysClose((await c1.data()).length, 0);
  });

  it('checkCompileCompletionAsync is nop if parallel compilation is false',
     async () => {
       const a1 = tf.tensor1d([1, 1, 1]);
       const b1 = tf.tensor1d([1, 1, 1]);
       // If parallel compilation is false, checkCompileCompletionAsync is nop.
       tf.env().set('WEBGPU_ENGINE_COMPILE_ONLY', false);
       const c1 = tf.add(a1, b1);
       await (tf.backend() as WebGPUBackend).checkCompileCompletionAsync();
       expectArraysClose(await c1.data(), [2, 2, 2]);
     });
});

function createStagingGPUBufferFromData(
    device: GPUDevice, data: number[], dtype: tf.DataType) {
  const bytesPerElement = 4;
  const sizeInBytes = data.length * bytesPerElement;

  const gpuWriteBuffer = device.createBuffer({
    mappedAtCreation: true,
    size: sizeInBytes,
    usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC
  });
  const arrayBuffer = gpuWriteBuffer.getMappedRange();
  if (dtype === 'float32') {
    new Float32Array(arrayBuffer).set(data);
  } else if (dtype === 'int32') {
    new Int32Array(arrayBuffer).set(data);
  } else {
    throw new Error(
        `Creating tensor from GPUBuffer only supports` +
        `'float32'|'int32' dtype, while the dtype is ${dtype}.`);
  }
  gpuWriteBuffer.unmap();
  return gpuWriteBuffer;
}

function createGPUBufferFromData(
    device: GPUDevice, data: number[], dtype: tf.DataType,
    bufferUsage = GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC) {
  const bytesPerElement = 4;
  const sizeInBytes = data.length * bytesPerElement;

  const gpuWriteBuffer = createStagingGPUBufferFromData(device, data, dtype);
  const gpuReadBuffer = device.createBuffer(
      {mappedAtCreation: false, size: sizeInBytes, usage: bufferUsage});

  const copyEncoder = device.createCommandEncoder();
  copyEncoder.copyBufferToBuffer(
      gpuWriteBuffer, 0, gpuReadBuffer, 0, sizeInBytes);
  const copyCommands = copyEncoder.finish();
  device.queue.submit([copyCommands]);
  gpuWriteBuffer.destroy();
  return gpuReadBuffer;
}

async function testCreateTensorFromGPUBuffer(
    dtype: tf.DataType, useDefaultShapeAndType = false, zeroCopy = false) {
  const webGPUBackend = tf.backend() as WebGPUBackend;
  const device = webGPUBackend.device;
  const aData = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
  const bData = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4];
  const expected = [2, 4, 6, 8, 6, 8, 10, 12, 10, 12, 14, 16, 14, 16, 18, 20];
  const aBuffer = createGPUBufferFromData(device, aData, dtype);
  const shape: number[] = [aData.length];
  const startNumBytes = tf.memory().numBytes;
  const startNumTensors = tf.memory().numTensors;
  const webGPUData = {buffer: aBuffer, zeroCopy};
  const a = useDefaultShapeAndType ? tf.tensor(webGPUData) :
                                     tf.tensor(webGPUData, shape, dtype);
  if (zeroCopy !== true) {
    aBuffer.destroy();
  }
  const b = tf.tensor(bData, shape, dtype);
  const result = tf.add(a, b);
  tf.test_util.expectArraysClose(await result.data(), expected);
  a.dispose();
  b.dispose();
  result.dispose();
  const endNumBytes = tf.memory().numBytes;
  const endNumTensors = tf.memory().numTensors;
  expect(endNumBytes - startNumBytes).toEqual(0);
  expect(endNumTensors - startNumTensors).toEqual(0);
  if (zeroCopy === true) {
    aBuffer.destroy();
  }
}

function createTensorFromGPUTest(zeroCopy = false) {
  it('use default shape and data type(float32)', async () => {
    await testCreateTensorFromGPUBuffer('float32', true, zeroCopy);
  });

  it('work for float32', async () => {
    await testCreateTensorFromGPUBuffer('float32', false, zeroCopy);
  });

  it('work for int32', async () => {
    await testCreateTensorFromGPUBuffer('int32', false, zeroCopy);
  });

  it('work for read', async () => {
    const webGPUBackend = tf.backend() as WebGPUBackend;
    const device = webGPUBackend.device;
    const aData = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    const dtype = 'float32';
    const aBuffer = createGPUBufferFromData(device, aData, dtype);
    const shape: number[] = [aData.length];
    const a = tf.tensor({buffer: aBuffer, zeroCopy}, shape, dtype);
    if (zeroCopy !== true) {
      aBuffer.destroy();
    }
    await a.data();
    if (zeroCopy === true) {
      aBuffer.destroy();
    }
  });

  it('two tensors share the same GPUBuffer', async () => {
    const webGPUBackend = tf.backend() as WebGPUBackend;
    const device = webGPUBackend.device;
    const aData = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    const dtype = 'float32';
    const aBuffer = createGPUBufferFromData(device, aData, dtype);
    const startNumBytes = tf.memory().numBytes;
    const startNumTensors = tf.memory().numTensors;
    const shape: number[] = [aData.length];
    const webGPUData = {buffer: aBuffer, zeroCopy};
    const a = tf.tensor(webGPUData, shape, dtype);
    const b = tf.tensor(webGPUData, shape, dtype);
    if (zeroCopy !== true) {
      aBuffer.destroy();
    }
    const result = tf.add(a, b);
    const expected =
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32];
    tf.test_util.expectArraysClose(await result.data(), expected);
    a.dispose();
    b.dispose();
    result.dispose();
    const endNumBytes = tf.memory().numBytes;
    const endNumTensors = tf.memory().numTensors;
    expect(endNumBytes - startNumBytes).toEqual(0);
    expect(endNumTensors - startNumTensors).toEqual(0);
    if (zeroCopy === true) {
      aBuffer.destroy();
    }
  });

  it('GPUBuffer size is bigger than tensor size', async () => {
    const webGPUBackend = tf.backend() as WebGPUBackend;
    const device = webGPUBackend.device;
    const aData = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    const dtype = 'float32';
    const aBuffer = createGPUBufferFromData(device, aData, dtype);
    const startNumBytes = tf.memory().numBytes;
    const startNumTensors = tf.memory().numTensors;
    // GPUBuffer.size is bigger than shape size
    const shape: number[] = [aData.length - 1];
    const webGPUData = {buffer: aBuffer, zeroCopy};
    const a = tf.tensor(webGPUData, shape, dtype);
    const b = tf.tensor(webGPUData, shape, dtype);
    if (zeroCopy !== true) {
      aBuffer.destroy();
    }
    const result = tf.add(a, b);
    const expected = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30];
    tf.test_util.expectArraysClose(await result.data(), expected);
    a.dispose();
    b.dispose();
    result.dispose();
    const endNumBytes = tf.memory().numBytes;
    const endNumTensors = tf.memory().numTensors;
    expect(endNumBytes - startNumBytes).toEqual(0);
    expect(endNumTensors - startNumTensors).toEqual(0);
    if (zeroCopy === true) {
      aBuffer.destroy();
    }
  });

  it('throw when GPUBuffer size is smaller than tensor size', async () => {
    const webGPUBackend = tf.backend() as WebGPUBackend;
    const device = webGPUBackend.device;
    const aData = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    const dtype = 'float32';
    const aBuffer = createGPUBufferFromData(device, aData, dtype);
    // Throw when GPUBuffer.size is smaller than shape size
    const shape: number[] = [aData.length + 1];
    const a = () => tf.tensor({buffer: aBuffer}, shape, dtype);
    expect(a).toThrowError();
    aBuffer.destroy();
  });

  it('throw when GPUBuffer usage is not correct', async () => {
    const webGPUBackend = tf.backend() as WebGPUBackend;
    const device = webGPUBackend.device;
    const aData = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    const dtype = 'float32';
    // Create a GPUBuffer without GPUBufferUsage.STORAGE.
    const aBuffer = createStagingGPUBufferFromData(device, aData, dtype);
    // Throw when GPUBuffer usage is not correct.
    const shape: number[] = [aData.length];
    const a = () => tf.tensor({buffer: aBuffer, zeroCopy}, shape, dtype);
    expect(a).toThrowError();
    aBuffer.destroy();
  });
}

describeWebGPU('create tensor from GPUBuffer', () => {
  createTensorFromGPUTest();
});

describeWebGPU('create tensor from GPUBuffer with zero copy', () => {
  createTensorFromGPUTest(true);
});
