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
import glslangInit from '@webgpu/glslang/dist/web-devel/glslang.onefile';

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
    const savedFlag = tf.env().get('WEBGPU_IMMEDIATE_EXECUTION_ENABLED');
    tf.env().set('WEBGPU_IMMEDIATE_EXECUTION_ENABLED', true);

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

    tf.test_util.expectArraysClose(
        dData, new Float32Array([9, 12, 15, 19, 26, 33]));
    tf.env().set('WEBGPU_IMMEDIATE_EXECUTION_ENABLED', savedFlag);
  });
});

describeWebGPU('backend webgpu', () => {
  it('should not leak memory in delayed mode', async () => {
    const savedFlag = tf.env().get('WEBGPU_IMMEDIATE_EXECUTION_ENABLED');
    tf.env().set('WEBGPU_IMMEDIATE_EXECUTION_ENABLED', false);
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
    expect(endNumBytesInGPU - startNumBytesInGPU).toEqual(0);

    tf.test_util.expectArraysClose(
        dData, new Float32Array([9, 12, 15, 19, 26, 33]));
    tf.env().set('WEBGPU_IMMEDIATE_EXECUTION_ENABLED', savedFlag);
  });

  it('should not leak memory in immediate mode', async () => {
    const savedFlag = tf.env().get('WEBGPU_IMMEDIATE_EXECUTION_ENABLED');
    tf.env().set('WEBGPU_IMMEDIATE_EXECUTION_ENABLED', true);
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
    expect(endNumBytesInGPU - startNumBytesInGPU).toEqual(24);

    tf.test_util.expectArraysClose(
        dData, new Float32Array([9, 12, 15, 19, 26, 33]));
    tf.env().set('WEBGPU_IMMEDIATE_EXECUTION_ENABLED', savedFlag);
  });

  it('should recycle buffers in immediate mode', () => {
    const savedFlag = tf.env().get('WEBGPU_IMMEDIATE_EXECUTION_ENABLED');
    tf.env().set('WEBGPU_IMMEDIATE_EXECUTION_ENABLED', true);
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
    tf.env().set('WEBGPU_IMMEDIATE_EXECUTION_ENABLED', savedFlag);
  });

  it('should not recycle buffers in delayed mode', async () => {
    const savedFlag = tf.env().get('WEBGPU_IMMEDIATE_EXECUTION_ENABLED');
    tf.env().set('WEBGPU_IMMEDIATE_EXECUTION_ENABLED', false);
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
    tf.env().set('WEBGPU_IMMEDIATE_EXECUTION_ENABLED', savedFlag);
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

  it('lazily upload', async () => {
    const glslang = await glslangInit();
    const adapter = await navigator.gpu.requestAdapter({});
    const device = await adapter.requestDevice({});
    const backend = new WebGPUBackend(device, glslang);
    tf.registerBackend('test-storage', () => backend);
    tf.setBackend('test-storage');

    const bufferManager = backend.getBufferManager();
    const t = tf.tensor1d([1, 2, 3], 'float32');

    expect(bufferManager.getNumUsedBuffers()).toBe(0);

    backend.getBuffer(t.dataId);
    expect(bufferManager.getNumUsedBuffers()).toBe(1);
  });

  it('should be possible to move data from webgl to webgpu', async () => {
    tf.setBackend('webgl');
    const a = tf.randomNormal([1, 65, 65, 256]);
    const b = tf.randomNormal([1, 65, 65, 256]);
    const c = tf.add(a, b);
    await c.data();

    const f = async () => {
      tf.setBackend('webgpu');
      const d = tf.add(a, b);
      await d.data();
    };

    expect(f).not.toThrow();
  });

  fit('x=[2,2,1] f=[2,2,1,1] s=1 d=1 p=0', async () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const outputDepth = 1;
    const fSize = 2;
    const pad = 'same';
    const stride = 1;
    const dataFormat = 'NHWC';
    const dilation = 1;

    const x = tf.tensor3d([1, 2, 3, 4], inputShape);
    const w =
        tf.tensor4d([3, 1, 5, 0], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.conv2d(x, w, stride, pad, dataFormat, dilation);

    const resultData = await result.data();
    console.log(resultData);
    // expect(result.shape).toEqual([2, 2, 1]);
    // expectArraysClose(resultData, new Float32Array([20, 26, 13, 12]));
  });

  // async function time(
  //     doRep: (r: number) => tf.Tensor[] | tf.Tensor,
  //     endTrial?: () => Promise<void>, disposeAfterEachTrial = false,
  //     trials = 50, reps = 1) {
  //   const times = [];

  //   let toDispose: tf.Tensor[] = [];
  //   const dispose = () => {
  //     for (const t of toDispose) {
  //       t.dispose();
  //     }
  //     toDispose = [];
  //   };

  //   const trial = async () => {
  //     let result;
  //     for (let r = 0; r < reps; ++r) {
  //       result = doRep(r);

  //       toDispose = toDispose.concat(Array.isArray(result) ? result :
  //       [result]);
  //     }

  //     if (endTrial != null) {
  //       await endTrial();
  //     } else {
  //       await (Array.isArray(result) ? result[0] : result).data();
  //     }
  //   };

  //   // Warm-up. Specifically, this pre-allocates enough memory for an entire
  //   // trial, ensuring that no allocations happen when timing a trial (if the
  //   // backend reuses allocations).
  //   await trial();
  //   dispose();

  //   for (let t = 0; t < trials; ++t) {
  //     const start = tf.util.now();
  //     await trial();
  //     times.push(tf.util.now() - start);
  //     if (disposeAfterEachTrial) {
  //       dispose();
  //     }
  //   }

  //   const mean = times.reduce((a, b) => a + b, 0) / trials;
  //   const min = Math.min(...times);
  //   const fmt = (n: number) => n.toFixed(3);
  //   console.log(`Mean time: ${fmt(mean)} ms -> ${fmt(mean / reps)} / rep`);
  //   console.log(`Min time: ${fmt(min)} ms -> ${fmt(min / reps)} / rep`);
  // }

  // fit('conv2d', async () => {
  //   const a = tf.randomNormal<tf.Rank.R4>([1, 128, 128, 4]);
  //   const b = tf.randomNormal<tf.Rank.R4>([25, 25, 4, 4]);

  // const result = tf.conv2d(a, b, 1, 'same');
  // const data = await result.data();
  // console.log(data);
  //   await time(() => tf.conv2d(a, b, 1, 'same'));
  // });

  // it('conv2d', async () => {
  //   const a = tf.randomNormal<tf.Rank.R4>([1, 263, 263, 3]);
  //   const b = tf.randomNormal<tf.Rank.R4>([7, 7, 3, 64]);

  //   await time(() => tf.conv2d(a, b, 1, 'same'));
  // });
});
