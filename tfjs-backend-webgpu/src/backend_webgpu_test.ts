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
    const bufferManager = backend.getBufferManager();
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
    const bufferManager = backend.getBufferManager();
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

  it('should reuse textures when fromPixels have same input size', async () => {
    const useImport = tf.env().getBool('WEBGPU_USE_IMPORT');
    tf.env().set('WEBGPU_USE_IMPORT', false);
    const backend = tf.backend() as WebGPUBackend;
    const textureManager = backend.getTextureManager();
    textureManager.dispose();

    {
      const video = document.createElement('video');
      video.autoplay = true;
      const source = document.createElement('source');
      source.src =
          // tslint:disable-next-line:max-line-length
          'data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQAAAu1tZGF0AAACrQYF//+p3EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE1NSByMjkwMSA3ZDBmZjIyIC0gSC4yNjQvTVBFRy00IEFWQyBjb2RlYyAtIENvcHlsZWZ0IDIwMDMtMjAxOCAtIGh0dHA6Ly93d3cudmlkZW9sYW4ub3JnL3gyNjQuaHRtbCAtIG9wdGlvbnM6IGNhYmFjPTEgcmVmPTMgZGVibG9jaz0xOjA6MCBhbmFseXNlPTB4MzoweDExMyBtZT1oZXggc3VibWU9NyBwc3k9MSBwc3lfcmQ9MS4wMDowLjAwIG1peGVkX3JlZj0xIG1lX3JhbmdlPTE2IGNocm9tYV9tZT0xIHRyZWxsaXM9MSA4eDhkY3Q9MSBjcW09MCBkZWFkem9uZT0yMSwxMSBmYXN0X3Bza2lwPTEgY2hyb21hX3FwX29mZnNldD0tMiB0aHJlYWRzPTMgbG9va2FoZWFkX3RocmVhZHM9MSBzbGljZWRfdGhyZWFkcz0wIG5yPTAgZGVjaW1hdGU9MSBpbnRlcmxhY2VkPTAgYmx1cmF5X2NvbXBhdD0wIGNvbnN0cmFpbmVkX2ludHJhPTAgYmZyYW1lcz0zIGJfcHlyYW1pZD0yIGJfYWRhcHQ9MSBiX2JpYXM9MCBkaXJlY3Q9MSB3ZWlnaHRiPTEgb3Blbl9nb3A9MCB3ZWlnaHRwPTIga2V5aW50PTI1MCBrZXlpbnRfbWluPTEgc2NlbmVjdXQ9NDAgaW50cmFfcmVmcmVzaD0wIHJjX2xvb2thaGVhZD00MCByYz1jcmYgbWJ0cmVlPTEgY3JmPTI4LjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAAwZYiEAD//8m+P5OXfBeLGOfKE3xkODvFZuBflHv/+VwJIta6cbpIo4ABLoKBaYTkTAAAC7m1vb3YAAABsbXZoZAAAAAAAAAAAAAAAAAAAA+gAAAPoAAEAAAEAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAIYdHJhawAAAFx0a2hkAAAAAwAAAAAAAAAAAAAAAQAAAAAAAAPoAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAACgAAAAWgAAAAAAJGVkdHMAAAAcZWxzdAAAAAAAAAABAAAD6AAAAAAAAQAAAAABkG1kaWEAAAAgbWRoZAAAAAAAAAAAAAAAAAAAQAAAAEAAVcQAAAAAAC1oZGxyAAAAAAAAAAB2aWRlAAAAAAAAAAAAAAAAVmlkZW9IYW5kbGVyAAAAATttaW5mAAAAFHZtaGQAAAABAAAAAAAAAAAAAAAkZGluZgAAABxkcmVmAAAAAAAAAAEAAAAMdXJsIAAAAAEAAAD7c3RibAAAAJdzdHNkAAAAAAAAAAEAAACHYXZjMQAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAACgAFoASAAAAEgAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABj//wAAADFhdmNDAWQACv/hABhnZAAKrNlCjfkhAAADAAEAAAMAAg8SJZYBAAZo6+JLIsAAAAAYc3R0cwAAAAAAAAABAAAAAQAAQAAAAAAcc3RzYwAAAAAAAAABAAAAAQAAAAEAAAABAAAAFHN0c3oAAAAAAAAC5QAAAAEAAAAUc3RjbwAAAAAAAAABAAAAMAAAAGJ1ZHRhAAAAWm1ldGEAAAAAAAAAIWhkbHIAAAAAAAAAAG1kaXJhcHBsAAAAAAAAAAAAAAAALWlsc3QAAAAlqXRvbwAAAB1kYXRhAAAAAQAAAABMYXZmNTguMTIuMTAw';
      source.type = 'video/mp4';
      video.appendChild(source);
      document.body.appendChild(video);

      // On mobile safari the ready state is ready immediately.
      if (video.readyState < 2) {
        await new Promise(resolve => {
          video.addEventListener('loadeddata', () => resolve(video));
        });
      }
      const res = await tf.browser.fromPixelsAsync(video);
      expect(res.shape).toEqual([90, 160, 3]);
      const data = await res.data();
      expect(data.length).toEqual(90 * 160 * 3);
      const freeTexturesAfterFromPixel = textureManager.getNumFreeTextures();
      const usedTexturesAfterFromPixel = textureManager.getNumUsedTextures();
      expect(freeTexturesAfterFromPixel).toEqual(1);
      expect(usedTexturesAfterFromPixel).toEqual(0);
      document.body.removeChild(video);
    }

    {
      const img = new Image(10, 10);
      img.src = 'data:image/gif;base64' +
          ',R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==';

      await new Promise(resolve => {
        img.onload = () => resolve(img);
      });

      const resImage = await tf.browser.fromPixelsAsync(img);
      expect(resImage.shape).toEqual([10, 10, 3]);

      const dataImage = await resImage.data();
      expect(dataImage[0]).toEqual(0);
      expect(dataImage.length).toEqual(10 * 10 * 3);
      const freeTexturesAfterFromPixel = textureManager.getNumFreeTextures();
      const usedTexturesAfterFromPixel = textureManager.getNumUsedTextures();
      expect(freeTexturesAfterFromPixel).toEqual(2);
      expect(usedTexturesAfterFromPixel).toEqual(0);
    }

    {
      const img = new Image(10, 10);
      img.src =
          'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAAEnQAABJ0Ad5mH3gAAABfSURBVChTY/gPBu8NLd/KqLxT1oZw4QAqDZSDoPeWDj9WrYUIAgG6NBAhm4FFGoIgxuCUBiKgMfikv1bW4pQGav334wdUGshBk/6SVQAUh0p/mzIDTQ6oFSGNHfz/DwAwi8mNzTi6rwAAAABJRU5ErkJggg==';
      await new Promise(resolve => {
        img.onload = () => resolve(img);
      });
      const resImage = await tf.browser.fromPixelsAsync(img);
      expect(resImage.shape).toEqual([10, 10, 3]);

      const dataImage = await resImage.data();
      expect(dataImage[0]).toEqual(255);
      expect(dataImage.length).toEqual(10 * 10 * 3);
      const freeTexturesAfterFromPixel = textureManager.getNumFreeTextures();
      const usedTexturesAfterFromPixel = textureManager.getNumUsedTextures();
      expect(freeTexturesAfterFromPixel).toEqual(2);
      expect(usedTexturesAfterFromPixel).toEqual(0);
    }

    tf.env().set('WEBGPU_USE_IMPORT', useImport);
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

    const bufferManager = backend.getBufferManager();
    const t = tf.tensor1d([1, 2, 3], 'float32');

    expect(bufferManager.getNumUsedBuffers()).toBe(0);

    backend.getBuffer(t.dataId);
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
