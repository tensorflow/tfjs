/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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
import {describeWebGPU} from './test_util';

async function createReadonlyGPUBufferFromData(
    device: GPUDevice, data: number[], dtype: string) {
  const bytesPerElement = 4;
  const sizeInBytes = data.length * bytesPerElement;

  const gpuWriteBuffer = device.createBuffer({
    mappedAtCreation: true,
    size: sizeInBytes,
    usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC
  });
  const arrayBuffer = gpuWriteBuffer.getMappedRange();

  new Float32Array(arrayBuffer).set(data);

  gpuWriteBuffer.unmap();

  const aBuffer = device.createBuffer({
    mappedAtCreation: false,
    size: sizeInBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
  });

  const copyEncoder = device.createCommandEncoder();
  copyEncoder.copyBufferToBuffer(gpuWriteBuffer, 0, aBuffer, 0, sizeInBytes);

  const copyCommands = copyEncoder.finish();
  device.queue.submit([copyCommands]);
  gpuWriteBuffer.destroy();
  return aBuffer;
}

type Shape = [number]|[number, number]|[number, number, number]|
    [number, number, number, number]|[number, number, number, number, number]|
        [number, number, number, number, number, number];

describeWebGPU('tensor', () => {
  it('tensor from GPUBuffer 1d 2d 3d 4d 5d 6d', async () => {
    const device = tf.backend().getGPUDevice();
    const aData = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    const b =
        new Float32Array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]);
    const expected = [2, 4, 6, 8, 6, 8, 10, 12, 10, 12, 14, 16, 14, 16, 18, 20];
    const dtype = 'float32';
    const aBuffer = await createReadonlyGPUBufferFromData(device, aData, dtype);
    {
      const shape: Shape = [16];
      const gpuReadData = new tf.GPUReadData(aBuffer, shape, dtype);
      const a = tf.tensor1d(gpuReadData);
      const result = tf.add(a, tf.tensor1d(b, 'float32', shape));
      tf.test_util.expectArraysClose(await result.data(), expected);
      a.dispose();
      result.dispose();
    }
    {
      const shape: Shape = [8, 2];
      const gpuReadData = new tf.GPUReadData(aBuffer, shape, dtype);
      const a = tf.tensor2d(gpuReadData);
      const result = tf.add(a, tf.tensor2d(b, shape));
      tf.test_util.expectArraysClose(await result.data(), expected);
      a.dispose();
      result.dispose();
    }
    {
      const shape: Shape = [2, 4, 2];
      const gpuReadData = new tf.GPUReadData(aBuffer, shape, dtype);
      const a = tf.tensor3d(gpuReadData);
      const result = tf.add(a, tf.tensor3d(b, shape));
      tf.test_util.expectArraysClose(await result.data(), expected);
      a.dispose();
      result.dispose();
    }
    {
      const shape: Shape = [2, 2, 2, 2];
      const gpuReadData = new tf.GPUReadData(aBuffer, shape, dtype);
      const a = tf.tensor4d(gpuReadData);
      const result = tf.add(a, tf.tensor4d(b, shape));
      tf.test_util.expectArraysClose(await result.data(), expected);
      a.dispose();
      result.dispose();
    }
    {
      const shape: Shape = [1, 2, 2, 2, 2];
      const gpuReadData = new tf.GPUReadData(aBuffer, shape, dtype);
      const a = tf.tensor5d(gpuReadData);
      const result = tf.add(a, tf.tensor5d(b, shape));
      tf.test_util.expectArraysClose(await result.data(), expected);
      a.dispose();
      result.dispose();
    }
    {
      const shape: Shape = [1, 1, 2, 2, 2, 2];
      const gpuReadData = new tf.GPUReadData(aBuffer, shape, dtype);
      const a = tf.tensor6d(gpuReadData);
      const result = tf.add(a, tf.tensor6d(b, shape));
      tf.test_util.expectArraysClose(await result.data(), expected);
      a.dispose();
      result.dispose();
    }
    aBuffer.destroy();
  });

  it('two tensors share the same GPUBuffer', async () => {
    const device = tf.backend().getGPUDevice();
    const aData = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    const expected =
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32];
    const dtype = 'float32';
    const aBuffer = await createReadonlyGPUBufferFromData(device, aData, dtype);

    const shape: Shape = [16];
    const gpuReadData = new tf.GPUReadData(aBuffer, shape, dtype);
    const a = tf.tensor1d(gpuReadData);
    const b = tf.tensor1d(gpuReadData);
    const result = tf.add(a, b);
    tf.test_util.expectArraysClose(await result.data(), expected);
    a.dispose();
    result.dispose();
    aBuffer.destroy();
  });
});
