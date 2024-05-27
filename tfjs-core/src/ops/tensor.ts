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

import {Tensor} from '../tensor';
import {inferShape} from '../tensor_util_env';
import {TensorLike} from '../types';
import {DataType, Rank, ShapeMap, WebGLData, WebGPUData} from '../types';

import {makeTensor} from './tensor_ops_util';

/**
 * Creates a `tf.Tensor` with the provided values, shape and dtype.
 *
 * ```js
 * // Pass an array of values to create a vector.
 * tf.tensor([1, 2, 3, 4]).print();
 * ```
 *
 * ```js
 * // Pass a nested array of values to make a matrix or a higher
 * // dimensional tensor.
 * tf.tensor([[1, 2], [3, 4]]).print();
 * ```
 *
 * ```js
 * // Pass a flat array and specify a shape yourself.
 * tf.tensor([1, 2, 3, 4], [2, 2]).print();
 * ```
 *
 * ```js
 * // Pass a `WebGLData` object and specify a shape yourself.
 *
 * // This makes it possible for TF.js applications to avoid GPU / CPU sync.
 * // For example, if your application includes a preprocessing step on the GPU,
 * // you could upload the GPU output directly to TF.js, rather than first
 * // downloading the values.
 *
 * // Example for WebGL2:
 * if (tf.findBackend('custom-webgl') == null) {
 *   const customCanvas = document.createElement('canvas');
 *   const customBackend = new tf.MathBackendWebGL(customCanvas);
 *   tf.registerBackend('custom-webgl', () => customBackend);
 * }
 * const savedBackend = tf.getBackend();
 * await tf.setBackend('custom-webgl');
 * const gl = tf.backend().gpgpu.gl;
 * const texture = gl.createTexture();
 * const tex2d = gl.TEXTURE_2D;
 * const width = 2;
 * const height = 2;
 *
 * gl.bindTexture(tex2d, texture);
 * gl.texParameteri(tex2d, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
 * gl.texParameteri(tex2d, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
 * gl.texParameteri(tex2d, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
 * gl.texParameteri(tex2d, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
 * gl.texImage2D(
 *   tex2d, 0, gl.RGBA32F, // internalFormat
 *   width, height, 0,
 *   gl.RGBA, // textureFormat
 *   gl.FLOAT, // textureType
 *   new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
 * );
 *
 * // Currently, the `texture` has 4 pixels:
 * // Pixel0 is {R:0, G:1, B:2, A:3}
 * // Pixel1 is {R:4, G:5, B:6, A:7}
 * // Pixel2 is {R:8, G:9, B:10, A:11}
 * // Pixel3 is {R:12, G:13, B:14, A:15}
 *
 * const logicalShape = [height * width * 2];
 * const a = tf.tensor({texture, height, width, channels: 'BR'}, logicalShape);
 * a.print();
 * // Tensor value will be [2, 0, 6, 4, 10, 8, 14, 12], since [2, 0] is the
 * // values of 'B' and 'R' channels of Pixel0, [6, 4] is the values of 'B' and
 * 'R'
 * // channels of Pixel1...
 *
 * // For postprocessing on the GPU, it's possible to retrieve the texture
 * // backing any tensor by calling the tensor's `dataToGPU` method like
 * // so:
 *
 * const tex = a.dataToGPU();
 * await tf.setBackend(savedBackend);
 * ```
 *
 * ```js
 * // Pass a `WebGPUData` object and specify a shape yourself.
 *
 * // This makes it possible for TF.js applications to avoid GPU / CPU sync.
 * // For example, if your application includes a preprocessing step on the GPU,
 * // you could upload the GPU output directly to TF.js, rather than first
 * // downloading the values. Unlike WebGL, this optionally supports zero copy
 * // by WebGPUData.zeroCopy. When zeroCopy is false or undefined(default), this
 * // passing GPUBuffer can be destroyed after tensor is created. When zeroCopy
 * // is true, this GPUBuffer is bound directly by the tensor, so do not destroy
 * // this GPUBuffer until all access is done.
 *
 * // Example for WebGPU:
 * function createGPUBufferFromData(device, data, dtype) {
 *   const bytesPerElement = 4;
 *   const sizeInBytes = data.length * bytesPerElement;
 *
 *   const gpuWriteBuffer = device.createBuffer({
 *     mappedAtCreation: true,
 *     size: sizeInBytes,
 *     usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC
 *   });
 *   const arrayBuffer = gpuWriteBuffer.getMappedRange();
 *   if (dtype === 'float32') {
 *     new Float32Array(arrayBuffer).set(data);
 *   } else if (dtype === 'int32') {
 *     new Int32Array(arrayBuffer).set(data);
 *   } else {
 *     throw new Error(
 *         `Creating tensor from GPUBuffer only supports` +
 *         `'float32'|'int32' dtype, while the dtype is ${dtype}.`);
 *   }
 *   gpuWriteBuffer.unmap();
 *
 *   const gpuReadBuffer = device.createBuffer({
 *     mappedAtCreation: false,
 *     size: sizeInBytes,
 *     usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE |
 *         GPUBufferUsage.COPY_SRC
 *   });
 *
 *   const copyEncoder = device.createCommandEncoder();
 *   copyEncoder.copyBufferToBuffer(
 *       gpuWriteBuffer, 0, gpuReadBuffer, 0, sizeInBytes);
 *   const copyCommands = copyEncoder.finish();
 *   device.queue.submit([copyCommands]);
 *   gpuWriteBuffer.destroy();
 *   return gpuReadBuffer;
 * }
 *
 * const savedBackend = tf.getBackend();
 * await tf.setBackend('webgpu').catch(
 *     () => {throw new Error(
 *         'Failed to use WebGPU backend. Please use Chrome Canary to run.')});
 * const dtype = 'float32';
 * const device = tf.backend().device;
 * const aData = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
 * const bData = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4];
 * const expected = [2, 4, 6, 8, 6, 8, 10, 12, 10, 12, 14, 16, 14, 16, 18, 20];
 * const aBuffer = createGPUBufferFromData(device, aData, dtype);
 * const shape = [aData.length];
 * // To use zeroCopy, use {buffer: aBuffer, zeroCopy: true} instead and destroy
 * // aBuffer untill all access is done.
 * const a = tf.tensor({buffer: aBuffer}, shape, dtype);
 * const b = tf.tensor(bData, shape, dtype);
 * const result = tf.add(a, b);
 * result.print();
 * a.dispose();
 * b.dispose();
 * result.dispose();
 * aBuffer.destroy();
 * await tf.setBackend(savedBackend);
 * ```
 * @param values The values of the tensor. Can be nested array of numbers,
 * or a flat array, or a `TypedArray`(At the moment it supports Uint8Array,
 * Uint8ClampedArray, Int32Array, Float32Array) data types, or a `WebGLData`
 * object, or a `WebGPUData` object. If the values are strings, they will be
 * encoded as utf-8 and kept as `Uint8Array[]`. If the values is a `WebGLData`
 * object, the dtype could only be 'float32' or 'int32' and the object has to
 * have: 1. texture, a `WebGLTexture`, the texture must share the same
 * `WebGLRenderingContext` with TFJS's WebGL backend (you could create a custom
 * WebGL backend from your texture's canvas) and the internal texture format
 * for the input texture must be floating point or normalized integer; 2.
 * height, the height of the texture; 3. width, the width of the texture; 4.
 * channels, a non-empty subset of 'RGBA', indicating the values of which
 * channels will be passed to the tensor, such as 'R' or 'BR' (The order of the 
 * channels affect the order of tensor values. ). (If the values passed from 
 * texture is less than the tensor size, zeros will be padded at the rear.). If 
 * the values is a `WebGPUData` object, the dtype could only be 'float32' or
 * 'int32 and the object has to have: buffer, a `GPUBuffer`. The buffer must:
 * 1. share the same `GPUDevice` with TFJS's WebGPU backend; 2. buffer.usage
 * should at least support GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC; 3.
 * buffer.size should not be smaller than the byte size of tensor shape.
 * WebGPUData optionally supports zero copy by flag zeroCopy. When zeroCopy is
 * false or undefined(default),this passing GPUBuffer can be destroyed after
 * tensor is created. When zeroCopy is true, this GPUBuffer is bound directly
 * by the tensor, so do not destroy this GPUBuffer until all access is done.
 * @param shape The shape of the tensor. Optional. If not provided,
 *   it is inferred from `values`.
 * @param dtype The data type.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
export function tensor<R extends Rank>(
    values: TensorLike|WebGLData|WebGPUData, shape?: ShapeMap[R],
    dtype?: DataType): Tensor<R> {
  const inferredShape = inferShape(values, dtype);
  return makeTensor(values, shape, inferredShape, dtype) as Tensor<R>;
}