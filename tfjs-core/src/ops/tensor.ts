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
import {DataType, Rank, ShapeMap, WebGLData} from '../types';

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
 * const customCanvas = document.createElement('canvas');
 * const customBackend = new tf.MathBackendWebGL(customCanvas);
 * tf.registerBackend('custom-webgl', () => customBackend);
 * await tf.setBackend('custom-webgl');
 * const gl = customBackend.gpgpu.gl;
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
 * ```
 * @param values The values of the tensor. Can be nested array of numbers,
 *     or a flat array, or a `TypedArray`, or a `WebGLData` object. If the
 * values are strings, they will be encoded as utf-8 and kept as `Uint8Array[]`.
 * If the values is a `WebGLData` object, the dtype could only be 'float32' or
 * 'int32' and the object has to have: 1. texture, a `WebGLTexture`, the texture
 * must share the same `WebGLRenderingContext` with TFJS's WebGL backend (you
 * could create a custom WebGL backend from your texture's canvas) and the
 * internal texture format for the input texture must be floating point or
 * normalized integer; 2. height, the height of the texture; 3. width, the width
 * of the texture; 4. channels, a non-empty subset of 'RGBA', indicating the
 * values of which channels will be passed to the tensor, such as 'R' or 'BR'
 * (The order of the channels affect the order of tensor values. ). (If the
 * values passed from texture is less than the tensor size, zeros will be padded
 * at the rear.)
 * @param shape The shape of the tensor. Optional. If not provided,
 *   it is inferred from `values`.
 * @param dtype The data type.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
export function tensor<R extends Rank>(
    values: TensorLike|WebGLData, shape?: ShapeMap[R],
    dtype?: DataType): Tensor<R> {
  const inferredShape = inferShape(values, dtype);
  return makeTensor(values, shape, inferredShape, dtype) as Tensor<R>;
}
