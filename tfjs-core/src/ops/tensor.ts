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

import {Tensor, WebGLData} from '../tensor';
import {inferShape} from '../tensor_util_env';
import {TensorLike} from '../types';
import {DataType, Rank, ShapeMap} from '../types';

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
 * await tf.setBackend('webgl');
 * const gl = tf.backend().gpgpu.gl;
 * const texture = gl.createTexture();
 * const tex2d = gl.TEXTURE_2D;
 * const width = 3;
 * const height = 4;
 *
 * gl.bindTexture(tex2d, texture);
 * gl.texParameteri(tex2d, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
 * gl.texParameteri(tex2d, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
 * gl.texParameteri(tex2d, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
 * gl.texParameteri(tex2d, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
 * gl.texImage2D(
 *   tex2d, 0, gl.R32F, // internalFormat
 *   width, height, 0,
 *   gl.RED, // textureFormat
 *   gl.FLOAT, // textureType
 *   new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) // data
 * );
 *
 * const logicalShape = [height, width];
 * const a = tf.tensor({texture, height, width, channels: 'R'}, logicalShape);
 *
 * // For postprocessing on the GPU, it's possible to retrieve the texture
 * // backing any tensor by calling the WebGL backend's `getTexture` method like
 * // so:
 *
 * const tex = tf.backend().getTexture(a.dataId);
 * ```
 * @param values The values of the tensor. Can be nested array of numbers,
 *     or a flat array, or a `TypedArray`, or a `WebGLData` object. If the
 * values are strings, they will be encoded as utf-8 and kept as `Uint8Array[]`.
 * If the values is a `WebGLData` object, the object has to have: 1. texture, a
 * `WebGLTexture`; 2. height, the height of the texture; 3. width, the width of
 * the texture; 4. channels, a non-empty subsequence of 'RGBA', indicating the
 * values of which channels will be passed to the tensor (such as 'R' or 'GA').
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
