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

import {DataType, engine, env, Tensor, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from './backend_webgl';
import * as gpgpu_util from './gpgpu_util';
import {WebGLTextureFormat, WebGLTextureInternalFormat, WebGLTextureType} from './tex_util';
import * as webgl_util from './webgl_util';

export {MathBackendWebGL, WebGLMemoryInfo, WebGLTimingInfo} from './backend_webgl';
export {setWebGLContext} from './canvas_util';
export {GPGPUContext} from './gpgpu_context';
export {GPGPUProgram} from './gpgpu_math';
// WebGL specific utils.
export {gpgpu_util, webgl_util};

/**
 * Enforce use of half precision textures if available on the platform.
 *
 * @doc {heading: 'Environment', namespace: 'webgl'}
 */
export function forceHalfFloat(): void {
  env().set('WEBGL_FORCE_F16_TEXTURES', true);
}

type TensorFromTextureConfig = {
  texture: WebGLTexture,
  shape: number[],
  dtype: DataType,
  texShapeRC: [number, number],
  internalFormat: WebGLTextureInternalFormat,
  textureFormat: WebGLTextureFormat,
  textureType: WebGLTextureType
};

/**
 * Create a tensor out of an existing WebGL texture.
 *
 * ```js
 * // Example for WebGL2:
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
 * const physicalShape = [height, width];
 * const a = tf.webgl.createTensorFromTexture(texture, logicalShape,
 *   physicalShape);
 *
 * ```
 *
 * @param obj An object with the following properties:
 *  @param texture The WebGL texture to create a tensor from. The texture must
 * be unpacked - each texel should only store a single value.
 *  @param shape The logical shape of the texture.
 *  @param dtype The dtype of the tensor to be created.
 *  @param texShapeRC The physical dimensions of the texture expressed as [rows,
 * columns].
 *  @param internalFormat The internalFormat of the texture provided to
 * gl.texImage2D during texture creation.
 *  @param textureFormat The textureFormat of the texture provided to
 * gl.texImage2D during texture creation.
 *  @param textureType The textureType of the texture provided to gl.texImage2D
 * during texture creation.
 * @doc {heading: 'Environment', namespace: 'webgl'}
 */
export function createTensorFromTexture({
  texture,
  shape,
  dtype,
  texShapeRC,
  internalFormat,
  textureFormat,
  textureType
}: TensorFromTextureConfig): Tensor {
  // OpenGL / WebGL do not make it possible to query textures for their
  // properties (physical dimensions, internalFormat, etc.), therefore we ask
  // the user to provide this information in order to validate their texture.

  // References that this information cannot be queried:
  // https://stackoverflow.com/questions/30140178/opengl-es-2-0-get-texture-size-and-other-info
  // https://stackoverflow.com/questions/26315021/is-there-a-way-to-retrieve-the-dimensions-of-a-texture-after-binding-with-gl-bin
  // https://stackoverflow.com/questions/46387922/how-to-check-a-texture-is-2d-texture-or-cube-texture-in-webgl

  const backend = engine().backend as MathBackendWebGL;
  const gl = backend.gpgpu.gl;
  const texConfig = backend.gpgpu.textureConfig;
  let params: gpgpu_util.TextureCreationParams;

  if (env().getBool('WEBGL_RENDER_FLOAT32_ENABLED') === true) {
    params = gpgpu_util.getTextureParamsForFloat32MatrixTexture(gl, texConfig);
  } else {
    params = gpgpu_util.getTextureParamsForFloat16MatrixTexture(gl, texConfig);
  }

  // Ensure that the properties of the texture match the expectations of the
  // WebGL backend.
  util.assert(
      internalFormat === params.internalFormat,
      () => `The internalFormat must be ${params.internalFormat}.`);
  util.assert(
      textureFormat === params.textureFormat,
      () => `The textureFormat must be ${params.textureFormat}.`);
  util.assert(
      textureType === params.textureType,
      () => `The textureType must be ${params.textureType}.`);

  const dataId = backend.writeTexture(texture, shape, dtype, texShapeRC);
  return engine().makeTensorFromDataId(dataId, shape, dtype, backend);
}
