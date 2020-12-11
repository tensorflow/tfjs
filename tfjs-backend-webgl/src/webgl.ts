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
import {getTextureConfig} from './tex_util';
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

export function createTensorFromTexture(
    texture: WebGLTexture, shape: number[],
    texShapeRC: [number, number]): Tensor {
  const backend = engine().backend as MathBackendWebGL;
  const dataId = backend.writeTexture(texture, shape, 'float32', texShapeRC);
  return engine().makeTensorFromDataId(dataId, shape, 'float32', backend);
}

type TensorFromTextureConfig = {
  texture: WebGLTexture,
  shape: number[],
  dtype: DataType,
  texShapeRC: [number, number],
  internalFormat: number,
  textureFormat: number,
  textureType: number
};

export function createTensorFromTexture2({
  texture,
  shape,
  dtype,
  texShapeRC,
  internalFormat,
  textureFormat,
  textureType
}: TensorFromTextureConfig): Tensor {
  const backend = engine().backend as MathBackendWebGL;
  const gl = backend.gpgpu.gl;
  const texConfig = backend.gpgpu.textureConfig;
  let params: gpgpu_util.TextureCreationParams;

  if (env().getBool('WEBGL_RENDER_FLOAT32_ENABLED') === true) {
    params = gpgpu_util.getTextureParamsForFloat32MatrixTexture(gl, texConfig);
  } else {
    params = gpgpu_util.getTextureParamsForFloat16MatrixTexture(gl, texConfig);
  }

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
