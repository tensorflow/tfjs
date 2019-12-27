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

import {downloadTextureData, drawTexture, runResizeProgram, uploadTextureData} from './camera_utils/camera_webgl_util';

interface Dimensions {
  width: number;
  height: number;
  depth: number;
}

interface Size {
  width: number;
  height: number;
}

interface FromTextureOptions {
  alignCorners?: boolean;
}

/**
 * Transfers tensor data to an RGB(A) texture.
 *
 * @param gl the WebGL context that owns the texture.
 * @param imageTensor the tensor to upload
 * @param texture optional the target texture. If none is passed in a new
 *     texture will be created.
 */
export async function toTexture(
    gl: WebGL2RenderingContext, imageTensor: tf.Tensor3D,
    texture?: WebGLTexture): Promise<WebGLTexture> {
  const imageData = Uint8Array.from(await imageTensor.data());
  const dims = {
    x: 0,
    y: 0,
    width: imageTensor.shape[0],
    height: imageTensor.shape[1],
    depth: imageTensor.shape[2],
  };
  return uploadTextureData(imageData, gl, dims, texture);
}

/**
 * Creates a tensor3D from a texture.
 *
 * Allows for resizing the image and dropping the alpha channel from the data.
 *
 * Note that the tensor's data will live in the internal tfjs webgl context.
 *
 * @param gl the WebGL context that owns the input texture
 * @param texture the texture to convert into a tensor
 * @param sourceDims source dimensions of input texture (width, height, depth)
 * @param targetShape desired shape of output tensor
 */
export function fromTexture(
    gl: WebGL2RenderingContext, texture: WebGLTexture, sourceDims: Dimensions,
    targetShape: Dimensions, options: FromTextureOptions = {}): tf.Tensor3D {
  tf.util.assert(
      targetShape.depth === 3 || targetShape.depth === 4,
      () => 'fromTexture Error: target depth must be 3 or 4');

  const _sourceDims = {
    width: Math.floor(sourceDims.width),
    height: Math.floor(sourceDims.height),
    depth: sourceDims.depth,
  };

  const _targetShape = {
    width: Math.floor(targetShape.width),
    height: Math.floor(targetShape.height),
    depth: targetShape.depth
  };

  const alignCorners =
      options.alignCorners != null ? options.alignCorners : false;
  console.log('fromTexture:alignCorners', alignCorners, options);

  const resizedTexture =
      runResizeProgram(gl, texture, _sourceDims, targetShape, alignCorners);
  // console.log('resizedTexture', resizedTexture);
  const textureData = downloadTextureData(gl, resizedTexture, _targetShape);

  return tf.tensor3d(
      textureData,
      [_targetShape.width, _targetShape.height, _targetShape.depth], 'int32');
}

/**
 * Render a texture to the GLView. This will use the default framebuffer
 * and present the contents of the texture on the screen.
 *
 * @param gl
 * @param texture
 * @param dims Dimensions of tensor
 */
export function renderToGLView(
    gl: WebGL2RenderingContext, texture: WebGLTexture, size: Size) {
  const _size = {
    width: Math.floor(size.width),
    height: Math.floor(size.width),
  };
  drawTexture(gl, texture, _size);
}
