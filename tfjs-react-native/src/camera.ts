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

// import {downloadTextureData, getPassthroughProgram, runResizeProgram,
// uploadTextureData} from './camera_webgl_util';
import {downloadTextureData, drawTexture, uploadTextureData} from './camera_utils/camera_webgl_util';

interface Dimensions {
  x?: number;
  y?: number;
  width: number;
  height: number;
  depth: number;
}

interface Size {
  width: number;
  height: number;
  depth: number;
}

// Draws image tensorData to a textre
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

export function fromTexture(
    gl: WebGL2RenderingContext, texture: WebGLTexture,
    dims: Dimensions): tf.Tensor {
  // const sourceDims = {
  //   x: 0,
  //   y: 0,
  //   width: gl.drawingBufferWidth,
  //   height: gl.drawingBufferHeight,
  //   depth: 4 as 4,
  // };

  const targetDims = {
    x: 0,
    y: 0,
    width: dims.width,
    height: dims.height,
    depth: dims.depth,
  };

  //@ts-ignore
  // const resizedTexture = runResizeProgram(gl, texture, sourceDims,
  // targetDims); console.log('resizedTexture', resizedTexture);
  const textureData = downloadTextureData(gl, texture, targetDims);

  return tf.tensor(
      textureData, [targetDims.width, targetDims.height, targetDims.depth],
      'int32');
}

/**
 * Render a texture to the GLView. This will use the default framebuffer
 * and present the contents of the texture on the screen.
 *
 * @param gl
 * @param texture
 */
export function renderToGLView(
    gl: WebGL2RenderingContext, texture: WebGLTexture, dims: Dimensions) {
  drawTexture(gl, texture, dims);
}
