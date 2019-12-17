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

import {downloadTextureData, getPassthroughProgram, runResizeProgram, uploadTextureData} from './camera_webgl_util';

interface Dimensions {
  x?: number;
  y?: number;
  width: number;
  height: number;
  depth: 3|4;
}

// Draws image tensorData to a textre
export async function toTexture(
    imageTensor: tf.Tensor, gl: WebGL2RenderingContext, dims: Dimensions,
    texture?: WebGLTexture): Promise<WebGLTexture> {
  const imageData = Uint8Array.from(await imageTensor.data());
  return uploadTextureData(imageData, gl, dims, texture);
}

export function fromTexture(
    gl: WebGL2RenderingContext, texture: WebGLTexture,
    dims: Dimensions): tf.Tensor {
  const sourceDims = {
    x: 0,
    y: 0,
    width: gl.drawingBufferWidth,
    height: gl.drawingBufferHeight,
    depth: 4 as 4,
  };

  const targetDims = {
    x: 0,
    y: 0,
    width: dims.width || gl.drawingBufferWidth,
    height: dims.height || gl.drawingBufferHeight,
    depth: dims.depth,
  };

  const resizedTexture = runResizeProgram(gl, texture, sourceDims, targetDims);
  const textureData = downloadTextureData(gl, resizedTexture, targetDims);

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
    gl: WebGL2RenderingContext, texture: WebGLTexture) {
  const program = getPassthroughProgram(gl);
  gl.useProgram(program);

  // Set up geometry
  const positionAttrib = gl.getAttribLocation(program, 'position');
  gl.enableVertexAttribArray(positionAttrib);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  const verts = new Float32Array([-2, 0, 0, -2, 2, 2]);
  gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
  gl.vertexAttribPointer(positionAttrib, 2, gl.FLOAT, false, 0, 0);

  // Set texture sampler uniform
  const TEXTURE_UNIT = 0;
  gl.uniform1i(gl.getUniformLocation(program, 'displayTexture'), TEXTURE_UNIT);
  gl.activeTexture(gl.TEXTURE0 + TEXTURE_UNIT);
  gl.bindTexture(gl.TEXTURE_2D, texture);

  // Draw to screen
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.clearColor(1, 1, 1, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.drawArrays(gl.TRIANGLES, 0, verts.length / 2);
}
