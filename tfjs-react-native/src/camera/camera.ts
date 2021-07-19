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

import {downloadTextureData, drawTexture, runResizeProgram, uploadTextureData} from './camera_webgl_util';
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
  interpolation?: 'nearest_neighbor'|'bilinear';
}

const glCapabilities = {
  canDownloadFromRGBTexture: new WeakMap<WebGL2RenderingContext, boolean>(),
  // Has detectGLCapabilities been run on a particular GL context;
  glCapabilitiesTested: new WeakMap<WebGL2RenderingContext, boolean>()
};

/**
 * Utility function that tests the GL context for capabilities to enable
 * optimizations.
 *
 * For best performance this should be be called once before using the other
 * camera related functions.
 *
 * @doc {heading: 'Media', subheading: 'Camera'}
 */
export async function detectGLCapabilities(gl: WebGL2RenderingContext) {
  if (glCapabilities.glCapabilitiesTested.get(gl)) {
    return;
  }
  // Test whether we can successfully download from an RGB texture.
  // Notably this isn't supported on iOS, but we use this test rather than a
  // platform check to be more robust on android devices we may not have
  // directly tested.

  // Set this to true temporarily so that fromTexture does not
  // use its workaround.
  glCapabilities.canDownloadFromRGBTexture.set(gl, true);

  try {
    const height = 2;
    const width = 4;  // This must be a multiple of 4.
    const data = new Uint8Array(height * width * 4);
    for (let i = 0; i < data.length; i++) {
      data[i] = i;
    }
    const sourceDims = {height, width, depth: 4};
    const tex = uploadTextureData(data, gl, sourceDims);

    const targetDims = {height, width, depth: 3};
    const downloaded = fromTexture(gl, tex, sourceDims, targetDims);
    const downloadedData = await downloaded.data();
    tf.dispose(downloaded);

    const matches = tf.util.arraysEqual(downloadedData, [
      0,  1,  2,  4,  5,  6,  8,  9,  10, 12, 13, 14,
      16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30
    ]);

    if (matches) {
      glCapabilities.canDownloadFromRGBTexture.set(gl, true);
    } else {
      glCapabilities.canDownloadFromRGBTexture.set(gl, false);
    }
  } catch (e) {
    glCapabilities.canDownloadFromRGBTexture.set(gl, false);
  } finally {
    glCapabilities.glCapabilitiesTested.set(gl, true);
  }
}

/**
 * Transfers tensor data to an RGB(A) texture.
 *
 * @param gl the WebGL context that owns the texture.
 * @param imageTensor the tensor to upload
 * @param texture optional the target texture. If none is passed in a new
 *     texture will be created.
 *
 * @doc {heading: 'Media', subheading: 'Camera'}
 */
export async function toTexture(
    gl: WebGL2RenderingContext, imageTensor: tf.Tensor3D,
    texture?: WebGLTexture): Promise<WebGLTexture> {
  tf.util.assert(
      imageTensor.dtype === 'int32', () => 'imageTensor must be of type int32');

  tf.util.assert(
      imageTensor.rank === 3, () => 'imageTensor must be a Tensor3D');

  const imageData = Uint8Array.from(await imageTensor.data());
  const dims = {
    height: imageTensor.shape[0],
    width: imageTensor.shape[1],
    depth: imageTensor.shape[2],
  };
  return uploadTextureData(imageData, gl, dims, texture);
}

/**
 * Creates a tensor3D from a texture.
 *
 * Allows for resizing the image and dropping the alpha channel from the
 * resulting tensor.
 *
 * Note that if you the output depth is 3 then the output width should be a
 * multiple of 4.
 *
 * @param gl the WebGL context that owns the input texture
 * @param texture the texture to convert into a tensor
 * @param sourceDims source dimensions of input texture (width, height, depth)
 * @param targetShape desired shape of output tensor
 *
 * @doc {heading: 'Media', subheading: 'Camera'}
 */
export function fromTexture(
    gl: WebGL2RenderingContext, texture: WebGLTexture, sourceDims: Dimensions,
    targetShape: Dimensions, options: FromTextureOptions = {}): tf.Tensor3D {
  tf.util.assert(
      targetShape.depth === 3 || targetShape.depth === 4,
      () => 'fromTexture Error: target depth must be 3 or 4');

  if (targetShape.depth === 3 && targetShape.width % 4 !== 0) {
    // We throw an error here rather than use the CPU workaround as the user is
    // likely trying to get the maximum performance.
    if (glCapabilities.canDownloadFromRGBTexture.get(gl)) {
      // See
      // https://www.khronos.org/opengl/wiki/Common_Mistakes#Texture_upload_and_pixel_reads
      // for more details. At the moment gl.pixelStorei(gl.PACK_ALIGNMENT, 1);
      // is not supported on expo. "EXGL: gl.pixelStorei() doesn't support this
      // parameter yet!"
      throw new Error(
          'When using targetShape.depth=3, targetShape.width must be' +
          ' a multiple of 4. Alternatively do not call detectGLCapabilities()');
    }
  }

  const originalTargetDepth = targetShape.depth;
  const targetDepth = glCapabilities.canDownloadFromRGBTexture.get(gl) ?
      originalTargetDepth :
      4;

  sourceDims = {
    height: Math.floor(sourceDims.height),
    width: Math.floor(sourceDims.width),
    depth: sourceDims.depth,
  };

  targetShape = {
    height: Math.floor(targetShape.height),
    width: Math.floor(targetShape.width),
    depth: targetDepth
  };

  const alignCorners =
      options.alignCorners != null ? options.alignCorners : false;
  const interpolation =
      options.interpolation != null ? options.interpolation : 'bilinear';

  tf.util.assert(
      interpolation === 'bilinear' || interpolation === 'nearest_neighbor',
      () => 'fromTexture Error: interpolation must be one of' +
          ' "bilinear" or "nearest_neighbor"');

  const resizedTexture = runResizeProgram(
      gl, texture, sourceDims, targetShape, alignCorners, interpolation);
  const downloadedTextureData =
      downloadTextureData(gl, resizedTexture, targetShape);

  let finalTexData;
  if (originalTargetDepth !== targetDepth && originalTargetDepth === 3) {
    // We are on a device that does not support downloading from an RGB texture.
    // Remove the alpha channel values on the CPU.
    const area = targetShape.height * targetShape.width;
    finalTexData = new Uint8Array(area * originalTargetDepth);

    for (let i = 0; i < area; i++) {
      const flatIndexRGB = i * 3;
      const flatIndexRGBA = i * 4;
      finalTexData[flatIndexRGB] = downloadedTextureData[flatIndexRGBA];
      finalTexData[flatIndexRGB + 1] = downloadedTextureData[flatIndexRGBA + 1];
      finalTexData[flatIndexRGB + 2] = downloadedTextureData[flatIndexRGBA + 2];
    }
  } else {
    finalTexData = downloadedTextureData;
  }

  return tf.tensor3d(
      finalTexData,
      [targetShape.height, targetShape.width, originalTargetDepth], 'int32');
}

/**
 * Render a texture to the GLView. This will use the default framebuffer
 * and present the contents of the texture on the screen.
 *
 * @param gl
 * @param texture
 * @param dims Dimensions of tensor
 *
 * @doc {heading: 'Media', subheading: 'Camera'}
 */
export function renderToGLView(
    gl: WebGL2RenderingContext, texture: WebGLTexture, size: Size,
    flipHorizontal = true) {
  size = {
    width: Math.floor(size.width),
    height: Math.floor(size.height),
  };
  drawTexture(gl, texture, size, flipHorizontal);
}
