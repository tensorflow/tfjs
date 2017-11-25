/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

import {GPGPUContext} from './gpgpu_context';
import * as gpgpu_util from './gpgpu_util';
import * as render_ndarray_gpu_util from './render_ndarray_gpu_util';

function uploadRenderRGBDownload(
    source: Float32Array, sourceShape: [number, number, number]) {
  const canvas = document.createElement('canvas');
  canvas.width = sourceShape[0];
  canvas.height = sourceShape[1];

  const gpgpu = new GPGPUContext();
  gpgpu.enableAutomaticDebugValidation(true);

  const program =
      render_ndarray_gpu_util.getRenderRGBShader(gpgpu, sourceShape[1]);

  const sourceTexShapeRC: [number, number] =
      [sourceShape[0], sourceShape[1] * sourceShape[2]];

  const sourceTex =
      gpgpu.createMatrixTexture(sourceTexShapeRC[0], sourceTexShapeRC[1]);
  gpgpu.uploadMatrixToTexture(
      sourceTex, sourceTexShapeRC[0], sourceTexShapeRC[1], source);

  const resultTex = gpgpu_util.createColorMatrixTexture(
      gpgpu.gl, sourceShape[0], sourceShape[1]);
  gpgpu.setOutputMatrixTexture(resultTex, sourceShape[0], sourceShape[1]);
  render_ndarray_gpu_util.renderToFramebuffer(gpgpu, program, sourceTex);

  const result = new Float32Array(sourceShape[0] * sourceShape[1] * 4);
  gpgpu.gl.readPixels(
      0, 0, sourceShape[1], sourceShape[0], gpgpu.gl.RGBA, gpgpu.gl.FLOAT,
      result);
  return result;
}

describe('render_gpu', () => {
  it('Packs a 1x1x3 vector to a 1x1 color texture', () => {
    const source = new Float32Array([1, 2, 3]);
    const result = uploadRenderRGBDownload(source, [1, 1, 3]);
    expect(result).toEqual(new Float32Array([1, 2, 3, 1]));
  });

  it('Packs a 2x2x3 vector to a 2x2 color texture, mirrored vertically', () => {
    const source = new Float32Array([1, 2, 3, 30, 20, 10, 2, 3, 4, 40, 30, 20]);
    const result = uploadRenderRGBDownload(source, [2, 2, 3]);
    // The resulting rendered image is flipped vertically.
    expect(result).toEqual(new Float32Array(
        [2, 3, 4, 1, 40, 30, 20, 1, 1, 2, 3, 1, 30, 20, 10, 1]));
  });
});
