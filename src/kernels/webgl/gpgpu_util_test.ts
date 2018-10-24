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

import * as tf from '../../index';
import {describeWithFlags} from '../../jasmine_util';
import {expectArraysClose, WEBGL_ENVS} from '../../test_util';

import {GPGPUContext} from './gpgpu_context';
import * as gpgpu_util from './gpgpu_util';

const DOWNLOAD_FLOAT_ENVS = {
  'WEBGL_DOWNLOAD_FLOAT_ENABLED': true
};

describeWithFlags('gpgpu_util createWebGLContext', WEBGL_ENVS, () => {
  let gpgpu: GPGPUContext;

  beforeEach(() => {
    gpgpu = new GPGPUContext();
  });

  afterEach(() => {
    gpgpu.dispose();
  });

  it('disables DEPTH_TEST and STENCIL_TEST', () => {
    expect(gpgpu.gl.getParameter(gpgpu.gl.DEPTH_TEST)).toEqual(false);
    expect(gpgpu.gl.getParameter(gpgpu.gl.STENCIL_TEST)).toEqual(false);
  });

  it('disables BLEND', () => {
    expect(gpgpu.gl.getParameter(gpgpu.gl.BLEND)).toEqual(false);
  });

  it('disables DITHER, POLYGON_OFFSET_FILL', () => {
    expect(gpgpu.gl.getParameter(gpgpu.gl.DITHER)).toEqual(false);
    expect(gpgpu.gl.getParameter(gpgpu.gl.POLYGON_OFFSET_FILL)).toEqual(false);
  });

  it('enables CULL_FACE with BACK', () => {
    expect(gpgpu.gl.getParameter(gpgpu.gl.CULL_FACE)).toEqual(true);
    expect(gpgpu.gl.getParameter(gpgpu.gl.CULL_FACE_MODE))
        .toEqual(gpgpu.gl.BACK);
  });

  it('enables SCISSOR_TEST', () => {
    expect(gpgpu.gl.getParameter(gpgpu.gl.SCISSOR_TEST)).toEqual(true);
  });
});

describeWithFlags('gpgpu_util createFloat32MatrixTexture', WEBGL_ENVS, () => {
  it('sets the TEXTURE_WRAP S+T parameters to CLAMP_TO_EDGE', () => {
    const gpgpu = new GPGPUContext();
    const textureConfig = gpgpu_util.getTextureConfig(gpgpu.gl);
    const tex =
        gpgpu_util.createFloat32MatrixTexture(gpgpu.gl, 32, 32, textureConfig);
    gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, tex);
    expect(
        gpgpu.gl.getTexParameter(gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_WRAP_S))
        .toEqual(gpgpu.gl.CLAMP_TO_EDGE);
    expect(
        gpgpu.gl.getTexParameter(gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_WRAP_T))
        .toEqual(gpgpu.gl.CLAMP_TO_EDGE);
    gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, null);
    gpgpu.deleteMatrixTexture(tex);
    gpgpu.dispose();
  });

  it('sets the TEXTURE_[MIN|MAG]_FILTER parameters to NEAREST', () => {
    const gpgpu = new GPGPUContext();
    const textureConfig = gpgpu_util.getTextureConfig(gpgpu.gl);
    const tex =
        gpgpu_util.createFloat32MatrixTexture(gpgpu.gl, 32, 32, textureConfig);
    gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, tex);
    expect(gpgpu.gl.getTexParameter(
               gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_MIN_FILTER))
        .toEqual(gpgpu.gl.NEAREST);
    expect(gpgpu.gl.getTexParameter(
               gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_MAG_FILTER))
        .toEqual(gpgpu.gl.NEAREST);
    gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, null);
    gpgpu.deleteMatrixTexture(tex);
    gpgpu.dispose();
  });
});

describeWithFlags('gpgpu_util createPackedMatrixTexture', WEBGL_ENVS, () => {
  it('sets the TEXTURE_WRAP S+T parameters to CLAMP_TO_EDGE', () => {
    const gpgpu = new GPGPUContext();
    const textureConfig = gpgpu_util.getTextureConfig(gpgpu.gl);
    const tex =
        gpgpu_util.createPackedMatrixTexture(gpgpu.gl, 32, 32, textureConfig);
    gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, tex);
    expect(
        gpgpu.gl.getTexParameter(gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_WRAP_S))
        .toEqual(gpgpu.gl.CLAMP_TO_EDGE);
    expect(
        gpgpu.gl.getTexParameter(gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_WRAP_T))
        .toEqual(gpgpu.gl.CLAMP_TO_EDGE);
    gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, null);
    gpgpu.deleteMatrixTexture(tex);
    gpgpu.dispose();
  });

  it('sets the TEXTURE_[MIN|MAG]_FILTER parameters to NEAREST', () => {
    const gpgpu = new GPGPUContext();
    const textureConfig = gpgpu_util.getTextureConfig(gpgpu.gl);
    const tex =
        gpgpu_util.createPackedMatrixTexture(gpgpu.gl, 32, 32, textureConfig);
    gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, tex);
    expect(gpgpu.gl.getTexParameter(
               gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_MIN_FILTER))
        .toEqual(gpgpu.gl.NEAREST);
    expect(gpgpu.gl.getTexParameter(
               gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_MAG_FILTER))
        .toEqual(gpgpu.gl.NEAREST);
    gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, null);
    gpgpu.deleteMatrixTexture(tex);
    gpgpu.dispose();
  });
});

describeWithFlags(
    'gpgpu_util downloadMatrixFromPackedOutputTexture', DOWNLOAD_FLOAT_ENVS,
    () => {
      it('should work when texture shape != logical shape', () => {
        const gpgpu = new GPGPUContext();
        const textureConfig = gpgpu_util.getTextureConfig(gpgpu.gl);

        const tex =
            gpgpu_util.createPackedMatrixTexture(gpgpu.gl, 4, 6, textureConfig);

        const mat =
            tf.tensor2d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [1, 12]);
        /*
        This is how the tensor is arranged in a 2x3 packed texture

        0|1   2|3   4|5
        –––   –––   –––
        x|x   x|x   x|x

        6|7   8|9  10|11
        –––   –––   –––
        x|x   x|x   x|x

        Each group of four is one texel. x's represent empty channels. To
        obtain the flattened representation in the call to gl.texSubImage2D
        below, one moves through the texels from left to right, top to bottom,
        reading off the 4 channels for each texel (x's become 0's).
         */

        gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, tex);
        gpgpu.gl.texSubImage2D(
            gpgpu.gl.TEXTURE_2D, 0, 0, 0, 3, 2, gpgpu.gl.RGBA, gpgpu.gl.FLOAT,
            new Float32Array([
              0, 1, 0, 0, 2, 3, 0, 0, 4,  5,  0, 0,
              6, 7, 0, 0, 8, 9, 0, 0, 10, 11, 0, 0
            ]));

        const result =
            gpgpu.downloadMatrixFromPackedTexture(tex, 1, 1, 12, 4, 6);

        expectArraysClose(result, mat.dataSync());
      });

      it('should work when different batches occupy the same physical row',
         () => {
           const gpgpu = new GPGPUContext();
           const textureConfig = gpgpu_util.getTextureConfig(gpgpu.gl);

           // these dimensions will be halved to create the packed texture
           const physicalRows = 10;
           const physicalCols = 16;
           const tex = gpgpu_util.createPackedMatrixTexture(
               gpgpu.gl, physicalRows, physicalCols, textureConfig);

           const mat = tf.tensor3d(
               [
                 1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,
                 13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,
                 25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,
                 37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,
                 49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,
                 61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,
                 73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,
                 85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,
                 97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108,
                 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120
               ],
               [2, 20, 3]);
           /*
           Here we pretend that gl.MAX_TEXTURE_SIZE is small enough that the
           texture dimensions must be squarified. This way, values from
           different batches are encoded in the same physical row of the
           texture

            Physical row 1:
             1| 2   3| x   7| 8   9| x  13|14  15| x  19|20  21| x
            -----  -----  -----  -----  -----  -----  -----  -----
             4| 5   6| x  10|11  12| x  16|17  18| x  22|23  24| x

            Row 2:
            25|26  27| x  31|32  33| x  37|38  39| x  43|44  45| x
            -----  -----  -----  -----  -----  -----  -----  -----
            28|29  30| x  34|35  36| x  40|41  42| x  46|47  48| x

            Row 3:
            49|50  51| x  55|56  57| x  61|62  63| x  67|68  69| x
            -----  -----  -----  -----  -----  -----  -----  -----
            52|53  54| x  58|59  60| x  64|65  66| x  70|71  72| x

            Row 4:
            73|74  75| x  79|80  81| x  85|86  87| x  91|92  93| x
            -----  -----  -----  -----  -----  -----  -----  -----
            76|77  78| x  82|83  84| x  88|89  90| x  94|95  96| x

            Row 5:
            97|98  99| x 103|104105| x 109|110111| x 115|116117| x
            -----  -----  -----  -----  -----  -----  -----  -----
           100|101102| x 106|107108| x 112|113114| x 118|119120| x

           Note that physical row 3 is split between the two batches.
             */

           gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, tex);
           gpgpu.gl.texSubImage2D(
               gpgpu.gl.TEXTURE_2D, 0, 0, 0, 8, 5, gpgpu.gl.RGBA,
               gpgpu.gl.FLOAT, new Float32Array([
                 1,   2,   4,   5,   3,   0,   6,   0,   7,   8,   10,  11,
                 9,   0,   12,  0,   13,  14,  16,  17,  15,  0,   18,  0,
                 19,  20,  22,  23,  21,  0,   24,  0,   25,  26,  28,  29,
                 27,  0,   30,  0,   31,  32,  34,  35,  33,  0,   36,  0,
                 37,  38,  40,  41,  39,  0,   42,  0,   43,  44,  46,  47,
                 45,  0,   48,  0,   49,  50,  52,  53,  51,  0,   54,  0,
                 55,  56,  58,  59,  57,  0,   60,  0,   61,  62,  64,  65,
                 63,  0,   66,  0,   67,  68,  70,  71,  69,  0,   72,  0,
                 73,  74,  76,  77,  75,  0,   78,  0,   79,  80,  82,  83,
                 81,  0,   84,  0,   85,  86,  88,  89,  87,  0,   90,  0,
                 91,  92,  94,  95,  93,  0,   96,  0,   97,  98,  100, 101,
                 99,  0,   102, 0,   103, 104, 106, 107, 105, 0,   108, 0,
                 109, 110, 112, 113, 111, 0,   114, 0,   115, 116, 118, 119,
                 117, 0,   120, 0
               ]));

           const result = gpgpu.downloadMatrixFromPackedTexture(
               tex, 2, 20, 3, physicalRows, physicalCols);
           expectArraysClose(result, mat.dataSync());
         });
    });
