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

import * as test_util from '../../../test_util';
import {GPGPUContext} from './gpgpu_context';
import * as tex_util from './tex_util';

test_util.describeCustom(
    'GPGPUContext .downloadMatrixFromTexture',
    () => {
      let gpgpu: GPGPUContext;
      let texture: WebGLTexture;

      beforeEach(() => {
        gpgpu = new GPGPUContext();
        gpgpu.enableAutomaticDebugValidation(true);
        texture = gpgpu.createMatrixTexture(1, 1);
      });

      afterEach(() => {
        gpgpu.deleteMatrixTexture(texture);
        gpgpu.dispose();
      });

      it('returns 1x1 matrix that was uploaded', () => {
        gpgpu.uploadMatrixToTexture(texture, 1, 1, new Float32Array([1.234]));
        const result = gpgpu.downloadMatrixFromTexture(texture, 1, 1);
        test_util.expectNumbersClose(result[0], 1.234);
      });

      it('returns 2x2 matrix that was uploaded', () => {
        const texture2 = gpgpu.createMatrixTexture(2, 2);
        gpgpu.uploadMatrixToTexture(
            texture2, 2, 2, new Float32Array([1.234, 2, 3, 4]));
        const result = gpgpu.downloadMatrixFromTexture(texture2, 2, 2);
        test_util.expectArraysClose(result, new Float32Array([1.234, 2, 3, 4]));
        gpgpu.deleteMatrixTexture(texture2);
      });

      it('uses texture parameter', () => {
        const texture2: WebGLTexture = gpgpu.createMatrixTexture(1, 1);
        gpgpu.uploadMatrixToTexture(texture, 1, 1, new Float32Array([1]));
        gpgpu.uploadMatrixToTexture(texture2, 1, 1, new Float32Array([2]));
        const read1 = gpgpu.downloadMatrixFromTexture(texture, 1, 1);
        const read2 = gpgpu.downloadMatrixFromTexture(texture2, 1, 1);

        test_util.expectNumbersClose(read1[0], 1);
        test_util.expectNumbersClose(read2[0], 2);

        gpgpu.deleteMatrixTexture(texture2);
      });
    },
    [
      {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
      {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
      {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
    ]);

test_util.describeCustom(
    'GPGPUContext color texture with float textures',
    () => {
      let gpgpu: GPGPUContext;
      let texture: WebGLTexture;

      afterEach(() => {
        gpgpu.deleteMatrixTexture(texture);
        gpgpu.dispose();
      });

      it('basic', () => {
        gpgpu = new GPGPUContext();
        gpgpu.enableAutomaticDebugValidation(true);
        texture = gpgpu.createMatrixTexture(1, 1);

        gpgpu.setOutputMatrixTexture(texture, 1, 1);
        gpgpu.gl.clearColor(0.123, 0, 0, 0);
        gpgpu.gl.clear(gpgpu.gl.COLOR_BUFFER_BIT);
        const result = gpgpu.downloadMatrixFromTexture(texture, 1, 1);
        test_util.expectNumbersClose(result[0], 0.123);
      });
    },
    [
      {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
      {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    ]);

test_util.describeCustom('GPGPUContext color texture with byte packing', () => {
  let gpgpu: GPGPUContext;
  let texture: WebGLTexture;

  afterEach(() => {
    gpgpu.deleteMatrixTexture(texture);
    gpgpu.dispose();
  });

  it('basic', () => {
    gpgpu = new GPGPUContext();
    gpgpu.enableAutomaticDebugValidation(true);
    texture = gpgpu.createMatrixTexture(1, 1);

    gpgpu.setOutputMatrixTexture(texture, 1, 1);
    const uintArray = tex_util.encodeFloatArray(new Float32Array([0.123]));
    gpgpu.gl.clearColor(
        uintArray[0] / 255, uintArray[1] / 255, uintArray[2] / 255,
        uintArray[3] / 255);
    gpgpu.gl.clear(gpgpu.gl.COLOR_BUFFER_BIT);
    const result = gpgpu.downloadMatrixFromTexture(texture, 1, 1);
    test_util.expectNumbersClose(result[0], 0.123);
  });
}, [{'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}]);

test_util.describeCustom(
    'GPGPUContext setOutputMatrixTexture',
    () => {
      let gpgpu: GPGPUContext;
      let texture: WebGLTexture;

      beforeEach(() => {
        gpgpu = new GPGPUContext();
        gpgpu.enableAutomaticDebugValidation(true);
        texture = gpgpu.createMatrixTexture(1, 1);
      });

      afterEach(() => {
        gpgpu.deleteMatrixTexture(texture);
        gpgpu.dispose();
      });

      it('sets the output texture property to the output texture', () => {
        gpgpu.setOutputMatrixTexture(texture, 1, 1);
        expect(gpgpu.outputTexture).toBe(texture);
      });

      it('rebinds the output texture to the color buffer target', () => {
        const output: WebGLTexture = gpgpu.createMatrixTexture(1, 1);
        gpgpu.uploadMatrixToTexture(texture, 1, 1, new Float32Array([10]));
        gpgpu.setOutputMatrixTexture(output, 1, 1);
        const tBeforeClear = gpgpu.downloadMatrixFromTexture(texture, 1, 1);
        test_util.expectNumbersClose(tBeforeClear[0], 10);
        gpgpu.gl.clearColor(1, 0, 0, 0);
        gpgpu.gl.clear(gpgpu.gl.COLOR_BUFFER_BIT);
        const tAfterClear = gpgpu.downloadMatrixFromTexture(texture, 1, 1);
        test_util.expectNumbersClose(tAfterClear[0], 10);
        gpgpu.deleteMatrixTexture(output);
      });

      it('resets output texture to null if nothing was previously bound',
         () => {
           expect(gpgpu.outputTexture).toBeNull();
           gpgpu.downloadMatrixFromTexture(texture, 1, 1);
           expect(gpgpu.outputTexture).toBeNull();
         });

      it('sets the gl viewport to the output texture dimensions', () => {
        const columns = 456;
        const rows = 123;
        const output = gpgpu.createMatrixTexture(rows, columns);
        gpgpu.setOutputMatrixTexture(output, rows, columns);
        const expected = new Int32Array([0, 0, columns, rows]);
        expect(gpgpu.gl.getParameter(gpgpu.gl.VIEWPORT)).toEqual(expected);
        gpgpu.deleteMatrixTexture(output);
      });

      it('doesn\'t change gl viewport when downloading a non-output tex',
         () => {
           const output = gpgpu.createMatrixTexture(128, 128);
           gpgpu.setOutputMatrixTexture(output, 128, 128);
           gpgpu.downloadMatrixFromTexture(texture, 1, 1);
           const expected = new Int32Array([0, 0, 128, 128]);
           expect(gpgpu.gl.getParameter(gpgpu.gl.VIEWPORT)).toEqual(expected);
           gpgpu.deleteMatrixTexture(output);
         });
    },
    [
      {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
      {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
      {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
    ]);

describe('GPGPUContext setOutputPackedMatrixTexture', () => {
  let gpgpu: GPGPUContext;
  let texture: WebGLTexture;

  beforeEach(() => {
    gpgpu = new GPGPUContext();
    gpgpu.enableAutomaticDebugValidation(true);
  });

  afterEach(() => {
    if (texture != null) {
      gpgpu.deleteMatrixTexture(texture);
    }
    gpgpu.dispose();
  });

  it('sets the output texture property to the output texture', () => {
    texture = gpgpu.createPackedMatrixTexture(1, 1);
    gpgpu.setOutputPackedMatrixTexture(texture, 1, 1);
    expect(gpgpu.outputTexture).toBe(texture);
  });

  it('sets the gl viewport to the output packed texture dimensions', () => {
    const columns = 456;
    const rows = 123;
    texture = gpgpu.createPackedMatrixTexture(rows, columns);
    gpgpu.setOutputPackedMatrixTexture(texture, rows, columns);
    const [width, height] =
        tex_util.getPackedMatrixTextureShapeWidthHeight(rows, columns);
    const expected = new Int32Array([0, 0, width, height]);
    expect(gpgpu.gl.getParameter(gpgpu.gl.VIEWPORT)).toEqual(expected);
  });
});

describe('GPGPUContext setOutputMatrixWriteRegion', () => {
  let gpgpu: GPGPUContext;
  let program: WebGLProgram;
  let output: WebGLTexture;

  beforeEach(() => {
    gpgpu = new GPGPUContext();
    gpgpu.enableAutomaticDebugValidation(true);
    const src =
        'precision highp float; void main() { gl_FragColor = vec4(2,0,0,0); }';
    program = gpgpu.createProgram(src);
    output = gpgpu.createMatrixTexture(4, 4);
    gpgpu.uploadMatrixToTexture(output, 4, 4, new Float32Array(16));
    gpgpu.setOutputMatrixTexture(output, 4, 4);
    gpgpu.setProgram(program);
  });

  afterEach(() => {
    gpgpu.deleteMatrixTexture(output);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();
  });

  it('writes to all pixels by default', () => {
    gpgpu.executeProgram();
    const result = gpgpu.downloadMatrixFromTexture(output, 4, 4);
    const expected = new Float32Array(4 * 4);
    expected.fill(2);
    test_util.expectArraysClose(result, expected);
  });

  it('sets the scissor box to the requested parameters', () => {
    gpgpu.setOutputMatrixWriteRegion(0, 1, 2, 3);
    const scissorBox = gpgpu.gl.getParameter(gpgpu.gl.SCISSOR_BOX);
    expect(scissorBox[0]).toEqual(2);
    expect(scissorBox[1]).toEqual(0);
    expect(scissorBox[2]).toEqual(3);
    expect(scissorBox[3]).toEqual(1);
  });

  it('writes only to center 2x2 region of 4x4 texture', () => {
    gpgpu.setOutputMatrixWriteRegion(1, 2, 1, 2);
    gpgpu.executeProgram();
    const result = gpgpu.downloadMatrixFromTexture(output, 4, 4);
    const expected =
        new Float32Array([0, 0, 0, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0]);
    test_util.expectArraysClose(result, expected);
  });

  it('preserves data from previous writes outside of write region', () => {
    gpgpu.setOutputMatrixWriteRegion(0, 1, 0, 4);  // top row
    gpgpu.executeProgram();
    gpgpu.setOutputMatrixWriteRegion(3, 1, 0, 4);  // bottom row
    gpgpu.executeProgram();
    const result = gpgpu.downloadMatrixFromTexture(output, 4, 4);
    const expected =
        new Float32Array([2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2]);
    test_util.expectArraysClose(result, expected);
  });

  it('writes adjacent cells across multiple calls', () => {
    for (let row = 0; row < 4; ++row) {
      for (let col = 0; col < 4; ++col) {
        gpgpu.setOutputMatrixWriteRegion(row, 1, col, 1);
        gpgpu.executeProgram();
      }
    }
    const result = gpgpu.downloadMatrixFromTexture(output, 4, 4);
    const expected = new Float32Array(4 * 4);
    expected.fill(2);
    test_util.expectArraysClose(result, expected);
  });
});

describe('GPGPUContext', () => {
  let gpgpu: GPGPUContext;

  beforeEach(() => {
    gpgpu = new GPGPUContext();
    gpgpu.enableAutomaticDebugValidation(true);
  });

  afterEach(() => {
    gpgpu.dispose();
  });

  it('throws an error if used after dispose', () => {
    const gpgpuContext = new GPGPUContext();
    gpgpuContext.dispose();
    expect(gpgpuContext.dispose).toThrowError();
  });

  it('throws an error if validation is on and framebuffer incomplete', () => {
    const src = `precision highp float; void main() {}`;
    const program = gpgpu.createProgram(src);
    const result = gpgpu.createMatrixTexture(1, 1);
    gpgpu.setOutputMatrixTexture(result, 1, 1);
    gpgpu.setProgram(program);
    gpgpu.deleteMatrixTexture(result);
    expect(gpgpu.executeProgram).toThrowError();
    gpgpu.deleteProgram(program);
  });
});
