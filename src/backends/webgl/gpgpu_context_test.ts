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

import {ENV} from '../../environment';
import {describeWithFlags} from '../../jasmine_util';

import {WEBGL_ENVS} from './backend_webgl_test_registry';
import {getGlslDifferences} from './glsl_version';
import {GPGPUContext, linearSearchLastTrue} from './gpgpu_context';
import * as tex_util from './tex_util';

const DOWNLOAD_FLOAT_ENVS = {
  flags: {'WEBGL_DOWNLOAD_FLOAT_ENABLED': true},
  predicate: WEBGL_ENVS.predicate
};

describeWithFlags(
    'GPGPUContext setOutputMatrixTexture', DOWNLOAD_FLOAT_ENVS, () => {
      let gpgpu: GPGPUContext;
      let texture: WebGLTexture;

      beforeEach(() => {
        gpgpu = new GPGPUContext();
        // Silences debug warnings.
        spyOn(console, 'warn');
        ENV.set('DEBUG', true);
        texture = gpgpu.createFloat32MatrixTexture(1, 1);
      });

      afterEach(() => {
        gpgpu.deleteMatrixTexture(texture);
        gpgpu.dispose();
      });

      it('sets the output texture property to the output texture', () => {
        gpgpu.setOutputMatrixTexture(texture, 1, 1);
        expect(gpgpu.outputTexture).toBe(texture);
      });

      it('sets the gl viewport to the output texture dimensions', () => {
        const columns = 456;
        const rows = 123;
        const output = gpgpu.createFloat32MatrixTexture(rows, columns);
        gpgpu.setOutputMatrixTexture(output, rows, columns);
        const expected = new Int32Array([0, 0, columns, rows]);
        expect(gpgpu.gl.getParameter(gpgpu.gl.VIEWPORT)).toEqual(expected);
        gpgpu.deleteMatrixTexture(output);
      });
    });

describeWithFlags(
    'GPGPUContext setOutputPackedMatrixTexture', DOWNLOAD_FLOAT_ENVS, () => {
      let gpgpu: GPGPUContext;
      let texture: WebGLTexture;

      beforeEach(() => {
        gpgpu = new GPGPUContext();
        // Silences debug warnings.
        spyOn(console, 'warn');
        ENV.set('DEBUG', true);
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

describeWithFlags(
    'GPGPUContext setOutputMatrixWriteRegion', DOWNLOAD_FLOAT_ENVS, () => {
      let gpgpu: GPGPUContext;
      let program: WebGLProgram;
      let output: WebGLTexture;

      beforeEach(() => {
        gpgpu = new GPGPUContext();
        // Silences debug warnings.
        spyOn(console, 'warn');
        ENV.set('DEBUG', true);
        const glsl = getGlslDifferences();
        const src = `${glsl.version}
          precision highp float;
          ${glsl.defineOutput}
          void main() {
            ${glsl.output} = vec4(2,0,0,0);
          }
        `;
        program = gpgpu.createProgram(src);
        output = gpgpu.createFloat32MatrixTexture(4, 4);
        gpgpu.uploadDenseMatrixToTexture(output, 4, 4, new Float32Array(16));
        gpgpu.setOutputMatrixTexture(output, 4, 4);
        gpgpu.setProgram(program);
      });

      afterEach(() => {
        gpgpu.deleteMatrixTexture(output);
        gpgpu.deleteProgram(program);
        gpgpu.dispose();
      });

      it('sets the scissor box to the requested parameters', () => {
        gpgpu.setOutputMatrixWriteRegion(0, 1, 2, 3);
        const scissorBox = gpgpu.gl.getParameter(gpgpu.gl.SCISSOR_BOX);
        expect(scissorBox[0]).toEqual(2);
        expect(scissorBox[1]).toEqual(0);
        expect(scissorBox[2]).toEqual(3);
        expect(scissorBox[3]).toEqual(1);
      });
    });

describeWithFlags('GPGPUContext', DOWNLOAD_FLOAT_ENVS, () => {
  let gpgpu: GPGPUContext;

  beforeEach(() => {
    gpgpu = new GPGPUContext();
    // Silences debug warnings.
    spyOn(console, 'warn');
    ENV.set('DEBUG', true);
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
    const glsl = getGlslDifferences();
    const src = `${glsl.version}
      precision highp float;
      void main() {}
    `;
    const program = gpgpu.createProgram(src);
    const result = gpgpu.createFloat32MatrixTexture(1, 1);
    gpgpu.setOutputMatrixTexture(result, 1, 1);
    gpgpu.setProgram(program);
    gpgpu.deleteMatrixTexture(result);
    expect(gpgpu.executeProgram).toThrowError();
    gpgpu.deleteProgram(program);
  });
});

describe('gpgpu_context linearSearchLastTrue', () => {
  it('[false]', () => {
    const a: boolean[] = [false];
    const arr = a.map(x => () => x);
    expect(linearSearchLastTrue(arr)).toBe(-1);
  });

  it('[true]', () => {
    const a: boolean[] = [true];
    const arr = a.map(x => () => x);
    expect(linearSearchLastTrue(arr)).toBe(0);
  });

  it('[false, false]', () => {
    const a: boolean[] = [false, false];
    const arr = a.map(x => () => x);
    expect(linearSearchLastTrue(arr)).toBe(-1);
  });

  it('[true, false]', () => {
    const a: boolean[] = [true, false];
    const arr = a.map(x => () => x);
    expect(linearSearchLastTrue(arr)).toBe(0);
  });

  it('[true, true]', () => {
    const a: boolean[] = [true, true];
    const arr = a.map(x => () => x);
    expect(linearSearchLastTrue(arr)).toBe(1);
  });

  it('[false, false, false]', () => {
    const a: boolean[] = [false, false, false];
    const arr = a.map(x => () => x);
    expect(linearSearchLastTrue(arr)).toBe(-1);
  });

  it('[true, false, false]', () => {
    const a: boolean[] = [true, false, false];
    const arr = a.map(x => () => x);
    expect(linearSearchLastTrue(arr)).toBe(0);
  });

  it('[true, true, false]', () => {
    const a: boolean[] = [true, true, false];
    const arr = a.map(x => () => x);
    expect(linearSearchLastTrue(arr)).toBe(1);
  });

  it('[true, true, true]', () => {
    const a: boolean[] = [true, true, true];
    const arr = a.map(x => () => x);
    expect(linearSearchLastTrue(arr)).toBe(2);
  });

  it('[false, false, false, false]', () => {
    const a: boolean[] = [false, false, false, false];
    const arr = a.map(x => () => x);
    expect(linearSearchLastTrue(arr)).toBe(-1);
  });

  it('[true, false, false, false]', () => {
    const a: boolean[] = [true, false, false, false];
    const arr = a.map(x => () => x);
    expect(linearSearchLastTrue(arr)).toBe(0);
  });

  it('[true, true, false, false]', () => {
    const a: boolean[] = [true, true, false, false];
    const arr = a.map(x => () => x);
    expect(linearSearchLastTrue(arr)).toBe(1);
  });

  it('[true, true, true, false]', () => {
    const a: boolean[] = [true, true, true, false];
    const arr = a.map(x => () => x);
    expect(linearSearchLastTrue(arr)).toBe(2);
  });

  it('[true, true, true, true]', () => {
    const a: boolean[] = [true, true, true, true];
    const arr = a.map(x => () => x);
    expect(linearSearchLastTrue(arr)).toBe(3);
  });
});
