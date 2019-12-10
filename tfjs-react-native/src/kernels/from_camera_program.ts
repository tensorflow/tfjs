/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

// function validateTextureUnit(gl: WebGLRenderingContext, textureUnit:
// number) {
//   //@ts-ignore
//   const maxTextureUnit = gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS - 1;
//   //@ts-ignore
//   const glTextureUnit = textureUnit + gl.TEXTURE0;
//   //@ts-ignore
//   if (glTextureUnit < gl.TEXTURE0 || glTextureUnit > maxTextureUnit) {
//     const textureUnitRange = `[gl.TEXTURE0, gl.TEXTURE${maxTextureUnit}]`;
//     throw new Error(`textureUnit must be in ${textureUnitRange}.`);
//   }
// }

export function callAndCheck<T>(
    gl: WebGLRenderingContext, debugMode: boolean, func: () => T): T {
  const returnValue = func();
  if (debugMode) {
    checkWebGLError(gl);
  }
  return returnValue;
}

function checkWebGLError(gl: WebGLRenderingContext) {
  //@ts-ignore
  const error = gl.getError();
  //@ts-ignore
  if (error !== gl.NO_ERROR) {
    throw new Error('WebGL Error: ' + getWebGLErrorMessage(gl, error));
  }
}

export function getWebGLErrorMessage(
    gl: WebGLRenderingContext, status: number): string {
  switch (status) {
    //@ts-ignore
    case gl.NO_ERROR:
      return 'NO_ERROR';
      //@ts-ignore
    case gl.INVALID_ENUM:
      return 'INVALID_ENUM';
      //@ts-ignore
    case gl.INVALID_VALUE:
      return 'INVALID_VALUE';
      //@ts-ignore
    case gl.INVALID_OPERATION:
      return 'INVALID_OPERATION';
      //@ts-ignore
    case gl.INVALID_FRAMEBUFFER_OPERATION:
      return 'INVALID_FRAMEBUFFER_OPERATION';
      //@ts-ignore
    case gl.OUT_OF_MEMORY:
      return 'OUT_OF_MEMORY';
      //@ts-ignore
    case gl.CONTEXT_LOST_WEBGL:
      return 'CONTEXT_LOST_WEBGL';
    default:
      return `Unknown error code ${status}`;
  }
}

export class FromCameraProgram implements tf.webgl.GPGPUProgram {
  variableNames = [] as string[];
  userCode: string;
  outputShape: number[];

  // Caching uniform location for speed.
  camTexLoc: WebGLUniformLocation;
  myu: WebGLUniformLocation;

  constructor(outputShape: number[]) {
    const [height, width, ] = outputShape;
    this.outputShape = outputShape;
    this.userCode = `
      uniform float myTestUniform;
      uniform sampler2D cameraTexture;

      void main() {
        ivec3 coords = getOutputCoords();
        int texR = coords[0];
        int texC = coords[1];
        int depth = coords[2];
        vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${width}.0, ${height}.0);

        // Useless but program errors without it.
        vec4 texVals = vec4(10., 8., 6., myTestUniform);

        vec4 values = texture(cameraTexture, uv);
        float value;
        if (depth == 0) {
          value = values[0];
        } else if (depth == 1) {
          value = values[1];
        } else if (depth == 2) {
          value = values[2];
        } else if (depth == 3) {
          value = values[3];
        } else {
          value = texVals[3];
        }

        setOutput(floor(value * 255.0 + 0.5));
      }
    `;
  }

  getCustomSetupFunc(cameraTexture: WebGLTexture) {
    return (gpgpu: tf.webgl.GPGPUContext, webGLProgram: WebGLProgram) => {
      console.log('Inside custom setup function for FROM_CAMERA_PROGRAM');
      const gl = gpgpu.gl;
      //@ts-ignore
      console.log('gl sentinel', gl.sentinel);
      const textureUnit = 5;

      // if (this.myu == null) {
      //   this.myu =
      //       gpgpu.getUniformLocationNoThrow(webGLProgram, 'myTestUniform');
      //   if (this.myu == null) {
      //     console.warn('no location for myTestUniform');
      //     return;
      //   }
      // }

      // console.log('attempt to upload uniform');
      // callAndCheck(
      //     //@ts-ignore
      //     gl, true, () => gl.uniform1f(this.myu, 1.5));
      // console.log('attempt to upload uniform success');

      if (this.camTexLoc == null) {
        this.camTexLoc =
            gpgpu.getUniformLocationNoThrow(webGLProgram, 'cameraTexture');
        if (this.camTexLoc == null) {
          // This means the compiler has optimized and realized it doesn't need
          // the uniform.
          console.warn('no location for cameraTexture uniform');
          return;
        }
      }

      console.log('found location for cameraTexture uniform');
      console.log('cameraTexture', cameraTexture);

      // //@ts-ignore
      // gl.activeTexture(gl.TEXTURE0 + textureUnit);
      // //@ts-ignore
      // let error = gl.getError();
      // console.log('customtexfunc activeTexture', error);
      // //@ts-ignore
      // if (error !== gl.NO_ERROR) {
      //   throw new Error('WebGL Error: ' + getWebGLErrorMessage(gl, error));
      // }
      // //@ts-ignore
      // gl.bindTexture(gl.TEXTURE_2D, cameraTexture);

      // //@ts-ignore
      // error = gl.getError();
      // console.log('customtexfunc bindTexture', error);
      // //@ts-ignore
      // if (error !== gl.NO_ERROR) {
      //   throw new Error('WebGL Error: ' + getWebGLErrorMessage(gl, error));
      // }

      tf.webgl.webgl_util.bindTextureToProgramUniformSampler(
          gl,
          true,
          webGLProgram,
          cameraTexture,
          this.camTexLoc,
          textureUnit,
      );
      console.log('done with bindTextureToProgramUniformSampler');
    };
  }
}
