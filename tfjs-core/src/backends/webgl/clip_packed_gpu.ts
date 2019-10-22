/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import {GPGPUProgram} from './gpgpu_math';

export class ClipPackedProgram implements GPGPUProgram {
  variableNames = ['A'];
  packedInputs = true;
  packedOutput = true;
  userCode: string;
  outputShape: number[];

  // Caching uniform locations for speed.
  minLoc: WebGLUniformLocation;
  maxLoc: WebGLUniformLocation;

  constructor(aShape: number[]) {
    this.outputShape = aShape;
    this.userCode = `
      uniform float minVal;
      uniform float maxVal;

      void main() {
        vec4 value = getAAtOutCoords();

        if (any(isnan(value))) {
          setOutput(value);
          return;
        }

        setOutput(clamp(value, vec4(minVal), vec4(maxVal)));
      }
    `;
  }

  getCustomSetupFunc(min: number, max: number) {
    return (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) => {
      if (this.minLoc == null) {
        this.minLoc = gpgpu.getUniformLocationNoThrow(webGLProgram, 'minVal');
        this.maxLoc = gpgpu.getUniformLocationNoThrow(webGLProgram, 'maxVal');
      }
      gpgpu.gl.uniform1f(this.minLoc, min);
      gpgpu.gl.uniform1f(this.maxLoc, max);
    };
  }
}
