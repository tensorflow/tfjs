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

import {GPGPUContext} from './gpgpu_context';
import {GPGPUProgram} from './gpgpu_math';

export class FillProgram implements GPGPUProgram {
  variableNames: string[];
  outputShape: number[] = [];
  userCode: string;

  valueLoc: WebGLUniformLocation;

  constructor(shape: number[], value: number) {
    this.variableNames = ['x'];
    this.outputShape = shape;

    this.userCode = `
      uniform float value;
      void main() {
        // Input can be obtained from uniform value.
        setOutput(value);
      }
    `;
  }

  getCustomSetupFunc(value: number) {
    return (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) => {
      if (this.valueLoc == null) {
        this.valueLoc = gpgpu.getUniformLocationNoThrow(webGLProgram, 'value');
      }
      gpgpu.gl.uniform1f(this.valueLoc, value);
    };
  }
}
