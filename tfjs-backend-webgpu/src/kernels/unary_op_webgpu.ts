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

import {getCoordsDataType} from '../shader_preprocessor';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export const RELU = 'return max(a, 0.0);';

export const SIGMOID = `return 1.0 / (1.0 + exp(-1.0 * a));`;

export class UnaryOpProgram implements WebGPUProgram {
  outputShape: number[];
  userCode: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['A'];

  constructor(outputShape: number[], op: string) {
    this.outputShape = outputShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape);
    const type = getCoordsDataType(this.outputShape.length);

    const workPerThread = 2;

    this.userCode = `
      float unaryOperation(float a) {
        ${op}
      }

      void main() {
        uint index = gl_GlobalInvocationID.x;

        if(mod(index, ${workPerThread}) == 0) {
          for(uint i=0; i<${workPerThread}; i++) {
            if(index + 1 < ${this.dispatch[0]}) {
              ${type} coords = getCoordsFromFlatIndex(index + i);
              float a = getA(coords);

              setOutput(index + i, unaryOperation(a));
            }
          }
        }
      }
    `;
  }
}
