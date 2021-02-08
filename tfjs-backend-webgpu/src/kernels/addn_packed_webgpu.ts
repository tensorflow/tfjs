/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {util} from '@tensorflow/tfjs-core';

import {getCoordsDataType} from '../shader_preprocessor';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class AddNPackedProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames: string[];
  workPerThread = 4;
  workGroupSize: [number, number, number] = [64, 1, 1];

  constructor(shapes: number[][]) {
    this.outputShape = shapes[0];
    this.variableNames = shapes.map((_, i) => `T${i}`);
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);
    this.shaderKey = 'addN';
  }

  getUserCode(): string {
    const size = util.sizeFromShape(this.outputShape);
    const snippets: string[] = [];
    // Get target elements from every input tensor.
    this.variableNames.forEach(variable => {
      snippets.push(`float v${variable} = get${variable}AtOutCoords(coords);`);
    });
    // Calculate the sum of all elements.
    const operation = this.variableNames
                          .map(variable => {
                            return `v${variable}`;
                          })
                          .join(' + ');

    const type = getCoordsDataType(this.outputShape.length);
    const userCode = `
      void main() {
        int index = int(gl_GlobalInvocationID.x);
        for (int i = 0; i < ${this.workPerThread}; i++) {
          int flatIndex = index * ${this.workPerThread} + i;
          if (flatIndex < ${size}) {
            ${type} coords = getCoordsFromFlatIndex(flatIndex);
            ${snippets.join('\n        ')}
            setOutput(flatIndex, ${operation});
          }
        }
      }
    `;
    return userCode;
  }
}
