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

import {getMainHeaderAndGlobalIndexString, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class AddNPackedProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames: string[];
  workPerThread = 4;
  workGroupSize: [number, number, number] = [64, 1, 1];
  size = true;

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
    const snippets: string[] = [];
    // Get target elements from every input tensor.
    this.variableNames.forEach(variable => {
      snippets.push(`let v${variable} = get${variable}ByOutputCoords(coords);`);
    });
    // Calculate the sum of all elements.
    const operation = this.variableNames
                          .map(variable => {
                            return `v${variable}`;
                          })
                          .join(' + ');

    const userCode = `
      ${getMainHeaderAndGlobalIndexString()}
        for (var i = 0; i < ${this.workPerThread}; i = i + 1) {
          let flatIndex = index * ${this.workPerThread} + i;
          if (flatIndex < uniforms.size) {
            let coords = getCoordsFromIndex(flatIndex);
            ${snippets.join('\n        ')}
            setOutputAtIndex(flatIndex, ${operation});
          }
        }
      }
    `;
    return userCode;
  }
}
