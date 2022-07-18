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

import {backend_util} from '@tensorflow/tfjs-core';
import {getMainHeaderAndGlobalIndexString, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class ConcatProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames: string[];
  uniforms = '';
  workPerThread = 4;
  workGroupSize: [number, number, number] = [64, 1, 1];
  size = true;
  offsetLength: number;

  constructor(shapes: Array<[number, number]>) {
    this.outputShape =
        backend_util.computeOutShape(shapes, 1 /* axis */) as [number, number];
    this.variableNames = shapes.map((_, i) => `T${i}`);
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);

    this.offsetLength = shapes.length - 1;
    for (let i = 0; i < this.offsetLength; i++) {
      this.uniforms += `offset${i} : i32,`;
    }
    this.shaderKey = 'concat';
  }

  getUserCode(): string {
    const snippets: string[] = [];
    if (this.offsetLength > 0) {
      snippets.push(
          `if (yC < uniforms.offset0){ setOutputAtCoords(coords.x, coords.y, getT0(yR, yC)); }`);
      for (let i = 1; i < this.offsetLength; i++) {
        snippets.push(
            `else if (yC < uniforms.offset${[i]}){ ` +
            `setOutputAtCoords(coords.x, coords.y, getT${
                i}(yR, yC - uniforms.offset${i - 1})); }`);
      }
      const lastIndex = this.offsetLength;
      const lastShiftIndex = this.offsetLength - 1;
      snippets.push(`else { setOutputAtCoords(coords.x, coords.y, getT${
          lastIndex}(yR, yC - uniforms.offset${lastShiftIndex})); }`);
    } else {
      snippets.push(`setOutputAtCoords(coords.x, coords.y, getT0(yR, yC));`);
    }

    const userCode = `
      ${getMainHeaderAndGlobalIndexString()}
        for(var i = 0; i < ${this.workPerThread}; i = i + 1) {
          let flatIndex = index * ${this.workPerThread} + i;
          if(flatIndex < uniforms.size) {
            let coords = getCoordsFromIndex(flatIndex);
            let yR = coords.x;
            let yC = coords.y;

            ${snippets.join('\n        ')}
          }
        }
      }
    `;
    return userCode;
  }
}
