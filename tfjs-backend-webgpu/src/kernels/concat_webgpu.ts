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

import {backend_util, util} from '@tensorflow/tfjs-core';

import {computeDispatch, flatDispatchLayout} from '../webgpu_util';
import {WebGPUProgram} from './webgpu_program';

export class ConcatProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames: string[];
  workPerThread = 4;
  workGroupSize: [number, number, number] = [64, 1, 1];
  shapes: Array<[number, number]>;

  constructor(shapes: Array<[number, number]>) {
    this.outputShape =
        backend_util.computeOutShape(shapes, 1 /* axis */) as [number, number];
    this.variableNames = shapes.map((_, i) => `T${i}`);
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);

    this.shapes = shapes;
    this.shaderKey = 'concat';
  }

  getUserCode(): string {
    const offsets: number[] = new Array(this.shapes.length - 1);
    const snippets: string[] = [];
    if (offsets.length > 0) {
      offsets[0] = this.shapes[0][1];
      for (let i = 1; i < offsets.length; i++) {
        offsets[i] = offsets[i - 1] + this.shapes[i][1];
      }

      snippets.push(`if (yC < ${
          offsets[0]}) setOutput(coords.x, coords.y, getT0(yR, yC));`);
      for (let i = 1; i < offsets.length; i++) {
        const shift = offsets[i - 1];
        snippets.push(
            `else if (yC < ${offsets[i]}) ` +
            `setOutput(coords.x, coords.y, getT${i}(yR, yC-${shift}));`);
      }
      const lastIndex = offsets.length;
      const lastShift = offsets[offsets.length - 1];
      snippets.push(`else setOutput(coords.x, coords.y, getT${
          lastIndex}(yR, yC-${lastShift}));`);
    } else {
      snippets.push(`setOutput(coords.x, coords.y, getT0(yR, yC));`);
    }
    const size = util.sizeFromShape(this.outputShape);
    const userCode = `
      void main() {
        int index = int(gl_GlobalInvocationID.x);

        for(int i = 0; i < ${this.workPerThread}; i++) {
          int flatIndex = index * ${this.workPerThread} + i;
          if(flatIndex < ${size}) {
            ivec2 coords = getCoordsFromFlatIndex(flatIndex);
            int yR = coords.x;
            int yC = coords.y;

            ${snippets.join('\n        ')}
          }
        }
      }
    `;
    return userCode;
  }
}
