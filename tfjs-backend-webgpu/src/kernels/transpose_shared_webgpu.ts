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

import {computeDispatch} from '../webgpu_util';
import {WebGPUProgram} from './webgpu_program';

export class TransposeSharedProgram implements WebGPUProgram {
  variableNames = ['A'];
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[]};
  dispatch: [number, number, number];
  workGroupSize: [number, number, number] = [32, 32, 1];

  constructor(aShape: number[], newDim: number[]) {
    const outputShape: number[] = new Array(aShape.length);
    for (let i = 0; i < outputShape.length; i++) {
      outputShape[i] = aShape[newDim[i]];
    }
    this.outputShape = outputShape;
    this.dispatchLayout = {x: [0], y: [1]};
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize, [1, 1, 1]);

    this.shaderKey = 'transposeShared';
  }

  getUserCode(): string {
    const userCode = `
    const int TILE_DIM = ${this.workGroupSize[0]};
    shared float tile[TILE_DIM][TILE_DIM + 1];
    void main() {
        int index = int(gl_GlobalInvocationID.x);
        int x = int(gl_WorkGroupID.x) * TILE_DIM + int(gl_LocalInvocationID.x);
        int y = int(gl_WorkGroupID.y) * TILE_DIM + int(gl_LocalInvocationID.y);
        int width = ${this.outputShape[0]};
        int height = ${this.outputShape[1]};
        if (x < width && y < height) {
          tile[gl_LocalInvocationID.y][gl_LocalInvocationID.x] =
              A[y * width + x];
        }
        barrier();

        x = int(gl_WorkGroupID.y) * TILE_DIM + int(gl_LocalInvocationID.x);
        y = int(gl_WorkGroupID.x) * TILE_DIM + int(gl_LocalInvocationID.y);
        if (x < height && y < width) {
          setOutput((y * height + x), tile[gl_LocalInvocationID.x]
            [gl_LocalInvocationID.y]);
        }
      }
    `;
    return userCode;
  }
}
