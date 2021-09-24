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

import {getGlobalIndexString, getMainHeaderString} from '../shader_preprocessor';
import {computeDispatch} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class TransposeSharedProgram implements WebGPUProgram {
  variableNames = ['A'];
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[]};
  dispatch: [number, number, number];
  // Note that the maximum number of workgroup invocations by webgpu is 256.
  workGroupSize: [number, number, number] = [16, 16, 1];

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
      let TILE_DIM = ${this.workGroupSize[0]};
      var<workgroup> tile : array<array<f32, ${this.workGroupSize[0] + 1}>, ${
        this.workGroupSize[0]}>;
      ${getMainHeaderString()} {
        ${getGlobalIndexString()}
        let workGroupID = (globalId - localId)/vec3<u32>(${
        this.workGroupSize[0]}u, ${this.workGroupSize[1]}u, ${
        this.workGroupSize[2]}u);
        var x = i32(workGroupID.x) * TILE_DIM + i32(localId.x);
        var y = i32(workGroupID.y) * TILE_DIM + i32(localId.y);
        let width = uniforms.outShape[0];
        let height = uniforms.outShape[1];
        if (x < width && y < height) {
          tile[localId.y][localId.x] =
              A.numbers[y * width + x];
        }
        workgroupBarrier();

        x = i32(workGroupID.y) * TILE_DIM + i32(localId.x);
        y = i32(workGroupID.x) * TILE_DIM + i32(localId.y);
        if (x < height && y < width) {
          setOutputFlat((y * height + x), tile[localId.x]
            [localId.y]);
        }
      }
    `;
    return userCode;
  }
}
