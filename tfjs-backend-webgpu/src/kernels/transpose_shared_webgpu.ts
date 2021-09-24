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

import {getWorkGroupSizeStringWgsl} from '../shader_preprocessor_wgsl';
import {computeDispatch} from '../webgpu_util';

import {getUseWgsl, WebGPUProgram} from './webgpu_program';

export class TransposeSharedProgram implements WebGPUProgram {
  variableNames = ['A'];
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[]};
  dispatch: [number, number, number];
  // Note that the maximum number of workgroup invocations by webgpu is 256.
  workGroupSize: [number, number, number] = [16, 16, 1];
  useWgsl: boolean;

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
    this.useWgsl = getUseWgsl();
  }

  getUserCode(): string {
    const userCode = `
    const int TILE_DIM = ${this.workGroupSize[0]};
    shared float tile[TILE_DIM][TILE_DIM + 1];
    void main() {
        int index = int(gl_GlobalInvocationID.x);
        int x = int(gl_WorkGroupID.x) * TILE_DIM + int(gl_LocalInvocationID.x);
        int y = int(gl_WorkGroupID.y) * TILE_DIM + int(gl_LocalInvocationID.y);
        int width = outShape[0];
        int height = outShape[1];
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

  getUserCodeWgsl(): string {
    const userCode = `
    let TILE_DIM = ${this.workGroupSize[0]}u;
    var<workgroup> tile : array<array<f32, ${this.workGroupSize[0] + 1}>, ${
        this.workGroupSize[0]}>;
    ${getWorkGroupSizeStringWgsl(this.workGroupSize)}
    fn main([[builtin(local_invocation_id)]] localId : vec3<u32>, [[builtin(global_invocation_id)]] globalId : vec3<u32>) {
        let index = globalId.x;
        let workGroupID = (globalId - localId)/vec3<u32>(${
        this.workGroupSize[0]}u, ${this.workGroupSize[1]}u, ${
        this.workGroupSize[2]}u);
        var x = workGroupID.x * TILE_DIM + localId.x;
        var y = workGroupID.y * TILE_DIM + localId.y;
        let width = uniforms.outShape[0];
        let height = uniforms.outShape[1];
        if (x < width && y < height) {
          tile[localId.y][localId.x] =
              A.numbers[y * width + x];
        }
        workgroupBarrier();

        x = workGroupID.y * TILE_DIM + localId.x;
        y = workGroupID.x * TILE_DIM + localId.y;
        if (x < height && y < width) {
          setOutputFlat((y * height + x), tile[localId.x]
            [localId.y]);
        }
      }
    `;
    return userCode;
  }
}
