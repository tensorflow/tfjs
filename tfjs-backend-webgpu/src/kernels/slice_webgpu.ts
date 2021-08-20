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

import {util} from '@tensorflow/tfjs-core';
import {getCoordsDataType} from '../shader_preprocessor';
import {getCoordsDataTypeWgsl, getWorkGroupSizeStringWgsl} from '../shader_preprocessor_wgsl';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {getUseWgsl, WebGPUProgram} from './webgpu_program';

export class SliceProgram implements WebGPUProgram {
  variableNames = ['source'];
  uniforms: string;
  uniformsWgsl: string;
  outputShape: number[];
  shaderKey: string;
  rank: number;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workPerThread = 1;
  workGroupSize: [number, number, number] = [64, 1, 1];
  start: number[];
  size: number;
  useWgsl: boolean;

  constructor(start: number[], destSize: number[]) {
    this.outputShape = destSize;
    this.rank = destSize.length;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);

    this.start = start;
    this.uniforms = `${getCoordsDataType(start.length)} start; `;
    this.uniformsWgsl = `start : ${getCoordsDataTypeWgsl(start.length)}; `;
    this.shaderKey = 'slice';
    this.size = util.sizeFromShape(this.outputShape);
    this.useWgsl = getUseWgsl();
  }

  getUserCode(): string {
    const dtype = getCoordsDataType(this.rank);
    const sourceCoords = getCoords(this.rank);
    let coordSum;
    if (this.start.length === 1) {
      coordSum = this.outputShape.map((_, i) => {
        return `sourceLoc.${coords[i]} = start + coords.${coords[i]};`;
      });
    } else {
      coordSum = this.outputShape.map((_, i) => {
        return `sourceLoc.${coords[i]} = start[${i}] + coords.${coords[i]};`;
      });
    }

    const userCode = `
      void main() {
        int index = int(gl_GlobalInvocationID.x);
        if (index < size)
        {
          ${dtype} sourceLoc;
          ${dtype} coords = getOutputCoords();
          ${coordSum.join('\n')}
          setOutput(index, getSource(${sourceCoords}));
        }
      }
    `;
    return userCode;
  }

  getUserCodeWgsl(): string {
    const dtype = getCoordsDataTypeWgsl(this.rank);
    const sourceCoords = getCoords(this.rank);
    let coordSum;
    if (this.start.length === 1) {
      coordSum = this.outputShape.map((_, i) => {
        return `sourceLoc = uniforms.start + coords;`;
      });
    } else {
      coordSum = this.outputShape.map((_, i) => {
        return `sourceLoc.${coords[i]} = uniforms.start[${i}] + coords.${
            coords[i]};`;
      });
    }

    const userCode = `
      ${getWorkGroupSizeStringWgsl(this.workGroupSize)}
      fn main([[builtin(global_invocation_id)]] globalId : vec3<u32>) {
        let index = globalId.x;
        if (index < uniforms.size)
        {
          var sourceLoc : ${dtype};
          let coords = getOutputCoords(globalId);
          ${coordSum.join('\n')}
          setOutputFlat(index, getSource(${sourceCoords}));
        }
      }
    `;
    return userCode;
  }
}

const coords = ['x', 'y', 'z', 'w', 'u', 'v'];

function getCoords(rank: number): string {
  if (rank === 1) {
    return 'sourceLoc';
  } else if (rank <= 6) {
    return coords.slice(0, rank).map(coord => `sourceLoc.${coord}`).join(',');
  } else {
    throw Error(`Slicing for rank ${rank} is not yet supported`);
  }
}
