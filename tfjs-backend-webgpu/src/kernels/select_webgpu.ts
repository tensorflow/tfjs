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
import {getWorkGroupSizeStringWgsl} from '../shader_preprocessor_wgsl';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {getUseWgsl, WebGPUProgram} from './webgpu_program';

export class SelectProgram implements WebGPUProgram {
  variableNames = ['c', 'a', 'b'];
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workGroupSize: [number, number, number] = [64, 1, 1];
  cRank: number;
  rank: number;
  size: number;
  useWgsl: boolean;

  constructor(cRank: number, shape: number[], rank: number) {
    this.outputShape = shape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

    this.cRank = cRank;
    this.rank = rank;
    this.shaderKey = 'select';
    this.size = util.sizeFromShape(this.outputShape);
    this.useWgsl = getUseWgsl();
  }

  getUserCode(): string {
    let cCoords;
    let abCoords;
    if (this.rank > 4) {
      throw Error(`Where for rank ${this.rank} is not yet supported`);
    }

    if (this.rank === 1) {
      abCoords = `resRC`;
      cCoords = `resRC`;
    } else {
      const currentCoords = ['resRC.x', 'resRC.y', 'resRC.z', 'resRC.w'];
      const cCoordVars = [];
      const abCoordVars = [];
      for (let i = 0; i < this.outputShape.length; i++) {
        abCoordVars.push(`${currentCoords[i]}`);
        if (i < this.cRank) {
          cCoordVars.push(`${currentCoords[i]}`);
        }
      }
      cCoords = cCoordVars.join();
      abCoords = abCoordVars.join();
    }

    const dtype = getCoordsDataType(this.rank);
    const userCode = `
      void main() {
        int index = int(gl_GlobalInvocationID.x);
        if (index < size) {
          ${dtype} resRC = getOutputCoords();

          float cVal = getC(${cCoords});
          if (cVal >= 1.0) {
            setOutput(index, getA(${abCoords}));
          } else {
            setOutput(index, getB(${abCoords}));
          }
        }
      }
    `;
    return userCode;
  }

  getUserCodeWgsl(): string {
    // TODO(WGSL): below code can be merged with getUserCode.
    let cCoords;
    let abCoords;
    if (this.rank > 4) {
      throw Error(`Where for rank ${this.rank} is not yet supported`);
    }

    if (this.rank === 1) {
      abCoords = `resRC`;
      cCoords = `resRC`;
    } else {
      const currentCoords = ['resRC.x', 'resRC.y', 'resRC.z', 'resRC.w'];
      const cCoordVars = [];
      const abCoordVars = [];
      for (let i = 0; i < this.outputShape.length; i++) {
        abCoordVars.push(`${currentCoords[i]}`);
        if (i < this.cRank) {
          cCoordVars.push(`${currentCoords[i]}`);
        }
      }
      cCoords = cCoordVars.join();
      abCoords = abCoordVars.join();
    }

    const userCode = `
      ${getWorkGroupSizeStringWgsl(this.workGroupSize)}
      fn main([[builtin(global_invocation_id)]] globalId : vec3<u32>) {
        let index = globalId.x;
        if (index < uniforms.size) {
          let resRC = getOutputCoords(globalId);
          let cVal = getC(${cCoords});
          if (cVal >= 1.0) {
            setOutputFlat(index, getA(${abCoords}));
          } else {
            setOutputFlat(index, getB(${abCoords}));
          }
        }
      }
    `;
    return userCode;
  }
}
