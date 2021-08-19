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

import {getWorkGroupSizeStringWgsl} from '../shader_preprocessor_wgsl';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {getUseWgsl, WebGPUProgram} from './webgpu_program';

export class GatherProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames: string[] = ['A', 'indices'];
  workGroupSize: [number, number, number] = [64, 1, 1];
  aShape: number[];
  size: number;
  useWgsl: boolean;

  constructor(aShape: number[], outputShape: number[]) {
    this.outputShape = aShape.slice();
    this.aShape = aShape;
    this.outputShape = outputShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);
    this.shaderKey = `gather`;
    this.size = util.sizeFromShape(this.outputShape);
    this.useWgsl = getUseWgsl();
  }
  getUserCode(): string {
    const sourceCoords = getSourceCoords(this.aShape);
    const userCode = `
      void main() {
        int index = int(gl_GlobalInvocationID.x);
        ivec4 resRC = getOutputCoords();
        if (index < size) {
          setOutput(index, getA(${sourceCoords}));
        }
      }
    `;
    return userCode;
  }

  getUserCodeWgsl(): string {
    const sourceCoords = getSourceCoords(this.aShape, 'u32');
    const userCode = `
      ${getWorkGroupSizeStringWgsl(this.workGroupSize)}
      fn main([[builtin(global_invocation_id)]] globalId : vec3<u32>) {
        let index = globalId.x;
        let resRC = getOutputCoords(globalId);
        if (index < uniforms.size) {
          setOutputFlat(index, getA(${sourceCoords}));
        }
      }
    `;
    return userCode;
  }
}

// The input and output are always flattened into rank 4 tensors.
function getSourceCoords(aShape: number[], typePrefix = 'int'): string {
  const currentCoords = ['resRC.x', 'resRC.y', 'resRC.z', 'resRC.w'];
  const sourceCoords = [];
  for (let i = 0; i < aShape.length; i++) {
    if (i === 2) {
      sourceCoords.push(`${typePrefix}(getIndices(resRC.x, resRC.z))`);
    } else {
      sourceCoords.push(`${currentCoords[i]}`);
    }
  }
  return sourceCoords.join();
}
