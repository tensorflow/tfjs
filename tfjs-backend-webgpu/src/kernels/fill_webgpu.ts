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

import {getWorkGroupSizeStringWgsl} from '../shader_preprocessor_wgsl';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {getUseWgsl, WebGPUProgram} from './webgpu_program';

export class FillProgram implements WebGPUProgram {
  variableNames: string[] = [];
  outputShape: number[] = [];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  uniforms = 'float value;';
  uniformsWgsl = 'value : f32;';
  workPerThread = 4;
  workGroupSize: [number, number, number] = [16, 1, 1];
  size: number;
  useWgsl: boolean;

  constructor(shape: number[]) {
    this.outputShape = shape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);

    this.shaderKey = 'fill';
    this.size = util.sizeFromShape(this.outputShape);
    this.useWgsl = getUseWgsl();
  }

  getUserCode(): string {
    const userCode = `
    void main() {
      int index = int(gl_GlobalInvocationID.x);
      for (int i = 0; i < ${this.workPerThread}; i++) {
        int flatIndex = index * ${this.workPerThread} + i;
        if (flatIndex < size) {
          setOutput(flatIndex, float(value));
        }
      }
    }
  `;
    return userCode;
  }

  getUserCodeWgsl(): string {
    const userCode = `
    ${getWorkGroupSizeStringWgsl(this.workGroupSize)}
    fn main([[builtin(global_invocation_id)]] globalId : vec3<u32>) {
      let index = globalId.x;
      for (var i = 0u; i < ${this.workPerThread}u; i = i + 1u) {
        let flatIndex = index * ${this.workPerThread}u + i;
        if (flatIndex < uniforms.size) {
          setOutputFlat(flatIndex, uniforms.value);
        }
      }
    }
  `;
    return userCode;
  }
}
