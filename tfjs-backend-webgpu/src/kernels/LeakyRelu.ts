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

 import {KernelConfig, KernelFunc, LeakyRelu,
  LeakyReluInputs, LeakyReluAttrs, TensorInfo} from '@tensorflow/tfjs-core';
 import {getMainHeaderAndGlobalIndexString} from '../shader_preprocessor';
 import {WebGPUBackend} from '../backend_webgpu';
 import {computeDispatch, flatDispatchLayout} from '../webgpu_util';
 import {WebGPUProgram} from './webgpu_program';

 class LeakyReluProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['A'];
  uniforms = 'alpha : f32;';
  workGroupSize: [number, number, number];
  size = true;

  constructor(outputShape: number[]) {
    // TODO: Heuristically select a good work group size.
    const workGroupSizeX = 128;
    this.workGroupSize = [workGroupSizeX, 1, 1];
    this.outputShape = outputShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);
    this.shaderKey = `leakyrelu`;
  }

  getUserCode(): string {
    return `
      fn leakyReluOperation(a : f32, b : f32) -> f32 {
        if (a < 0.0) { return b * a; }  return a;
      }
      ${getMainHeaderAndGlobalIndexString()}
        if (index < uniforms.size) {
          let a = getAAtOutCoordsByGlobalIndex(index);
          setOutputFlat(index, leakyReluOperation(a, uniforms.alpha));
        }
      }
      `;
  }
}

 export function leakyRelu(args:
  {inputs: LeakyReluInputs, backend: WebGPUBackend, attrs: LeakyReluAttrs}):
 TensorInfo {
   const {inputs, backend, attrs} = args;
   const {x} = inputs;
   const {alpha} = attrs;
   const program = new LeakyReluProgram(x.shape);
   const uniformData = [{type: 'float32', data: [alpha]}];
   return backend.runWebGPUProgram(program, [x], 'float32', uniformData);
 }

 export const leakyReluConfig: KernelConfig = {
   kernelName: LeakyRelu,
   backendName: 'webgpu',
   kernelFunc: leakyRelu as {} as KernelFunc
 };
