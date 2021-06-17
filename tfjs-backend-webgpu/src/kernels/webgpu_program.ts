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

import {DataType, env, Rank, ShapeMap, TensorInfo} from '@tensorflow/tfjs-core';
import {Glslang} from '@webgpu/glslang/dist/web-devel/glslang.onefile';

import * as shader_preprocessor from '../shader_preprocessor';
import * as shader_preprocessor_wgsl from '../shader_preprocessor_wgsl';

export interface WebGPUProgram {
  // The unique key to distinguish different shader source code.
  shaderKey: string;
  outputShape: number[];
  // dispatchLayout enumerates how tensor dimensions are distributed among
  // dispatch x,y,z dimensions.
  dispatchLayout: {x: number[], y?: number[], z?: number[]};
  // dispatch specifies geometry of thread groups - derived from dispatchLayout.
  dispatch: [number, number, number];
  variableNames: string[];
  uniforms?: string;
  uniforms_wgsl?: string;
  // Size of register cache in one dimension (assumes square cache).
  // Each thread writes to workPerThread * workPerThread locations in the output
  // buffer.
  workPerThread?: number;
  // workGroupSize.x * workGroupSize.y * workGroupSize.z = the number of threads
  // in a thread group. Individual dimensions determines thread layout within
  // the group.
  workGroupSize?: [number, number, number];
  useWgsl?: boolean;
  isVec4?: boolean;
  // size is used for bounds checking.
  size?: number;
  getUserCode: () => string;
}

export interface TensorData {
  dtype: DataType;
}

export const makeBindGroup =
    (device: GPUDevice, bindGroupLayout: GPUBindGroupLayout,
     inputs: GPUBindingResource[], output: GPUBindingResource,
     uniforms?: GPUBindingResource) => {
      const bindings = [output, ...inputs];
      if (uniforms) {
        bindings.push(uniforms);
      }
      return device.createBindGroup({
        layout: bindGroupLayout,
        entries: bindings.map((b, i) => ({binding: i, resource: b})),
      });
    };

export const compileProgram =
    (glslang: Glslang, device: GPUDevice, program: WebGPUProgram,
     pipelineLayout: GPUPipelineLayout,
     inputsData: shader_preprocessor.InputInfo[], output: TensorInfo,
     isFromPixel = false): GPUComputePipeline => {
      const outputData = {dtype: output.dtype, shape: output.shape};

      let source;
      let module;
      if (program.useWgsl) {
        source = shader_preprocessor_wgsl.makeShader(
            inputsData, outputData, program, isFromPixel);
        module = device.createShaderModule({code: source});
      } else {
        source = shader_preprocessor.makeShader(
            inputsData, outputData, program, isFromPixel);
        const result = glslang.compileGLSLZeroCopy(source, 'compute', false);
        if (result.data.length === 0) {
          throw new Error('Shader compilation failed');
        }
        result.free();
        module = device.createShaderModule({code: result.data});
      }
      const pipeline = device.createComputePipeline(
          {layout: pipelineLayout, compute: {module, entryPoint: 'main'}});

      return pipeline;
    };

export function makeShaderKey<R extends Rank>(
    program: WebGPUProgram, shapes: Array<ShapeMap[R]>, types: string[],
    broadcastDimsKey = '', inputShapesEqualsOutShape = ''): string {
  const key = (program.workGroupSize ? program.workGroupSize.join(',') : '') +
      shapes.map(shape => shape.length).join(',') + types.join(',') +
      program.variableNames.join(',') + broadcastDimsKey +
      inputShapesEqualsOutShape + program.shaderKey;
  return key;
}

// This is global flag, but program may ignore this flag.
export function getUseWgsl () {
  return !env().getBool('WEBGPU_USE_GLSL');
}
