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

import {DataType, Tensor} from '@tensorflow/tfjs-core';
import * as shaderc from '@webgpu/shaderc';

import * as shader_preprocessor from '../shader_preprocessor';

export interface WebGPUProgram {
  userCode: string;
  outputShape: number[];
  // dispatchLayout enumerates how tensor dimensions are distributed among
  // dispatch x,y,z dimensions.
  dispatchLayout: {x: number[], y?: number[], z?: number[]};
  // dispatch specifies geometry of thread groups - derived from dispatchLayout.
  dispatch: [number, number, number];
  variableNames: string[];
  uniforms?: string;
  // Size of register cache in one dimension (assumes square cache).
  // Each thread writes to workPerThread * workPerThread locations in the output
  // buffer.
  workPerThread?: number;
  // workGroupSize.x * workGroupSize.y * workGroupSize.z = the number of threads
  // in a thread group. Individual dimensions determines thread layout within
  // the group.
  workGroupSize?: [number, number, number];
}

export interface WebGPUBinary {
  bindGroupLayout: GPUBindGroupLayout;
  pipeline: GPUComputePipeline;
}

export interface TensorData {
  dtype: DataType;
}

export interface BindingInfo {
  resource: {offset: number, size: number, buffer: GPUBuffer};
}

export const makeBindGroup =
    (device: GPUDevice, bindGroupLayout: GPUBindGroupLayout,
     inputs: BindingInfo[], output: BindingInfo, uniforms?: BindingInfo) => {
      const bindings = [output, ...inputs];
      if (uniforms) {
        bindings.push(uniforms);
      }
      return device.createBindGroup({
        layout: bindGroupLayout,
        bindings: bindings.map((b, i) => ({binding: i, resource: b.resource})),
      });
    };

const makeBindGroupLayout =
    (device: GPUDevice, inputs: shader_preprocessor.InputInfo[], output: Tensor,
     uniforms?: BindingInfo): GPUBindGroupLayout => {
      const bindings = Array(1 + inputs.length).fill({
        visibility: GPUShaderStage.COMPUTE,
        type: 'storage-buffer' as GPUBindingType
      });
      if (uniforms) {
        bindings.push({
          visibility: GPUShaderStage.COMPUTE,
          type: 'uniform-buffer' as GPUBindingType
        });
      }
      return device.createBindGroupLayout({
        bindings: bindings.map((b, i) => ({binding: i, ...b})),
      });
    };

export const compileProgram =
    (shaderCompiler: shaderc.Compiler, shaderKind: shaderc.ShaderKind,
     compileOptions: shaderc.CompileOptions, device: GPUDevice,
     program: WebGPUProgram, inputsData: shader_preprocessor.InputInfo[],
     output: Tensor, uniforms?: BindingInfo): WebGPUBinary => {
      const outputData = {dtype: output.dtype, shape: output.shape};

      const source =
          shader_preprocessor.makeShader(inputsData, outputData, program);
      const result = shaderCompiler.CompileGlslToSpv(
          source, shaderKind, 'file', 'main', compileOptions);
      const error = result.GetErrorMessage();
      if (error.length) {
        console.error(
            source.split('\n')
                .map((s, l) => (l + 1).toString().padStart(5, ' ') + ' ' + s)
                .join('\n'));
        throw new Error(`Shader compilation failed: ${error}`);
      }
      const bindGroupLayout =
          makeBindGroupLayout(device, inputsData, output, uniforms);
      const code = result.GetBinary();
      const layout =
          device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]});
      const module = device.createShaderModule({code});
      const pipeline = device.createComputePipeline(
          {layout, computeStage: {module, entryPoint: 'main'}});

      return {bindGroupLayout, pipeline};
    };

// TODO: Consider allowing each program to specify its own shader key. E.g. some
// kernels account for different work group sizes, but some don't.
// TODO: Consider uploading shape info as vec4s regardless of rank to reduce
// recompilation.
export function makeShaderKey(program: WebGPUProgram, ranks: number[]): string {
  const key = (program.workGroupSize ? program.workGroupSize.join(',') : '') +
      ranks.join(',') + program.userCode;
  return key;
}
