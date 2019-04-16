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

export interface WebGPUBuffer {
  size: number, usage: any,
      setSubData: (index: number, buffer: ArrayBuffer|SharedArrayBuffer) => void
}

export interface WebGPUProgram {
  userCode: string;
  outputShape: number[];
  // Dispatch determines the layout of thread groups.
  dispatch: number[];
}

export interface WebGPUBinary {
  bindGroupLayout: any;
  pipeline: any;
}

export const compileProgram =
    (shaderCompiler: any, shaderKind: any, compileOptions: any, device: any,
     program: WebGPUProgram, bindings: any): WebGPUBinary => {
      const source = program.userCode;
      const result = shaderCompiler.CompileGlslToSpv(
          source, shaderKind, 'file', 'main', compileOptions);
      const error = result.GetErrorMessage();
      if (error.length) {
        throw new Error(`Shader compilation failed: ${error}`);
      }
      const code = result.GetBinary().slice(0).buffer;
      const bindGroupLayout = device.createBindGroupLayout({bindings});
      const layout =
          device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]});
      const module = device.createShaderModule({code});
      const pipeline = device.createComputePipeline(
          {layout, computeStage: {module, entryPoint: 'main'}});

      return {bindGroupLayout, pipeline};
    };

export function makeShaderKey(program: WebGPUProgram): string {
  const key = program.userCode;
  return key;
};