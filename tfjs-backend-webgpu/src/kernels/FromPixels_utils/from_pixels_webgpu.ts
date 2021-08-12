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

import {computeDispatch, flatDispatchLayout, WebGPULayout} from '../../webgpu_util';
import {WebGPUProgram} from '../webgpu_program';

export class FromPixelsProgram implements WebGPUProgram {
  outputShape: number[] = [0];
  shaderKey: string;
  workPerThread: number;
  dispatchLayout: {x: number[]};
  variableNames: string[] = [];
  dispatch: [number, number, number];
  workGroupSize: [number, number, number] =
      [256, 1, 1];  // The empirical value.

  pipeline: GPUComputePipeline;
  uniform: GPUBuffer;
  lastUniformData: number[] = [];

  inputTexture: GPUTexture = null;
  layout: WebGPULayout = null;
  lastPixelSize = {width: 0, height: 0};

  private disposed = false;

  updateOutputShape(outputShape: number[]) {
    if (util.arraysEqual(this.outputShape, outputShape)) {
      return;
    }

    this.outputShape = outputShape;
    this.workPerThread = outputShape[2];  // numChannels in outputShape.
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);
  }

  constructor() {
    this.shaderKey = 'fromPixels';
  }

  getUserCode(): string {
    const userCode = `
    layout (local_size_x = ${this.workGroupSize[0]},
      local_size_y = 1,
      local_size_z = 1) in;
    layout(set = 0, binding = 1, rgba8) uniform readonly image2D srcImage;
    layout(set = 0, binding = 2) uniform Meta {
      int size;
      int numChannels;
      ivec2 outShapeStrides;
    };

    ivec3 getCoordsFromFlatIndex(int flatIndexBase);

    void main() {
      int flatIndexBase = int(gl_GlobalInvocationID.x) * numChannels;
      ivec3 coords = getCoordsFromFlatIndex(flatIndexBase);
      int texR = coords[0];
      int texC = coords[1];
      int depth = coords[2];
      vec4 values = imageLoad(srcImage, ivec2(texC, texR));
      for(int i = 0; i < numChannels; i++) {
        float value = values[i];
        int flatIndex = flatIndexBase + i;
        if (flatIndex < size) {
          result[flatIndex] = int(floor(255.0 * value));
        }
      }
    }
    `;
    return userCode;
  }

  setPipeline(pipeline: GPUComputePipeline) {
    this.pipeline = pipeline;
  }

  setUniform(device: GPUDevice, uniformData: number[]) {
    // Create the uniform buffer if it does not exist.
    // The uniform buffer size is fixed so we can hold
    // and reuse it always.
    if (!this.uniform) {
      const uniformBuffer = device.createBuffer({
        size: uniformData.length *
            4,  // The uniform buffer contains two 4 bytes element always.
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      });

      this.uniform = uniformBuffer;
    }

    // No need to update uniform buffer if no changes.
    if (!uniformData ||
        ((uniformData.length === this.lastUniformData.length) &&
         uniformData.every((v, i) => v === this.lastUniformData[i]))) {
      return;
    }

    device.queue.writeBuffer(this.uniform, 0, new Uint32Array(uniformData));

    this.lastUniformData = uniformData;
  }

  makeInputTexture(device: GPUDevice, pixelWidth: number, pixelHeight: number):
      GPUTexture {
    if (!this.inputTexture || this.lastPixelSize.width !== pixelWidth ||
        this.lastPixelSize.height !== pixelHeight) {
      if (this.inputTexture) {
        this.inputTexture.destroy();
      }

      this.inputTexture = device.createTexture({
        size: [pixelWidth, pixelHeight],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE |
            GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.lastPixelSize.width = pixelWidth;
      this.lastPixelSize.height = pixelHeight;
    }
    return this.inputTexture;
  }

  dispose() {
    if (this.disposed) {
      return;
    }
    if (this.uniform) {
      this.uniform.destroy();
    }
    if (this.inputTexture) {
      this.inputTexture.destroy();
    }
    this.disposed = true;
  }

  getLayout(device: GPUDevice): WebGPULayout {
    if (this.layout === null) {
      this.layout = this.createTextureLayout(device);
    }
    return this.layout;
  }

  private createTextureLayout(device: GPUDevice): WebGPULayout {
    const bindGroupLayoutEntries: GPUBindGroupLayoutEntry[] = [];
    // Output buffer binding layout.
    bindGroupLayoutEntries.push({
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {type: 'storage' as const}
    });
    // Input buffer binding layout.
    bindGroupLayoutEntries.push({
      binding: 1,
      visibility: GPUShaderStage.COMPUTE,
      storageTexture: {access: 'read-only', format: 'rgba8unorm'}
    });
    // Uniform buffer binding layout.
    bindGroupLayoutEntries.push({
      binding: 2,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {type: 'uniform' as const}
    });
    const fromPixelBindGroupLayout =
        device.createBindGroupLayout({entries: bindGroupLayoutEntries});
    const fromPixelPipelineLayout = device.createPipelineLayout(
        {bindGroupLayouts: [fromPixelBindGroupLayout]});
    return {
      bindGroupLayout: fromPixelBindGroupLayout,
      pipelineLayout: fromPixelPipelineLayout
    };
  }
}
