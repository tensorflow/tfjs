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

import {computeDispatch, flatDispatchLayout} from '../../webgpu_util';

import {WebGPUProgram} from '../webgpu_program';

export class FromPixelsProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  userCode: string;
  workPerThread:number;
  dispatchLayout: {x: number[]};
  variableNames: string[] = [];
  dispatch: [number, number, number];
  workGroupSize: [number, number, number] = [256, 1, 1]; // The empirical value.

  pipeline: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;

  uniform: GPUBuffer;
  lastUniformData = [0, 0];

  inputTexture: GPUTexture = null;
  lastPixelSize = {width: 0, height: 0};

  private disposed = false;

  constructor(outputShape: number[]) {
    this.outputShape = outputShape;
    this.workPerThread = outputShape[2];  // numChannels in outputShape.
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
      this.dispatchLayout, this.outputShape, this.workGroupSize,
      [this.workPerThread, 1, 1]);

    this.userCode = `
    layout (local_size_x = ${this.workGroupSize[0]},
      local_size_y = 1,
      local_size_z = 1) in;
    layout(set = 0, binding = 1, rgba8) uniform readonly image2D srcImage;
    layout(set = 0, binding = 2) uniform Meta {
      int size;
      int numChannels;
    } outShape;

    void main() {
      int flatIndexBase = int(gl_GlobalInvocationID.x) * outShape.numChannels;
      ivec3 coords = getCoordsFromFlatIndex(flatIndexBase);
      int texR = coords[0];
      int texC = coords[1];
      int depth = coords[2];
      vec4 values = imageLoad(srcImage, ivec2(texC, texR));
      for(int i = 0; i < outShape.numChannels; i++) {
        float value = values[i];
        int flatIndex = flatIndexBase + i;
        if (flatIndex < outShape.size) {
          result[flatIndex] = int(floor(255.0 * value));
        }
      }
    }
    `;
  }

  setWebGPUBinary(
    bindGroupLayout: GPUBindGroupLayout, pipeline: GPUComputePipeline) {
      this.bindGroupLayout = bindGroupLayout;
      this.pipeline = pipeline;
  }

  setUniform(device: GPUDevice, uniformData: number[]) {
    // No need to update uniform buffer if no changes.
    if (!uniformData ||
       (uniformData[0] === this.lastUniformData[0] &&
        uniformData[1] === this.lastUniformData[1])) {
      return;
    }

    if (this.uniform) {
      this.uniform.destroy();
    }

    const uniformBuffer = device.createBuffer({
      size: uniformData.length * 4 , // bytesLength 
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    device.defaultQueue.writeBuffer(uniformBuffer, 0, 
                                      new Uint32Array(uniformData));
    this.uniform = uniformBuffer;
  }

  makeInputTexture(device: GPUDevice,
                   pixelWidth: number,
                   pixelHeight: number): GPUTexture {
    if (!this.inputTexture || this.lastPixelSize.width !== pixelWidth ||
        this.lastPixelSize.height !== pixelHeight) {
      if (this.inputTexture) {
        this.inputTexture.destroy();
      }

      this.inputTexture = device.createTexture({
        size: {
          width: pixelWidth,
          height: pixelHeight,
          depth: 1,
        },
        format: 'rgba8unorm',
        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE,
      });
      this.lastPixelSize.width = pixelWidth;
      this.lastPixelSize.height = pixelHeight;
    }
    return this.inputTexture;
  }

  generateEncoder(device: GPUDevice, output: GPUBuffer): GPUCommandEncoder {
    const bindGroup = device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: output,
          }
        },
        {
          binding: 1,
          resource: this.inputTexture.createView(),
        },
        {
          binding: 2,
          resource: {
            buffer: this.uniform,
          }
        }
      ],
    });

    const commandEncoder = device.createCommandEncoder({});
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatch(this.dispatch[0],
                         this.dispatch[1],
                         this.dispatch[2]);
    passEncoder.endPass();
    return commandEncoder;
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
}
