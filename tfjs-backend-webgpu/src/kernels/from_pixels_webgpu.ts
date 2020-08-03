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

import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class FromPixelsProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  userCode: string;
  variableNames = ['A'];
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  pipeline: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;

  uniform: GPUBuffer;
  lastUniformContent = [0, 0, 0];

  inputTexture: GPUTexture = null;
  lastPixelSize = {width: 0, height: 0};

  constructor(outputShape: number[]) {
    this.outputShape = outputShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape);
    const workGroupSize = 256;
    this.userCode = `
    layout (local_size_x = ${workGroupSize},
      local_size_y = 1,
      local_size_z = 1) in;
    layout(set = 0, binding = 1, rgba8) uniform readonly image2D srcImage;
    layout(set = 0, binding = 2) uniform Meta {
      int size;
      int width;
      int numChannels;} outShape;

    ivec3 getOutputCoordsIB() {
      int index0 = int(gl_GlobalInvocationID[0]) * outShape.numChannels;
      int d0 = index0 / (outShape.numChannels * outShape.width);
      index0 -= d0 * (outShape.numChannels * outShape.width);
      int d1 = index0 / outShape.numChannels;
      int d2 = index0 - d1 * outShape.numChannels;
      return ivec3(d0,d1,d2);
    }

    void main() {
      ivec3 coords = getOutputCoordsIB();
      int texR = coords[0];
      int texC = coords[1];
      int depth = coords[2];
      vec4 values = imageLoad(srcImage, ivec2(texC, texR));
      float value;
      for(int i = 0; i < outShape.numChannels; i++) {
        value = values[i];
        int flatIndex =
          int(gl_GlobalInvocationID.x) * outShape.numChannels + i;
        if (flatIndex < outShape.size) {
          result[flatIndex] = int(floor(255.0 * value));
        }
      }

    }
    `;
    this.shaderKey = 'fromPixel';
  }

  setProperties(
      bindGroupLayout: GPUBindGroupLayout, pipeline: GPUComputePipeline) {
    this.bindGroupLayout = bindGroupLayout;
    this.pipeline = pipeline;
  }

  setUniform(device: GPUDevice, uniformData: number[]) {
    const [uniformBuffer, uniformMapping] =
        device.createBufferMapped({size: 4 * 3, usage: GPUBufferUsage.UNIFORM});
    new Uint32Array(uniformMapping).set(uniformData);
    uniformBuffer.unmap();
    this.uniform = uniformBuffer;
  }

  getInputTexture(device: GPUDevice, pixelWidth: number, pixelHeight: number):
      GPUTexture {
    if (!this.inputTexture || this.lastPixelSize.width !== pixelWidth ||
        this.lastPixelSize.height !== pixelHeight) {
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

  generateEncoder(device: GPUDevice, output: GPUBuffer, uniformData: number[]):
      GPUCommandEncoder {
    if (uniformData &&
        (uniformData[0] !== this.lastUniformContent[0] ||
         uniformData[1] !== this.lastUniformContent[1] ||
         uniformData[2] !== this.lastUniformContent[2])) {
      // TODO(tfjs-optimization): using new mapAsync API to update the buffer.
      this.setUniform(device, uniformData);
    }
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
    passEncoder.dispatch(this.dispatch[0], this.dispatch[1], this.dispatch[2]);
    // passEncoder.dispatch(Math.ceil(size / numChannels / workGroupSize));
    passEncoder.endPass();
    return commandEncoder;
  }
}
