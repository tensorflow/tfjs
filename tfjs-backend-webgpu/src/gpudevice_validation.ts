/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

export class GPUDeviceValidation {
  device: GPUDevice;
  private errorPromises: Array<Promise<GPUError>> = [];

  constructor(device: GPUDevice) {
    this.device = device;
  }

  pushErrorScope(filter: GPUErrorFilter = 'validation') {
    this.device.pushErrorScope(filter);
  }

  popErrorScope() {
    this.errorPromises.push(this.device.popErrorScope());
  }

  async checkValidationErrors() {
    const errors = (await Promise.all(this.errorPromises));
    let hasError = false;
    errors.forEach((error) => {
      if (error instanceof GPUValidationError) {
        hasError = true;
        console.error(`GPUValidationError: ${error.message}`);
      } else if (error) {
        hasError = true;
        console.error('GPU error!');
      }
    });
    if (hasError) {
      throw new Error('GPU error(s) should be resolved!');
    }
    this.errorPromises = [];
  }

  destroy() {
    this.pushErrorScope();
    this.device.destroy();
    this.popErrorScope();
  }

  createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer {
    this.pushErrorScope('out-of-memory');
    const buffer = this.device.createBuffer(descriptor);
    this.popErrorScope();
    return buffer;
  }

  createTexture(descriptor: GPUTextureDescriptor): GPUTexture {
    this.pushErrorScope('out-of-memory');
    const texture = this.device.createTexture(descriptor);
    this.popErrorScope();
    return texture;
  }

  createSampler(descriptor?: GPUSamplerDescriptor): GPUSampler {
    this.pushErrorScope();
    const sampler = this.device.createSampler(descriptor);
    this.popErrorScope();
    return sampler;
  }

  importExternalTexture(descriptor: GPUExternalTextureDescriptor):
      GPUExternalTexture {
    this.pushErrorScope();
    const externalTexture = this.device.importExternalTexture(descriptor);
    this.popErrorScope();
    return externalTexture;
  }

  createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor):
      GPUBindGroupLayout {
    this.pushErrorScope();
    const layout = this.device.createBindGroupLayout(descriptor);
    this.popErrorScope();
    return layout;
  }

  createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor):
      GPUPipelineLayout {
    this.pushErrorScope();
    const layout = this.device.createPipelineLayout(descriptor);
    this.popErrorScope();
    return layout;
  }

  createBindGroup(descriptor: GPUBindGroupDescriptor): GPUBindGroup {
    this.pushErrorScope();
    const bindGroup = this.device.createBindGroup(descriptor);
    this.popErrorScope();
    return bindGroup;
  }

  createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule {
    this.pushErrorScope();
    const shaderModule = this.device.createShaderModule(descriptor);
    this.popErrorScope();
    return shaderModule;
  }

  createComputePipeline(descriptor: GPUComputePipelineDescriptor):
      GPUComputePipeline {
    this.pushErrorScope();
    const pipeline = this.device.createComputePipeline(descriptor);
    this.popErrorScope();
    return pipeline;
  }

  createRenderPipeline(descriptor: GPURenderPipelineDescriptor):
      GPURenderPipeline {
    this.pushErrorScope();
    const pipeline = this.device.createRenderPipeline(descriptor);
    this.popErrorScope();
    return pipeline;
  }

  createComputePipelineAsync(descriptor: GPUComputePipelineDescriptor):
      Promise<GPUComputePipeline> {
    this.pushErrorScope();
    const pipeline = this.device.createComputePipelineAsync(descriptor);
    this.popErrorScope();
    return pipeline;
  }

  createRenderPipelineAsync(descriptor: GPURenderPipelineDescriptor):
      Promise<GPURenderPipeline> {
    this.pushErrorScope();
    const pipeline = this.device.createRenderPipelineAsync(descriptor);
    this.popErrorScope();
    return pipeline;
  }

  createCommandEncoder(descriptor?: GPUCommandEncoderDescriptor):
      GPUCommandEncoder {
    this.pushErrorScope();
    const encoder = this.device.createCommandEncoder(descriptor);
    this.popErrorScope();
    return encoder;
  }

  createRenderBundleEncoder(descriptor: GPURenderBundleEncoderDescriptor):
      GPURenderBundleEncoder {
    this.pushErrorScope();
    const encoder = this.device.createRenderBundleEncoder(descriptor);
    this.popErrorScope();
    return encoder;
  }

  createQuerySet(descriptor: GPUQuerySetDescriptor): GPUQuerySet {
    this.pushErrorScope();
    const querySet = this.device.createQuerySet(descriptor);
    this.popErrorScope();
    return querySet;
  }
}
