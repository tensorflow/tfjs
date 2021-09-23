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

import {WebGPULayout} from '../../webgpu_util';
import {FromPixelsProgram} from './from_pixels_webgpu';

export class FromPixelsImportProgram extends FromPixelsProgram {
  layout: WebGPULayout = null;
  useImport = true;

  getUserCodeWgsl(): string {
    return this.makeFromPixelsSource();
  }

  getLayout(device: GPUDevice): WebGPULayout {
    if (this.layout === null) {
      this.layout = this.createTextureImportLayout(device);
    }
    return this.layout;
  }

  private createTextureImportLayout(device: GPUDevice): WebGPULayout {
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
      externalTexture: {},
    });
    // Uniform buffer binding layout.
    bindGroupLayoutEntries.push(
        {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {}});
    const fromPixelImportBindGroupLayout =
        device.createBindGroupLayout({entries: bindGroupLayoutEntries});
    const fromPixelImportPipelineLayout = device.createPipelineLayout(
        {bindGroupLayouts: [fromPixelImportBindGroupLayout]});
    return {
      bindGroupLayout: fromPixelImportBindGroupLayout,
      pipelineLayout: fromPixelImportPipelineLayout
    };
  }
}
