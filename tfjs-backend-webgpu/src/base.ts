/**
 * @license
 * Copyright 2022 Google Inc. All Rights Reserved.
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

import './flags_webgpu';

import {env, registerBackend} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from './backend_webgpu';
import {isWebGPUSupported} from './webgpu_util';

if (isWebGPUSupported()) {
  registerBackend('webgpu', async () => {
    const gpuDescriptor: GPURequestAdapterOptions = {
      powerPreference: env().get('WEBGPU_USE_LOW_POWER_GPU') ?
          'low-power' :
          'high-performance'
    };

    const adapter = await navigator.gpu.requestAdapter(gpuDescriptor);
    const deviceDescriptor: GPUDeviceDescriptor = {};

    const requiredFeatures = [];
    if (adapter.features.has('timestamp-query')) {
      requiredFeatures.push('timestamp-query');
    }
    if (adapter.features.has('bgra8unorm-storage')) {
      requiredFeatures.push(['bgra8unorm-storage']);
    }
    deviceDescriptor.requiredFeatures =
        requiredFeatures as Iterable<GPUFeatureName>;

    const adapterLimits = adapter.limits;
    deviceDescriptor.requiredLimits = {
      'maxComputeWorkgroupStorageSize':
          adapterLimits.maxComputeWorkgroupStorageSize,
      'maxComputeWorkgroupsPerDimension':
          adapterLimits.maxComputeWorkgroupsPerDimension,
      'maxStorageBufferBindingSize': adapterLimits.maxStorageBufferBindingSize,
      'maxBufferSize': adapterLimits.maxBufferSize,
      'maxComputeWorkgroupSizeX': adapterLimits.maxComputeWorkgroupSizeX,
      'maxComputeInvocationsPerWorkgroup':
          adapterLimits.maxComputeInvocationsPerWorkgroup,
    };

    const device: GPUDevice = await adapter.requestDevice(deviceDescriptor);
    const adapterInfo =
      'info' in adapter
        ? adapter.info
        : 'requestAdapterInfo' in adapter
          // tslint:disable-next-line:no-any
          ? await (adapter as any).requestAdapterInfo()
          : undefined;
    return new WebGPUBackend(device, adapterInfo);
  }, 3 /*priority*/);
}

// Export webgpu utilities
export * from './webgpu';
