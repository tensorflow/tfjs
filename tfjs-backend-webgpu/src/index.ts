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

import './flags_webgpu';
import './register_all_kernels';

import {env, registerBackend} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from './backend_webgpu';
import * as webgpu from './webgpu';
import {isWebGPUSupported} from './webgpu_util';

if (isWebGPUSupported()) {
  registerBackend('webgpu', async () => {
    // Remove it once we figure out how to correctly read the tensor data
    // before the tensor is disposed in profiling mode.
    env().set('CHECK_COMPUTATION_FOR_ERRORS', false);

    const gpuDescriptor: GPURequestAdapterOptions = {
      powerPreference: env().get('WEBGPU_USE_LOW_POWER_GPU') ?
          'low-power' :
          'high-performance'
    };

    const adapter = await navigator.gpu.requestAdapter(gpuDescriptor);
    const adapterLimits = adapter.limits;
    const deviceDescriptor: GPUDeviceDescriptor = {};
    const supportTimeQuery = adapter.features.has('timestamp-query');
    deviceDescriptor.requiredLimits = {
      'maxComputeWorkgroupStorageSize':
          adapterLimits.maxComputeWorkgroupStorageSize,
      'maxComputeWorkgroupsPerDimension':
          adapterLimits.maxComputeWorkgroupsPerDimension,
    };

    if (supportTimeQuery) {
      deviceDescriptor.requiredFeatures = ['timestamp-query' as const];
    } else {
      console.warn(
          `This device doesn't support timestamp-query extension. ` +
          `Start Chrome browser with flag ` +
          `--disable-dawn-features=disallow_unsafe_apis then try again. ` +
          `Or zero will shown for the kernel time when profiling mode is` +
          `enabled. Using performance.now is not workable for webgpu since` +
          `it doesn't support synchronously to read data from GPU.`);
    }
    const device: GPUDevice = await adapter.requestDevice(deviceDescriptor);
    return new WebGPUBackend(device, supportTimeQuery);
  }, 3 /*priority*/);
}

export {webgpu};
