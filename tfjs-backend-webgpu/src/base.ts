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
    // Remove it once we figure out how to correctly read the tensor data
    // before the tensor is disposed in profiling mode.
    env().set('CHECK_COMPUTATION_FOR_ERRORS', false);

    const gpuDescriptor: GPURequestAdapterOptions = {
      powerPreference: env().get('WEBGPU_USE_LOW_POWER_GPU') ?
          'low-power' :
          'high-performance'
    };

    const adapter = await navigator.gpu.requestAdapter(gpuDescriptor);

    console.log('kernel log: adapter ********************************');
    console.log(adapter);
    console.log('stringify: ' + JSON.stringify(adapter));
    console.log('log: adapter ******************************** end');

    console.error('kernel error: ********************************');
    console.error(adapter);
    console.error('stringify: ' + JSON.stringify(adapter));
    console.error('error: adapter ******************************** end');

    const deviceDescriptor: GPUDeviceDescriptor = {};

    // Note that timestamp-query-inside-passes is not formally in spec as
    // timestamp within a pass is not generally supported on all the platforms.
    // More details can be found at
    // https://github.com/gpuweb/gpuweb/blob/main/proposals/timestamp-query-inside-passes.md
    if (adapter.features.has('timestamp-query-inside-passes')) {
      deviceDescriptor.requiredFeatures =
          // tslint:disable-next-line:no-any
          ['timestamp-query-inside-passes' as any];
    }

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
    const adapterInfo = await adapter.requestAdapterInfo();
    console.log(
        'kernel log: adapterInfo kernel********************************');
    console.log(adapterInfo);
    console.log('adapterInfo stringify: ' + JSON.stringify(adapterInfo));
    console.log(adapterInfo.vendor);
    console.log(adapterInfo.device);
    console.log(adapterInfo.architecture);
    console.log('log: adapterInfo kernel******************************** end');

    console.error(
        'kernel error: adapterInfo kernel********************************');
    console.error(adapterInfo);
    console.error('adapterInfo stringify: ' + JSON.stringify(adapterInfo));
    console.error(adapterInfo.vendor);
    console.error(adapterInfo.device);
    console.error(adapterInfo.architecture);
    console.error(
        'error: adapterInfo kernel******************************** end');

    return new WebGPUBackend(device, adapterInfo);
  }, 3 /*priority*/);
}

// Export webgpu utilities
export * from './webgpu';
