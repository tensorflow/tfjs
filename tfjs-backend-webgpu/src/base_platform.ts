/**
 * @license
 * Copyright 2023 Google Inc.
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
import {device_util} from '@tensorflow/tfjs-core';
// @ts-ignore
import nodeGPUBinding from 'bindings';

let nodeGPU: GPU = null;
function getNodeGPU() {
  if (nodeGPU) {
    return nodeGPU;
  }
  const gpuProviderModule = nodeGPUBinding('dawn');
  const gpuProviderFlags = ['disable-dawn-features=disallow_unsafe_apis'];
  nodeGPU = gpuProviderModule.create(gpuProviderFlags);
  return nodeGPU;
}

export async function requestAdapter(gpuDescriptor: GPURequestAdapterOptions):
    Promise<GPUAdapter> {
  return device_util.isBrowser() ? navigator.gpu.requestAdapter(gpuDescriptor) :
                                   getNodeGPU().requestAdapter(gpuDescriptor);
}
