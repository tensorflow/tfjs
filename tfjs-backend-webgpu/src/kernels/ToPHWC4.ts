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

import {TensorInfo} from '@tensorflow/tfjs-core';
import {WebGPUBackend} from '../backend_webgpu';
import {nhwcTophwc4Program} from '../nhwcTophwc4_webgpu';
import {phwc4TonhwcProgram} from '../phwc4Tonhwc_webgpu';

export function toPHWC4(x: TensorInfo, backend: WebGPUBackend) {
  const program = new nhwcTophwc4Program(x.shape);
  const uniformData = [{
    type: 'int32',
    data: [x.shape[2], x.shape[1], Math.ceil(x.shape[3] / 4), x.shape[3]]
  }];
  return backend.runWebGPUProgram(
      program, [x], x.dtype, uniformData, null, true, false);
}

export function toHWC(x: TensorInfo, backend: WebGPUBackend) {
  const program = new phwc4TonhwcProgram(x.shape);
  const uniformData =
      [{type: 'int32', data: [x.shape[2], x.shape[1], x.shape[3], 0]}];
  return backend.runWebGPUProgram(
      program, [x], x.dtype, uniformData, null, false, true);
}
