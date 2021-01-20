/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {KernelConfig, MirrorPad, MirrorPadAttrs, MirrorPadInputs} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';

import {MirrorPadProgram} from './mirror_pad_webgpu';

export const mirrorPadConfig: KernelConfig = {
  kernelName: MirrorPad,
  backendName: 'webgpu',
  kernelFunc: ({inputs, attrs, backend}) => {
    const {x} = inputs as MirrorPadInputs;
    const {paddings, mode} = attrs as unknown as MirrorPadAttrs;
    const webGPUBackend = backend as WebGPUBackend;

    const program = new MirrorPadProgram(x.shape, paddings, mode);
    const output = webGPUBackend.runWebGPUProgram(program, [x], x.dtype);

    return output;
  }
};
