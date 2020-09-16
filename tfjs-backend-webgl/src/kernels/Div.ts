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

import {Div} from '@tensorflow/tfjs-core';
import {KernelConfig} from '@tensorflow/tfjs-core';
import * as binaryop_gpu from '../binaryop_gpu';
import * as binaryop_packed_gpu from '../binaryop_packed_gpu';
import {binaryKernelFunc} from '../kernel_utils/kernel_funcs_utils';

export const divKernelFunc =
    binaryKernelFunc(binaryop_gpu.DIV, binaryop_packed_gpu.DIV, true);

export const divConfig: KernelConfig = {
  kernelName: Div,
  backendName: 'webgl',
  kernelFunc: divKernelFunc,
};
