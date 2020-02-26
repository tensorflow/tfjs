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

import {SquaredDifference, SquaredDifferenceInputs} from '../../../kernel_names';
import {KernelConfig} from '../../../kernel_registry';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';
import {broadcastedBinaryOp} from '../utils/kernel_utils';

export const squaredDifferenceConfig: KernelConfig = {
  kernelName: SquaredDifference,
  backendName: 'cpu',
  kernelFunc: ({inputs, backend}) => {
    const {a, b} = inputs as SquaredDifferenceInputs;
    const cpuBackend = backend as MathBackendCPU;
    assertNotComplex([a, b], SquaredDifference);

    const resultData =
        broadcastedBinaryOp(a, b, a.dtype, cpuBackend, (aVal, bVal) => {
          const diff = aVal - bVal;
          return diff * diff;
        });

    return resultData;
  }
};
