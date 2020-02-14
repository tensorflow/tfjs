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

import {NamedTensorInfoMap, registerKernel, TensorInfo} from '../../../kernel_registry';
import {sizeFromShape} from '../../../util';
import {MathBackendCPU} from '../backend_cpu';

import {exp} from './exp_impl';

interface ExpInputs extends NamedTensorInfoMap {
  x: TensorInfo;
}

registerKernel({
  kernelName: 'Exp',
  backendName: 'cpu',
  kernelFunc: ({inputs, backend}) => {
    const {x} = inputs as ExpInputs;
    const cpuBackend = backend as MathBackendCPU;

    const xVals = cpuBackend.data.get(x.dataId).values as Float32Array;

    const result = exp(xVals, new Float32Array(sizeFromShape(x.shape)));

    const dataId = cpuBackend.write(result, x.shape, x.dtype);
    return {dataId, shape: x.shape, dtype: x.dtype};
  }
});
