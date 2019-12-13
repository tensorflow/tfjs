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

import {NamedTensorInfoMap, registerKernel, TensorInfo} from '../../kernel_registry';

import {MathBackendCPU} from './backend_cpu';

interface SoftmaxInputs extends NamedTensorInfoMap {
  logits: TensorInfo;
}

registerKernel({
  kernelName: 'Softmax',
  backendName: 'cpu',
  kernelFunc: (args: {backend: MathBackendCPU, inputs: SoftmaxInputs}) => {
    const {backend, inputs} = args;
    const {logits} = inputs;

    const values = backend.data.get(logits.dataId).values as Float32Array;
    // const newValues = new Float32Array(values.length);
    // for (let i = 0; i < values.length; ++i) {
    //   const value = values[i];
    //   newValues[i] = value * value;
    // }
    const keepDims = true;
    const lse = logits.logSumExp([dim], keepDims);
    const logResult = logits.toFloat().sub(lse);
    const y = logResult.exp() as T;
    const dataId = backend.write(newValues, x.shape, x.dtype);
    return {dataId, shape: x.shape, dtype: x.dtype};
  }
});
