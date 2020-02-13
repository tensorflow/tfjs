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

// import {max} from './kernel_names';
// import {ENGINE} from '../../engine';
// import {NamedAttrMap, NamedTensorInfoMap, registerKernel, TensorInfo} from
// '../../kernel_registry';
import {KernelFunc, NamedAttrMap, NamedTensorInfoMap, registerKernel, TensorInfo} from '../../kernel_registry';
import {parseAxisParam} from '../../util';

import {MathBackendCPU} from './backend_cpu';
import {assertNotComplex} from './cpu_util';
import {max} from './max';

interface SoftmaxInputs extends NamedTensorInfoMap {
  logits: TensorInfo;
}

interface SoftmaxAttrs extends NamedAttrMap {
  dim: number;
}

/**
 * max_impl exports max kernel func
 * max.ts uses register kernel, it imports max_impl
 * softmax imports max_impl
 */

export const softmax: KernelFunc = ({inputs, attrs, backend}) => {
  const {logits} = inputs as SoftmaxInputs;
  const {dim} = attrs as SoftmaxAttrs;

  const cpuBackend = backend as MathBackendCPU;
  assertNotComplex(logits, 'Softmax');
  console.log('SOFTMAXJasf sdf');
  console.log(logits);

  const axes = parseAxisParam([dim], logits.shape);
  const maxLogit = max(logits, axes);
  console.log(maxLogit);
  // const maxLogit = ENGINE.runKernel();

  const values = cpuBackend.data.get(logits.dataId).values as Float32Array;
  const newValues = new Float32Array(values.length);
  for (let i = 0; i < values.length; ++i) {
    const value = values[i];
    newValues[i] = value;
  }

  const dataId = cpuBackend.write(newValues, logits.shape, logits.dtype);
  return {dataId, shape: logits.shape, dtype: 'float32'};
};

registerKernel(
    {kernelName: 'Softmax', backendName: 'cpu', kernelFunc: softmax});
