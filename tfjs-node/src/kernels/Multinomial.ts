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

import {KernelConfig, Multinomial, MultinomialAttrs, MultinomialInputs, scalar} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const multinomialConfig: KernelConfig = {
  kernelName: Multinomial,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {logits} = args.inputs as MultinomialInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {numSamples, seed, normalized} = args.attrs as {} as MultinomialAttrs;

    if (normalized) {
      throw new Error(
          'TF Node backend does not support normalized logits ' +
          'passed to multinomial');
    }
    const opAttrs = [
      createTensorsTypeOpAttr('T', logits.dtype),
      createTensorsTypeOpAttr('output_dtype', 'int32'),
      {name: 'seed', type: backend.binding.TF_ATTR_INT, value: seed},
      {name: 'seed2', type: backend.binding.TF_ATTR_INT, value: seed * seed},
    ];
    const numSamplesTensor = scalar(numSamples, 'int32');
    const res = backend.executeSingleOutput(
        Multinomial, opAttrs, [logits, numSamplesTensor]);
    numSamplesTensor.dispose();
    return res;
  }
};
