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

import {backend_util, fill, greater, isNaN as tfIsNan, KernelConfig, ones, scalar, Step, StepAttrs, StepInputs, Tensor, tidy, where} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const stepConfig: KernelConfig = {
  kernelName: Step,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {x} = args.inputs as StepInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {alpha} = args.attrs as unknown as StepAttrs;

    const dtype = x.dtype;
    return tidy(() => {
      const nans = tfIsNan(x as Tensor);
      const stepNoNans = where(
          greater(x as Tensor, scalar(0, dtype)), ones(x.shape),
          fill(x.shape, alpha, dtype));

      const opAttrs = [createTensorsTypeOpAttr(
          'T', backend_util.upcastType(x.dtype, stepNoNans.dtype))];
      return backend.executeSingleOutput(
          'Select', opAttrs, [nans, x, stepNoNans]);
    });
  }
};
