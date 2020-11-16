/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {KernelConfig, Step, StepAttrs} from '@tensorflow/tfjs-core';

import {unaryKernelFunc} from '../utils/unary_utils';

export const step = unaryKernelFunc(Step, (xi, attrs) => {
  const stepAttrs = attrs as {} as StepAttrs;
  if (isNaN(xi)) {
    return NaN;
  } else {
    return xi > 0 ? 1 : stepAttrs.alpha;
  }
});

export const stepConfig: KernelConfig = {
  kernelName: Step,
  backendName: 'cpu',
  kernelFunc: step,
};
