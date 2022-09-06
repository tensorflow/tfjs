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

import {KernelConfig, Range, RangeAttrs, scalar, zeros} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const rangeConfig: KernelConfig = {
  kernelName: Range,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const backend = args.backend as NodeJSKernelBackend;
    const {start, stop, dtype} = args.attrs as {} as RangeAttrs;
    let {step} = args.attrs as {} as RangeAttrs;

    // TensorFlow.js specific allowances
    const sameStartStop = start === stop;
    const increasingRangeNegativeStep = start < stop && step < 0;
    const decreasingRangePositiveStep = stop < start && step > 1;

    if (sameStartStop || increasingRangeNegativeStep ||
        decreasingRangePositiveStep) {
      return zeros([0], dtype);
    }

    if (stop < start && step === 1) {
      // Auto adjust the step's sign if it hasn't been set
      // (or was set to 1)
      step = -1;
    }

    const opAttrs = [createTensorsTypeOpAttr('Tidx', dtype)];
    const startTensor = scalar(start);
    const stopTensor = scalar(stop);
    const stepTensor = scalar(step);
    const res = backend.executeSingleOutput(
        Range, opAttrs, [startTensor, stopTensor, stepTensor]);
    const castedRes = res.cast(dtype);

    startTensor.dispose();
    stopTensor.dispose();
    stepTensor.dispose();
    res.dispose();

    return castedRes;
  }
};
