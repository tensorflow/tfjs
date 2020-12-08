/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {BackendTimer} from './backends/backend';
import {env} from './environment';
import {Tensor} from './tensor';
import {NamedTensorMap} from './tensor_types';
import {DataType, DataTypeMap, TypedArray} from './types';
import * as util from './util';

export type KernelProfile = {
  kernelName: string,
  outputs: Tensor[],
  inputs: NamedTensorMap,
  timeMs: Promise<number|{error: string}>,
  extraInfo: Promise<string>
};

export class Profiler {
  constructor(private backendTimer: BackendTimer, private logger?: Logger) {
    if (logger == null) {
      this.logger = new Logger();
    }
  }

  profileKernel(kernelName: string, inputs: NamedTensorMap, f: () => Tensor[]):
      KernelProfile {
    let outputs: Tensor[];
    const holdResultWrapperFn = () => {
      outputs = f();
    };
    const timer = this.backendTimer.time(holdResultWrapperFn);

    if (env().getBool('CHECK_COMPUTATION_FOR_ERRORS')) {
      for (let i = 0; i < outputs.length; i++) {
        const output = outputs[i];
        // Dangling promise here because we don't want to propagate up
        // asynchronicity.
        output.data().then(tensorVals => {
          checkComputationForErrors(tensorVals, output.dtype, kernelName);
        });
      }
    }

    const kernelProfile = {
      kernelName,
      outputs,
      inputs,
      timeMs: timer.then(timing => timing.kernelMs),
      extraInfo: timer.then(
          timing => timing.getExtraProfileInfo != null ?
              timing.getExtraProfileInfo() :
              '')
    };
    return kernelProfile;
  }

  logKernelProfile(kernelProfile: KernelProfile): void {
    const {kernelName, outputs, timeMs, inputs, extraInfo} = kernelProfile;

    outputs.forEach(result => {
      Promise.all([result.data(), timeMs, extraInfo]).then(valueContainer => {
        this.logger.logKernelProfile(
            kernelName, result, valueContainer[0], valueContainer[1], inputs,
            valueContainer[2]);
      });
    });
  }
}

export function checkComputationForErrors<D extends DataType>(
    vals: DataTypeMap[D], dtype: D, kernelName: string): boolean {
  if (dtype !== 'float32') {
    // Only floating point computations will generate NaN values
    return false;
  }
  for (let i = 0; i < vals.length; i++) {
    const num = vals[i] as number;
    if (isNaN(num) || !isFinite(num)) {
      // Throwing custom exception so behavior is testable.
      console.warn(`Found ${num} in the result of '${kernelName}'`);
      return true;
    }
  }
  return false;
}

export class Logger {
  logKernelProfile(
      name: string, result: Tensor, vals: TypedArray,
      timeMs: number|{error: string}, inputs: NamedTensorMap,
      extraInfo?: string) {
    const time = typeof timeMs === 'number' ? util.rightPad(`${timeMs}ms`, 9) :
                                              timeMs['error'];
    const paddedName = util.rightPad(name, 25);
    const rank = result.rank;
    const size = result.size;
    const shape = util.rightPad(result.shape.toString(), 14);
    let inputShapesDescription = '';

    for (const name in inputs) {
      const input = inputs[name];
      if (input != null) {
        // The input might be a non-tensor (e.g HTMLImageElement), in which case
        // we claim the output shape as input shape.
        const inputShape = input.shape || result.shape;
        const inputRank = inputShape.length;
        inputShapesDescription +=
            `${name}: ${inputRank}D ${inputRank > 0 ? inputShape : ''} `;
      }
    }

    console.log(
        `%c${paddedName}\t%c${time}\t%c${rank}D ${shape}\t%c${size}\t%c${
            inputShapesDescription}\t%c${extraInfo}`,
        'font-weight:bold', 'color:red', 'color:blue', 'color: orange',
        'color: green', 'color: steelblue');
  }
}
