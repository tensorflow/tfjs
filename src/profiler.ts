/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import * as util from './util';
import {Tensor} from './tensor';
import {TypedArray} from './types';

import {BackendTimer} from './kernels/backend';
import {Kernel} from './kernels/kernel_registry';

export class Profiler {
  constructor(private backendTimer: BackendTimer, private logger?: Logger) {
    if (logger == null) {
      this.logger = new Logger();
    }
  }

  profileKernel<T extends Tensor>(kernelName: Kernel, f: () => T): T {
    let result: T;
    const holdResultWrapperFn = () => {
      result = f();
    };
    const timer = this.backendTimer.time(holdResultWrapperFn);

    const vals = result.dataSync();
    util.checkForNaN(vals, result.dtype, kernelName);

    timer.then((timeMs: number) => {
      this.logger.logKernelProfile(kernelName, result, vals, timeMs);
    });

    return result as T;
  }
}

export class Logger {
  logKernelProfile(
      kernelName: Kernel, result: Tensor, vals: TypedArray, timeMs: number) {
    const time = util.rightPad(`${timeMs}ms`, 9);
    const paddedName = util.rightPad(kernelName, 25);
    const rank = result.rank;
    const size = result.size;
    const shape = util.rightPad(result.shape.toString(), 14);
    console.log(
        `%c${paddedName}\t%c${time}\t%c${rank}D ${shape}\t%c${size}`,
        'font-weight:bold', 'color:red', 'color:blue', 'color: orange');
  }
}
