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

import {doc} from './doc';
import {TimingInfo} from './engine';
import {ENV} from './environment';
// tslint:disable-next-line:max-line-length
import {ScopeFn, ScopeResult, ScopeResultImmediate} from './tape';
import {Tensor} from './tensor';

export class Tracking {
  /**
   * Executes the provided function and after it is executed, cleans up all
   * intermediate tensors allocated by the function except those returned by
   * the function.
   *
   * Using this method helps avoid memory leaks. In general, wrap calls to
   * operations in `tidy` for automatic memory cleanup.
   *
   * When in safe mode, you must enclose all `Tensor` creation and ops
   * inside a `tidy` to prevent memory leaks.
   *
   * ```js
   * // y = 2 ^ 2 + 1
   * const y = dl.tidy(() => {
   *   // a, b, and one will be cleaned up when the tidy ends.
   *   const one = dl.scalar(1);
   *   const a = dl.scalar(2);
   *   const b = a.square();
   *
   *   console.log('numTensors (in tidy): ' + dl.memory().numTensors);
   *
   *   // The value returned inside the tidy function will return
   *   // through the tidy, in this case to the variable y.
   *   return b.add(one);
   * });
   *
   * console.log('numTensors (outside tidy): ' + dl.memory().numTensors);
   * y.print();
   * ```
   *
   * @param nameOrFn The name of the closure, or the function to execute.
   *     If a name is provided, the 2nd argument should be the function.
   *     If a name is provided, and debug mode is on, the timing and the memory
   *     usage of the function will be tracked and displayed on the console
   *     using the provided name.
   * @param fn The function to execute.
   * @param gradMode If true, starts a tape and doesn't dispose tensors.
   */
  @doc({heading: 'Performance', subheading: 'Memory'})
  static tidy<T extends ScopeResult>(
      nameOrFn: string|ScopeFn<T>, fn?: ScopeFn<T>, gradMode = false): T {
    let name = null;
    if (fn == null) {
      // Called with only 1 argument.
      if (typeof nameOrFn !== 'function') {
        throw new Error('Please provide a function to dl.tidy()');
      }
      fn = nameOrFn;
    } else {
      // Called with 2 arguments.
      if (typeof nameOrFn !== 'string' && !(nameOrFn instanceof String)) {
        throw new Error(
            'When calling with two arguments, the first argument ' +
            'to dl.tidy() must be a string');
      }
      if (typeof fn !== 'function') {
        throw new Error(
            'When calling with two arguments, the 2nd argument ' +
            'to dl.tidy() must be a function');
      }
      name = nameOrFn as string;
      // TODO(nsthorat,smilkov): Do operation logging and performance profiling.
    }
    ENV.engine.startScope(name, gradMode);

    const result = fn();
    if (result instanceof Promise) {
      result.then(r => ENV.engine.endScope(r, gradMode));
      return result;
    } else {
      ENV.engine.endScope(result as ScopeResultImmediate, gradMode);
      return result;
    }
  }

  /**
   * Keeps a `Tensor` generated inside a `tidy` from being disposed
   * automatically.
   *
   * ```js
   * let b;
   * const y = dl.tidy(() => {
   *   const one = dl.scalar(1);
   *   const a = dl.scalar(2);
   *
   *   // b will not be cleaned up by the tidy. a and one will be cleaned up
   *   // when the tidy ends.
   *   b = dl.keep(a.square());
   *
   *   console.log('numTensors (in tidy): ' + dl.memory().numTensors);
   *
   *   // The value returned inside the tidy function will return
   *   // through the tidy, in this case to the variable y.
   *   return b.add(one);
   * });
   *
   * console.log('numTensors (outside tidy): ' + dl.memory().numTensors);
   * console.log('y:');
   * y.print();
   * console.log('b:');
   * b.print();
   * ```
   *
   * @param result The tensor to keep from being disposed.
   */
  @doc({heading: 'Performance', subheading: 'Memory'})
  static keep<T extends Tensor>(result: T): T {
    return ENV.engine.keep(result);
  }

  /**
   * Executes `f()` and returns a promise that resolves with timing information.
   *
   * The result is an object with the following properties:
   *
   * - `wallMs`: wall execution time.
   * - `kernelMs`: kernel execution time, ignoring data transfer.
   * - On `WebGL` the following additional properties exist:
   *   - `uploadWaitMs`: cpu blocking time on texture uploads.
   *   - `downloadWaitMs`: cpu blocking time on texture downloads (readPixels).
   *
   * ```js
   * const x = dl.randomNormal([20, 20]);
   * const time = await dl.time(() => x.matMul(x));
   *
   * console.log(`kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`);
   * ```
   *
   * @param f The function to execute and time.
   */
  @doc({heading: 'Performance', subheading: 'Timing'})
  static time(f: () => void): Promise<TimingInfo> {
    return ENV.engine.time(f);
  }
}
