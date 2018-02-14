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
import {ScopeFn, ScopeResult, ScopeResultImmediate} from './tape_util';
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
    if (fn == null) {
      // Called with only 1 argument.
      if (typeof nameOrFn !== 'function') {
        throw new Error('Please provide a function to dl.tidy()');
      }
      fn = nameOrFn;
      nameOrFn = '';
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
      // TODO(nsthorat,smilkov): Do operation logging and performance profiling.
    }
    ENV.engine.startScope(gradMode);

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
   * Keeps a Tensor generated inside a `tidy` from being disposed
   * automatically.
   * @param result The Tensor to keep from being disposed.
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
   * @param f The function to execute and time.
   */
  @doc({heading: 'Performance', subheading: 'Timing'})
  static time(f: () => void): Promise<TimingInfo> {
    return ENV.engine.time(f);
  }
}
