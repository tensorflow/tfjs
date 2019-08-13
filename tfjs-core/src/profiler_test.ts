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

import {BackendTimer, BackendTimingInfo} from './backends/backend';
import * as tf from './index';
import {describeWithFlags, SYNC_BACKEND_ENVS} from './jasmine_util';
import {Logger, Profiler} from './profiler';
import {Tensor} from './tensor';
import {TypedArray} from './types';

class TestBackendTimer implements BackendTimer {
  private counter = 1;
  constructor(
      private delayMs: number, private queryTimeMs: number,
      private extraInfo: string) {}

  async time(query: () => void): Promise<BackendTimingInfo> {
    query();
    const kernelMs = await new Promise<number>(
        resolve => setTimeout(
            () => resolve(this.queryTimeMs * this.counter++), this.delayMs));
    return {kernelMs, getExtraProfileInfo: () => this.extraInfo};
  }
}

class TestLogger extends Logger {
  logKernelProfile(
      name: string, result: Tensor, vals: TypedArray, timeMs: number) {}
}

describeWithFlags('profiler.Profiler', SYNC_BACKEND_ENVS, () => {
  it('profiles simple function', doneFn => {
    const delayMs = 5;
    const queryTimeMs = 10;
    const extraInfo = '';
    const timer = new TestBackendTimer(delayMs, queryTimeMs, extraInfo);
    const logger = new TestLogger();
    const profiler = new Profiler(timer, logger);

    spyOn(timer, 'time').and.callThrough();
    spyOn(logger, 'logKernelProfile').and.callThrough();

    const timeSpy = timer.time as jasmine.Spy;
    const logKernelProfileSpy = logger.logKernelProfile as jasmine.Spy;

    let kernelCalled = false;
    const result = 1;
    const resultScalar = tf.scalar(result);

    profiler.profileKernel('MatMul', () => {
      kernelCalled = true;
      return resultScalar;
    });

    setTimeout(() => {
      expect(timeSpy.calls.count()).toBe(1);

      expect(logKernelProfileSpy.calls.count()).toBe(1);

      expect(logKernelProfileSpy.calls.first().args).toEqual([
        'MatMul', resultScalar, new Float32Array([result]), queryTimeMs,
        extraInfo
      ]);

      expect(kernelCalled).toBe(true);
      doneFn();
    }, delayMs * 2);
  });

  it('profiles nested kernel', doneFn => {
    const delayMs = 5;
    const queryTimeMs = 10;
    const extraInfo = '';
    const timer = new TestBackendTimer(delayMs, queryTimeMs, extraInfo);
    const logger = new TestLogger();
    const profiler = new Profiler(timer, logger);

    spyOn(timer, 'time').and.callThrough();
    spyOn(logger, 'logKernelProfile').and.callThrough();
    const timeSpy = timer.time as jasmine.Spy;
    const logKernelProfileSpy = logger.logKernelProfile as jasmine.Spy;

    let matmulKernelCalled = false;
    let maxKernelCalled = false;
    const result = 1;
    const resultScalar = tf.scalar(result);

    profiler.profileKernel('MatMul', () => {
      const result = profiler.profileKernel('Max', () => {
        maxKernelCalled = true;
        return resultScalar;
      });
      matmulKernelCalled = true;
      return result;
    });

    setTimeout(() => {
      expect(timeSpy.calls.count()).toBe(2);

      expect(logKernelProfileSpy.calls.count()).toBe(2);
      expect(logKernelProfileSpy.calls.first().args).toEqual([
        'Max', resultScalar, new Float32Array([result]), queryTimeMs, extraInfo
      ]);
      expect(logKernelProfileSpy.calls.argsFor(1)).toEqual([
        'MatMul', resultScalar, new Float32Array([result]), queryTimeMs * 2,
        extraInfo
      ]);

      expect(matmulKernelCalled).toBe(true);
      expect(maxKernelCalled).toBe(true);
      doneFn();
    }, delayMs * 2);
  });
});
