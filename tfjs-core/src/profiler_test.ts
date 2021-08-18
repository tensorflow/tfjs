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

import {BackendTimer, BackendTimingInfo} from './backends/backend';
import * as tf from './index';
import {ALL_ENVS, describeWithFlags, SYNC_BACKEND_ENVS} from './jasmine_util';
import {checkComputationForErrors, KernelProfile, Logger, Profiler} from './profiler';
import {Tensor} from './tensor';
import {NamedTensorMap} from './tensor_types';
import {TypedArray} from './types';

class TestBackendTimer implements BackendTimer {
  private counter = 1;
  constructor(
      private delayMs: number, private queryTimeMs: number,
      private extraInfo: string) {}

  timerAvailable() {
    return true;
  }

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

function promiseCheckWrapper(acturalValPromise: Promise<{}>, truthVal: {}) {
  return acturalValPromise.then(acturalVal => {
    expect(acturalVal).toEqual(truthVal);
  });
}

function checkKernelProfile(acturalVal: KernelProfile, truthVal: {
  kernelName: string,
  outputs: Tensor[],
  timeMs: number|{error: string},
  inputs: NamedTensorMap,
  extraInfo: string
}) {
  expect(acturalVal.kernelName).toBe(truthVal.kernelName);
  expect(acturalVal.inputs).toBe(truthVal.inputs);
  acturalVal.outputs.forEach((output, index) => {
    expect(output).toBe(truthVal.outputs[index]);
  });

  const promiseContainer = [
    promiseCheckWrapper(acturalVal.timeMs, truthVal.timeMs),
    promiseCheckWrapper(acturalVal.extraInfo, truthVal.extraInfo),
  ];
  return Promise.all(promiseContainer);
}

describeWithFlags('profiler.Profiler', SYNC_BACKEND_ENVS, () => {
  it('profiles simple function', doneFn => {
    const delayMs = 5;
    const queryTimeMs = 10;
    const inputs = {'x': tf.tensor1d([1])};
    const extraInfo = '';
    const timer = new TestBackendTimer(delayMs, queryTimeMs, extraInfo);
    const logger = new TestLogger();
    const profiler = new Profiler(timer, logger);

    spyOn(timer, 'time').and.callThrough();
    spyOn(logger, 'logKernelProfile').and.callThrough();

    const timeSpy = timer.time as jasmine.Spy;

    let kernelCalled = false;
    const result = 1;
    const resultScalar = tf.scalar(result);

    const kernelProfile = profiler.profileKernel('MatMul', inputs, () => {
      kernelCalled = true;
      return [resultScalar];
    });
    setTimeout(() => {
      expect(timeSpy.calls.count()).toBe(1);
      expect(kernelCalled).toBe(true);

      checkKernelProfile(kernelProfile, {
        kernelName: 'MatMul',
        outputs: [resultScalar],
        timeMs: queryTimeMs,
        inputs,
        extraInfo,
      }).then(() => doneFn());
    }, delayMs * 2);
  });

  it('profiles nested kernel with optional inputs', doneFn => {
    const delayMs = 5;
    const queryTimeMs = 10;
    const inputs: {'x': tf.Tensor,
                   'bias': null} = {'x': tf.tensor1d([1]), 'bias': null};
    const extraInfo = '';
    const timer = new TestBackendTimer(delayMs, queryTimeMs, extraInfo);
    const logger = new TestLogger();
    const profiler = new Profiler(timer, logger);

    spyOn(timer, 'time').and.callThrough();
    spyOn(logger, 'logKernelProfile').and.callThrough();
    const timeSpy = timer.time as jasmine.Spy;

    let matmulKernelCalled = false;
    let maxKernelCalled = false;
    const result = 1;
    const resultScalar = tf.scalar(result);

    let innerKernelProfile: KernelProfile;
    const outerKernelProfile = profiler.profileKernel('MatMul', inputs, () => {
      innerKernelProfile = profiler.profileKernel('Max', inputs, () => {
        maxKernelCalled = true;
        return [resultScalar];
      });
      matmulKernelCalled = true;
      return innerKernelProfile.outputs;
    });

    setTimeout(() => {
      expect(timeSpy.calls.count()).toBe(2);
      expect(matmulKernelCalled).toBe(true);
      expect(maxKernelCalled).toBe(true);

      const checkInnerKernelProfile = checkKernelProfile(innerKernelProfile, {
        kernelName: 'Max',
        outputs: [resultScalar],
        timeMs: queryTimeMs,
        inputs,
        extraInfo
      });
      const checkOuterKernelProfile = checkKernelProfile(outerKernelProfile, {
        kernelName: 'MatMul',
        outputs: [resultScalar],
        timeMs: queryTimeMs * 2,
        inputs,
        extraInfo
      });
      Promise.all([checkInnerKernelProfile, checkOuterKernelProfile])
          .then(() => doneFn());
    }, delayMs * 2);
  });

  it('log kernelProfile', doneFn => {
    const delayMs = 5;
    const queryTimeMs = 10;
    const inputs = {'x': tf.tensor1d([1])};
    const extraInfo = '';
    const timer = new TestBackendTimer(delayMs, queryTimeMs, extraInfo);
    const logger = new TestLogger();
    const profiler = new Profiler(timer, logger);

    spyOn(logger, 'logKernelProfile').and.callThrough();
    const logKernelProfileSpy = logger.logKernelProfile as jasmine.Spy;

    const result = 1;
    const resultScalar = tf.scalar(result);

    const kernelProfiles = profiler.profileKernel('MatMul', inputs, () => {
      return [resultScalar];
    });
    profiler.logKernelProfile(kernelProfiles);

    setTimeout(() => {
      expect(logKernelProfileSpy.calls.count()).toBe(1);

      expect(logKernelProfileSpy.calls.first().args).toEqual([
        'MatMul', resultScalar, new Float32Array([result]), queryTimeMs, inputs,
        extraInfo
      ]);
      doneFn();
    }, delayMs * 2);
  });
});

describe('profiler.checkComputationForErrors', () => {
  beforeAll(() => {
    // Silence warnings.
    spyOn(console, 'warn');
  });

  it('Float32Array has NaN', () => {
    expect(checkComputationForErrors(
               new Float32Array([1, 2, 3, NaN, 4, 255]), 'float32', 'test'))
        .toBe(true);
  });

  it('Float32Array has Infinity', () => {
    expect(
        checkComputationForErrors(
            new Float32Array([1, 2, 3, Infinity, 4, 255]), 'float32', 'test'))
        .toBe(true);
  });

  it('Float32Array no NaN', () => {
    // Int32 and Bool NaNs should not trigger an error.
    expect(checkComputationForErrors(
               new Float32Array([1, 2, 3, -1, 4, 255]), 'float32', 'test'))
        .toBe(false);
  });
});

describeWithFlags('profiler.Logger', ALL_ENVS, () => {
  it('skips logging for undefined input node in input tensor map', () => {
    const kernelName = 'FusedConv2D';
    const vals = new Float32Array(1);
    const outputs = tf.tensor1d([1]);
    const timeMs = 10;
    const inputs: NamedTensorMap = {
      'x': tf.tensor1d([1]),
      'filter': tf.tensor1d([1]),
      'bias': tf.tensor1d([1]),
      'preluActivationWeights': undefined
    };
    const extraInfo = '';
    const logger = new Logger();
    spyOn(console, 'log');
    const consoleLogSpy = console.log as jasmine.Spy;

    logger.logKernelProfile(
        kernelName, outputs, vals, timeMs, inputs, extraInfo);

    expect(consoleLogSpy.calls.first().args)
        .not.toContain('preluActivationWeights');
  });
});
