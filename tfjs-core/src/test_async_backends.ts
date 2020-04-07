#!/usr/bin/env node
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
/**
 * This file tests that we don't have any dataSyncs in the unconstrainted tests
 * so that we can run backends that have async init and async data reads against
 * our exported test files.
 */

// Use require here to workaround this being a circular dependency.
// This should only be done in tests.
// tslint:disable-next-line: no-require-imports
require('@tensorflow/tfjs-backend-cpu');
import './index';
import {setTestEnvs} from './jasmine_util';
import {registerBackend, engine} from './globals';
import {KernelBackend} from './backends/backend';
import {getKernelsForBackend, registerKernel} from './kernel_registry';

// tslint:disable-next-line:no-require-imports
const jasmine = require('jasmine');

process.on('unhandledRejection', e => {
  throw e;
});

class AsyncCPUBackend extends KernelBackend {}
const asyncBackend = new AsyncCPUBackend();

// backend is cast as any so that we can access methods through bracket
// notation.
const backend: KernelBackend = engine().findBackend('cpu');
const proxyBackend = new Proxy(asyncBackend, {
  get(target, name, receiver) {
    if (name === 'readSync') {
      throw new Error(
          `Found dataSync() in a unit test. This is disabled so unit tests ` +
          `can run in backends that only support async data. Please use ` +
          `.data() in unit tests or if you truly are testing dataSync(), ` +
          `constrain your test with SYNC_BACKEND_ENVS`);
    }
    //@ts-ignore;
    const origSymbol = backend[name];
    if (typeof origSymbol === 'function') {
      // tslint:disable-next-line:no-any
      return (...args: any[]) => {
        return origSymbol.apply(backend, args);
      };
    } else {
      return origSymbol;
    }
  }
});

const proxyBackendName = 'test-async-cpu';

// The registration is async on purpose, so we know our testing infra works
// with backends that have async init (e.g. WASM and WebGPU).
registerBackend(proxyBackendName, async () => proxyBackend);

// All the kernels are registered under the 'cpu' name, so we need to
// register them also under the proxy backend name.
const kernels = getKernelsForBackend('cpu');
kernels.forEach(({kernelName, kernelFunc, setupFunc}) => {
  registerKernel(
      {kernelName, backendName: proxyBackendName, kernelFunc, setupFunc});
});

setTestEnvs([{
  name: proxyBackendName,
  backendName: proxyBackendName,
  isDataSync: false,
}]);

const runner = new jasmine();

runner.loadConfig({spec_files: ['dist/**/**_test.js'], random: false});
runner.execute();
