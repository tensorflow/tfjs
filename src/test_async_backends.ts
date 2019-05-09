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
 * so that we can run async backends against our exported test files.
 */
import './index';

import {setTestEnvs} from './jasmine_util';
import {MathBackendCPU} from './backends/cpu/backend_cpu';
import {registerBackend} from './globals';
import {KernelBackend} from './backends/backend';

// tslint:disable-next-line:no-require-imports
const jasmine = require('jasmine');

process.on('unhandledRejection', e => {
  throw e;
});

class AsyncCPUBackend extends KernelBackend {}
const asyncBackend = new AsyncCPUBackend();

// backend is cast as any so that we can access methods through bracket
// notation.
// tslint:disable-next-line:no-any
const backend: any = new MathBackendCPU();
const proxyBackend = new Proxy(asyncBackend, {
  get(target, name, receiver) {
    if (name === 'readSync') {
      throw new Error(
          `Found dataSync() in a unit test. This is disabled so unit tests ` +
          `can run in backends that only support async data. Please use ` +
          `.data() in unit tests or if you truly are testing dataSync(), ` +
          `constrain your test with SYNC_BACKEND_ENVS`);
    }
    if (name === 'isDataSync') {
      return () => false;
    }
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

registerBackend('test-async-cpu', () => proxyBackend);

setTestEnvs([{name: 'test-async-cpu', backendName: 'test-async-cpu'}]);

const runner = new jasmine();
runner.loadConfig({spec_files: ['dist/**/**_test.js'], random: false});
runner.execute();
