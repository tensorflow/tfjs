/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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

import * as tf from '@tensorflow/tfjs-core';
import {registerBackend, removeBackend, test_util} from '@tensorflow/tfjs-core';
// tslint:disable-next-line:no-imports-from-dist
import {ALL_ENVS, BROWSER_ENVS, describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {init, resetWasmPath} from './backend_wasm';
import {BackendWasm, setWasmPath} from './index';

/**
 * Tests specific to the wasm backend. The name of these tests must start with
 * 'wasm' so that they are always included in the test runner. See
 * `env.specFilter` in `setup_test.ts` for details.
 */
describeWithFlags('wasm read/write', ALL_ENVS, () => {
  it('write and read values', async () => {
    const x = tf.tensor1d([1, 2, 3]);
    test_util.expectArraysClose([1, 2, 3], await x.data());
  });

  it('allocate repetitively and confirm reuse of heap space', () => {
    const backend = tf.backend() as BackendWasm;
    const size = 100;
    // Allocate for the first time, record the memory offset and dispose.
    const t1 = tf.zeros([size]);
    const memOffset1 = backend.getMemoryOffset(t1.dataId);
    t1.dispose();

    // Allocate again and make sure the offset is the same (memory was reused).
    const t2 = tf.zeros([size]);
    const memOffset2 = backend.getMemoryOffset(t2.dataId);
    // This should fail in case of a memory leak.
    expect(memOffset1).toBe(memOffset2);
  });
});

describeWithFlags('wasm init', BROWSER_ENVS, () => {
  beforeEach(() => {
    registerBackend('wasm-test', async () => {
      const {wasm} = await init();
      return new BackendWasm(wasm);
    }, 100);

    // Silences backend registration warnings.
    // spyOn(console, 'warn');
    // spyOn(console, 'log');
  });

  afterEach(() => {
    resetWasmPath();
    removeBackend('wasm-test');
  });

  it('backend init fails when the path is invalid', async () => {
    setWasmPath('invalid/path');
    let wasmPath: string;
    const realFetch = fetch;
    spyOn(self, 'fetch').and.callFake((path: string) => {
      wasmPath = path;
      return realFetch(path);
    });
    expect(await tf.setBackend('wasm-test')).toBe(false);
    expect(wasmPath).toBe('invalid/path');
  });

  it('backend init succeeds with default path', async () => {
    expect(await tf.setBackend('wasm-test')).toBe(true);
  });

  it('setWasmPath called too late', async () => {
    // Set an invalid path.
    setWasmPath('invalid/path');
    await tf.setBackend('wasm-test');

    // Setting the path too late.
    expect(() => setWasmPath('too/late'))
        .toThrowError(/The WASM backend was already initialized. Make sure/);
  });

  // it('should work for 2d', async () => {
  //   const indices = tf.tensor1d([0, 4, 2], 'int32');
  //   const updates = tf.tensor2d(
  //       [100, 101, 102, 777, 778, 779, 1000, 1001, 1002], [3, 3], 'int32');
  //   const shape = [5, 3];
  //   const result = tf.scatterND(indices, updates, shape);
  //   expect(result.shape).toEqual(shape);
  //   expect(result.dtype).toEqual(updates.dtype);
  //   const resultData = await result.data();
  //   console.log(Array.from(resultData));
  //   // expectArraysClose(
  //   //     await result.data(),
  //   //     [100, 101, 102, 0, 0, 0, 1000, 1001, 1002, 0, 0, 0, 777, 778,
  //   779]);
  // });

  // it('should work for multiple 1d', async () => {
  //   const indices = tf.tensor1d([0, 4, 2], 'int32');
  //   const updates = tf.tensor1d([100, 101, 102], 'float32');
  //   const shape = [5];
  //   const result = tf.scatterND(indices, updates, shape);
  //   expect(result.shape).toEqual(shape);
  //   expect(result.dtype).toEqual(updates.dtype);
  //   const resultData = await result.data();
  //   console.log(Array.from(resultData));
  //   // expectArraysClose(await result.data(), [100, 0, 102, 0, 101]);
  // });

  // it('should work for high rank indices', async () => {
  //   const indices = tf.tensor2d([0, 2, 0, 1], [2, 2], 'int32');
  //   const updates = tf.tensor1d([10, 20], 'float32');
  //   const shape = [3, 3];
  //   const result = tf.scatterND(indices, updates, shape);
  //   expect(result.shape).toEqual(shape);
  //   expect(result.dtype).toEqual(updates.dtype);
  //   const resultData = await result.data();
  //   console.log(Array.from(resultData));
  //   // expectArraysClose(await result.data(), [0, 20, 10, 0, 0, 0, 0, 0, 0]);
  // });

  // it('should sum the duplicated indices', async () => {
  //   const indices = tf.tensor1d([0, 4, 2, 1, 3, 0], 'int32');
  //   const updates = tf.tensor1d([10, 20, 30, 40, 50, 60], 'float32');
  //   const shape = [8];
  //   const result = tf.scatterND(indices, updates, shape);
  //   expect(result.shape).toEqual(shape);
  //   expect(result.dtype).toEqual(updates.dtype);
  //   const resultData = await result.data();
  //   console.log(Array.from(resultData));
  //   // expectArraysClose(await result.data(), [70, 40, 30, 50, 20, 0, 0, 0]);
  // });
});
