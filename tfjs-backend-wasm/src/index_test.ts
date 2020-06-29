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

import * as tf from '@tensorflow/tfjs-core';
import {registerBackend, removeBackend, test_util, util} from '@tensorflow/tfjs-core';
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

  it('allocates buffers with byteOffsets', async () => {
    const data = [-0.5, 0.5, 3.14];
    const buffer = new ArrayBuffer(32);
    const view = new Float32Array(buffer, 8, data.length);

    // Write values to buffer.
    for (let i = 0; i < data.length; ++i) {
      view[i] = data[i];
    }

    const t = tf.tensor(view);
    // Tensor values should match.
    test_util.expectArraysClose(await t.data(), view);
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

  it('backend init works when the path is valid and use platform fetch',
     async () => {
       const usePlatformFetch = true;
       const validPath = '/base/wasm-out/tfjs-backend-wasm.wasm';
       setWasmPath(validPath, usePlatformFetch);
       let wasmPath: string;
       const realFetch = util.fetch;
       spyOn(util, 'fetch').and.callFake((path: string) => {
         wasmPath = path;
         return realFetch(path);
       });
       expect(await tf.setBackend('wasm-test')).toBe(true);
       expect(wasmPath).toBe(validPath);
     });

  // Disabling this test because it intermittently times out on CI.
  // tslint:disable-next-line: ban
  xit('backend init fails when the path is invalid and use platform fetch',
      async () => {
        const usePlatformFetch = true;
        setWasmPath('invalid/path', usePlatformFetch);
        let wasmPath: string;
        const realFetch = util.fetch;
        spyOn(util, 'fetch').and.callFake((path: string) => {
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

  it('accepts a tensor-like object', async () => {
    const input = [1, 2, 3];
    const result = tf.reverse(input);
    expect(result.shape).toEqual([3]);
    const data = await result.data();
    console.log(data);
    // expectArraysClose(await result.data(), [3, 2, 1]);
  });

  fit('reverse a 4D array at axis [0]', async () => {
    const shape: [number, number, number, number] = [3, 2, 3, 4];
    const data = [
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
      18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
      36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
      54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71
    ];
    const input = tf.tensor4d(data, shape);
    const result = tf.reverse4d(input, [0]);
    expect(result.shape).toEqual(input.shape);
    const out = await result.data();
    console.log(Array.from(out));
    // expectArraysClose(await result.data(), [
    //   48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    //   66, 67, 68, 69, 70, 71, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
    //   36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 0,  1,  2,  3,  4,  5,
    //   6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
    // ]);
  });
});
