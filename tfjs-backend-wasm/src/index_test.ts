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
import {registerBackend, removeBackend, test_util, env} from '@tensorflow/tfjs-core';
// tslint:disable-next-line:no-imports-from-dist
import {ALL_ENVS, BROWSER_ENVS, describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {init, resetWasmPath} from './backend_wasm';
import {BackendWasm, setWasmPath, setWasmPaths} from './index';
import {VALID_PREFIX, setupCachedWasmPaths} from './test_util';

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

  // TODO(mattSoulanille): Re-enable this once it's working on iOS.
  // tslint:disable-next-line: ban
  xit('allocates more than two gigabytes', async () => {
    const size = 2**30 / 4; // 2**30 bytes (4 bytes per number) = 1GB

    // Allocate 3 gigabytes.
    const t0 = tf.zeros([size], 'float32');
    const t1 = tf.ones([size], 'float32');
    const t2 = t1.mul(2);

    // Helper function to check if all the values in a tensor equal an expected
    // value.
    async function check(tensor: tf.Tensor, name: string, val: number) {
      const arr = await tensor.data();
      for (let i = 0; i < size; i++) {
        if (arr[i] !== val) {
          throw new Error(`${name}[${i}] == ${arr[i]} but should be ${val}`);
        }
      }
    }

    await check(t0, 't0', 0);
    await check(t1, 't1', 1);
    await check(t2, 't2', 2);
  });
});

describeWithFlags('wasm init', BROWSER_ENVS, () => {
  beforeEach(async () => {
    await setupCachedWasmPaths();
    registerBackend('wasm-test', async () => {
      const {wasm} = await init();
      return new BackendWasm(wasm);
    }, 100);

    // Silences backend registration warnings.
    spyOn(console, 'warn');
    spyOn(console, 'log');
  });

  afterEach(async () => {
    removeBackend('wasm-test');
  });

  afterAll(setupCachedWasmPaths);

  it('backend init fails when the path is invalid', async () => {
    resetWasmPath();
    setWasmPath('invalid/path');
    let wasmPath: string;
    spyOn(self, 'fetch').and.callFake((path: string) => {
      wasmPath = path;
      return Promise.reject(
        new TypeError('Failed to fetch because invalid path'));
    });
    expect(await tf.setBackend('wasm-test')).toBe(false);
    expect(wasmPath).toBe('invalid/path');
  });

  it('backend init fails when setWasmPaths is called with ' +
         'an invalid prefix',
     async () => {
       resetWasmPath();
       setWasmPaths('invalid/prefix/');
       let wasmPath: string;
       spyOn(self, 'fetch').and.callFake((path: string) => {
         wasmPath = path;
         return Promise.reject(
           new TypeError('Failed to fetch because invalid prefix'));
       });
       expect(await tf.setBackend('wasm-test')).toBe(false);
       expect(wasmPath).toContain('invalid/prefix');
     });

  it('backend init fails when setWasmPaths is called with ' +
         'an invalid fileMap',
     async () => {
       resetWasmPath();
       setWasmPaths({
         'tfjs-backend-wasm.wasm': 'invalid/path',
         'tfjs-backend-wasm-simd.wasm': 'invalid/path',
         'tfjs-backend-wasm-threaded-simd.wasm': 'invalid/path'
       });
       let wasmPathPrefix: string;
       spyOn(self, 'fetch').and.callFake((path: string) => {
         wasmPathPrefix = path;
         return Promise.reject(
           new TypeError('Failed to fetch because invalid paths'));
       });
       expect(await tf.setBackend('wasm-test')).toBe(false);
       expect(wasmPathPrefix).toBe('invalid/path');
     });

  it('setWasmPaths throws error when called without specifying a path for ' +
         'each binary',
     async () => {
       expect(() => {
         setWasmPaths({
           'tfjs-backend-wasm.wasm': '/base/wasm-out/tfjs-backend-wasm.wasm'
         });
       }).toThrow();
     });

  describe('platform fetch', () => {
    let fetchSpy: jasmine.Spy;
    let realFetch: typeof fetch;

    beforeEach(() => {
      realFetch = env().platform.fetch;
      fetchSpy = spyOn(env().platform, 'fetch');
    });

    afterEach(() => {
      env().platform.fetch = realFetch;
    });

    it('backend init works when the path is valid', async () => {
      const usePlatformFetch = true;
      resetWasmPath();
      setWasmPaths(VALID_PREFIX, usePlatformFetch);
      let wasmPath: string;
      fetchSpy.and.callFake((path: string) => {
        wasmPath = path;
        return realFetch(path);
      });
      expect(await tf.setBackend('wasm-test')).toBe(true);
      const validPaths = new Set([
        VALID_PREFIX + 'tfjs-backend-wasm.wasm',
        VALID_PREFIX + 'tfjs-backend-wasm-simd.wasm',
        VALID_PREFIX + 'tfjs-backend-wasm-threaded-simd.wasm',
      ]);
      expect(validPaths).toContain(wasmPath);
    });

    it('backend init works when the wasm paths overrides map is valid',
        async () => {
          const usePlatformFetch = true;
          setWasmPaths({
            'tfjs-backend-wasm.wasm': `${VALID_PREFIX}tfjs-backend-wasm.wasm`,
            'tfjs-backend-wasm-simd.wasm':
            `${VALID_PREFIX}tfjs-backend-wasm-simd.wasm`,
            'tfjs-backend-wasm-threaded-simd.wasm':
            `${VALID_PREFIX}tfjs-backend-wasm-threaded-simd.wasm`,
          }, usePlatformFetch);
          let wasmPath: string;
          fetchSpy.and.callFake((path: string) => {
            wasmPath = path;
            return realFetch(path);
          });
          expect(await tf.setBackend('wasm-test')).toBe(true);
          const validPaths = new Set([
            VALID_PREFIX + 'tfjs-backend-wasm.wasm',
            VALID_PREFIX + 'tfjs-backend-wasm-simd.wasm',
            VALID_PREFIX + 'tfjs-backend-wasm-threaded-simd.wasm',
          ]);
          expect(validPaths).toContain(wasmPath);
        });

    it('backend init works when the path is valid', async () => {
      const usePlatformFetch = true;
      const validPath = VALID_PREFIX + 'tfjs-backend-wasm.wasm';
      setWasmPath(validPath, usePlatformFetch);
      let wasmPath: string;
      fetchSpy.and.callFake((path: string) => {
        wasmPath = path;
        return realFetch(path);
      });
      expect(await tf.setBackend('wasm-test')).toBe(true);
      expect(wasmPath).toBe(validPath);
    });

    // Disabling this test because it intermittently times out on CI.
    // tslint:disable-next-line: ban
    xit('backend init fails when the path is invalid', async () => {
      const usePlatformFetch = true;
      setWasmPath('invalid/path', usePlatformFetch);
      let wasmPath: string;
      fetchSpy.and.callFake((path: string) => {
        wasmPath = path;
        return Promise.reject(
          new TypeError('Failed to fetch because invalid path'));
      });
      expect(await tf.setBackend('wasm-test')).toBe(false);
      expect(wasmPath).toBe('invalid/path');
    });
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
});
