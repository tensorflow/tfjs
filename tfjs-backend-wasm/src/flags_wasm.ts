/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {env} from '@tensorflow/tfjs-core';

const ENV = env();

/**
 * True if SIMD is supported.
 */
// From: https://github.com/GoogleChromeLabs/wasm-feature-detect
ENV.registerFlag(
    // This typed array passed in to WebAssembly.validate is WebAssembly binary
    // code. In this case it is a small program that contains SIMD
    // instructions.
    'WASM_HAS_SIMD_SUPPORT', async () => WebAssembly.validate(new Uint8Array([
      0, 97, 115, 109, 1, 0, 0, 0, 1,  4, 1,   96, 0,  0, 3,
      2, 1,  0,   10,  9, 1, 7, 0, 65, 0, 253, 15, 26, 11
    ])));

/**
 * True if threads are supported.
 */
// From: https://github.com/GoogleChromeLabs/wasm-feature-detect
ENV.registerFlag('WASM_HAS_MULTITHREAD_SUPPORT', async () => {
  // TODO(annxingyuan): Enable node support once this is resolved:
  // https://github.com/tensorflow/tfjs/issues/3830
  if (ENV.get('IS_NODE')) {
    return false;
  }

  try {
    // Test for transferability of SABs (needed for Firefox)
    // https://groups.google.com/forum/#!msg/mozilla.dev.platform/IHkBZlHETpA/dwsMNchWEQAJ
    new MessageChannel().port1.postMessage(new SharedArrayBuffer(1));
    // This typed array is a WebAssembly program containing threaded
    // instructions.
    return WebAssembly.validate(new Uint8Array([
      0, 97, 115, 109, 1, 0,  0,  0, 1, 4, 1,  96, 0,   0,  3, 2, 1,  0, 5,
      4, 1,  3,   1,   1, 10, 11, 1, 9, 0, 65, 0,  254, 16, 2, 0, 26, 11
    ]));
  } catch (e) {
    return false;
  }
});
