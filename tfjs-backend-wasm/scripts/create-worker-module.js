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

/**
 * This file creates a TypeScript module that exports the contents of the
 * Emscripten-generated WASM worker script so that it can be inlined by the
 * tf-backend-wasm bundle.
 */

const fs = require('fs');

const BASE_PATH = './wasm-out/';
const WORKER_PATH = `${BASE_PATH}tfjs-backend-wasm-threaded-simd.worker.js`;

// Write out a worker TypeScript module.
const workerContents = fs.readFileSync(WORKER_PATH, "utf8");
fs.writeFileSync(`${BASE_PATH}tfjs-backend-wasm-threaded-simd.worker.ts`,
  `export const wasmWorkerContents = '${workerContents.trim()}';`);

// Delete the original worker file.
fs.unlinkSync(WORKER_PATH)
