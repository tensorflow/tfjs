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

import {BackendWasm} from '../backend_wasm';

// Analogous to `struct Result` in `non_max_suppression_impl.h`.
interface Result {
  pSelectedIndices: number;
  selectedSize: number;
  pSelectedScores: number;
  pValidOutputs: number;
}
/**
 * Parse the result of the c++ method, which has the shape equivalent to
 * `Result`.
 */
export function parseResultStruct(
    backend: BackendWasm, resOffset: number): Result {
  const result = new Int32Array(backend.wasm.HEAPU8.buffer, resOffset, 4);
  const pSelectedIndices = result[0];
  const selectedSize = result[1];
  const pSelectedScores = result[2];
  const pValidOutputs = result[3];
  // Since the result was allocated on the heap, we have to delete it.
  backend.wasm._free(resOffset);
  return {pSelectedIndices, selectedSize, pSelectedScores, pValidOutputs};
}
