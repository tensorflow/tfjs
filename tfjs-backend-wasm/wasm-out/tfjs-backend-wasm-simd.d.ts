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

export interface BackendWasmModuleSimd extends EmscriptenModule {
  onRuntimeInitialized: () => void;
  onAbort: (msg: string) => void;
  // Using the tfjs namespace to avoid conflict with emscripten's API.
  tfjs: {
    init(): void,
    registerTensor(id: number, size: number, memoryOffset: number): void,
    // Disposes the data behind the data bucket.
    disposeData(id: number): void,
    // Disposes the backend and all of its associated data.
    dispose(): void,
  }
}

export interface WasmFactoryConfigSimd {
  locateFile?(path: string, prefix: string): string;
  instantiateWasm?: Function;
}

declare var moduleFactory: (settings: WasmFactoryConfigSimd) =>
    BackendWasmModuleSimd;
export default moduleFactory;
