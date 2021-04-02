/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {TFWebModelRunnerClass} from './types/tfweb_model_runner';

/**
 * The root type for the tfweb client.
 *
 * The client is imported from src/tfweb_client.js which imports the actual JS
 * code from deps/tfweb_client.js (see comments in src/tfweb_client.js for more
 * details). The JS client code exports various components to the tfweb.*
 * namespaces, e.g. tfweb.tfweb.setWasmPath, tfweb.TFWebModelRunner, etc. The
 * types defined in this file will match the actual types for those exported
 * components.
 */
declare interface TFWebClient {
  tfweb: {
    /**
     * Sets the path to load all the WASM module files from.
     *
     * TFWeb will automatically load WASM module with the best performance
     * based on whether the current browser supports WebAssembly SIMD and
     * multi-threading.
     *
     * @param path The path to load WASM module files from.
     *     For relative path, use :
     *         'relative/path/' (relative to current url path) or
     *         '/relative/path' (relative to the domain root).
     *     For absolute path, use https://some-server.com/absolute/path/.
     */
    setWasmPath(path: string): void;
  };

  /**
   * The generic TFLite model runner class.
   */
  TFWebModelRunner: TFWebModelRunnerClass;

  // TODO: add task libraries.
}

/**
 * The main export.
 *
 * The variable name needs to be "tfweb" so it can match the root namespace of
 * the exported components from the JS client.
 */
export declare let tfweb: TFWebClient;
