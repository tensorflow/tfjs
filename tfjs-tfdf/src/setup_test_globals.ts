/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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
import jszip from 'jszip';
// Karma does not preserve the source path, causing the TFDF library to look for
// the wasm binary in the root path, so fix the path for the library.
// tslint:disable-next-line:no-any
(self as any).__filename = '/base/tfjs/tfjs-tfdf/wasm/';
// TFDF is Closure-compiled, and expects a global JSZIP variable to exist,
// rather than passed in as a module.
// tslint:disable-next-line:no-any
(window as any).JSZip = jszip;
