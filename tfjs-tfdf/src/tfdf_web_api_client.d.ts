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
import {TFDFWebAPIClient} from './types/tfdf_web_api';

/**
 * The main export for tfdf_web_api_client.js types.
 */
export declare const tfdfWeb: () => Promise<TFDFWebAPIClient>;

type LocateFileFunction = (path: string, prefix?: string) => string;
/** The global function to set WASM path. */
export declare const setLocateFile: (locateFile: LocateFileFunction) => void;
