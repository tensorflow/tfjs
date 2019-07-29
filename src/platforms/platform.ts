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

import {RequestDetails} from '../io/types';

/**
 * At any given time a single platform is active and represents and
 * implementation of this interface. In practice, a platform is an environment
 * where TensorFlow.js can be executed, e.g. the browser or Node.js.
 */
export interface Platform {
  /**
   * Makes an HTTP request.
   * @param path The URL path to make a request to
   * @param init The request init. See init here:
   *     https://developer.mozilla.org/en-US/docs/Web/API/Request/Request
   */
  fetch(path: string, requestInits?: RequestInit, options?: RequestDetails):
      Promise<Response>;

  /**
   * Returns the current high-resolution time in milliseconds relative to an
   * arbitrary time in the past. It works across different platforms (node.js,
   * browsers).
   */
  now(): number;

  /**
   * Encode the provided string into an array of bytes using the provided
   * encoding.
   */
  encode(text: string, encoding: string): Uint8Array;
  /** Decode the provided bytes into a string using the provided encoding. */
  decode(bytes: Uint8Array, encoding: string): string;
}
