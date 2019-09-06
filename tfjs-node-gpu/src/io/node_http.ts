/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {io} from '@tensorflow/tfjs-core';

/**
 * Factory function for HTTP IO Handler in Node.js.
 *
 * @param path URL path or an array of them.
 * @param requestInit Request init for the HTTP IOHandler. May include fields
 *   such as "credentials" and "cache". (Optional)
 * @param weightPathPrefix A path prefix for weight loading . (Optional).
 */
export function nodeHTTPRequest(
    path: string, requestInit?: RequestInit,
    weightPathPrefix?: string): io.IOHandler {
  return io.browserHTTPRequest(path, {requestInit, weightPathPrefix});
}

export const nodeHTTPRequestRouter = (url: string) => {
  let isHTTP = true;
  if (Array.isArray(url)) {
    isHTTP = url.every(urlItem => io.isHTTPScheme(urlItem));
  } else {
    isHTTP = io.isHTTPScheme(url);
  }
  if (isHTTP) {
    return nodeHTTPRequest(url);
  }
  return null;
};
