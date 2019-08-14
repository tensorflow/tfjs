/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 *
 * =============================================================================
 */

import {util} from '@tensorflow/tfjs-core';
import {FileChunkIterator, FileChunkIteratorOptions} from './file_chunk_iterator';

/**
 * Provide a stream of chunks from a URL.
 *
 * Note this class first downloads the entire file into memory before providing
 * the first element from the stream.  This is because the Fetch API does not
 * yet reliably provide a reader stream for the response body.
 */
export async function urlChunkIterator(
    url: RequestInfo, options: FileChunkIteratorOptions = {}) {
  let urlString;
  let requestInit;
  if ((typeof url) === 'string') {
    urlString = url as string;
  } else {
    urlString = (url as Request).url;
    requestInit = getRequestInitFromRequest(url as Request);
  }
  const response = await util.fetch(urlString, requestInit);
  if (response.ok) {
    const uint8Array = new Uint8Array(await response.arrayBuffer());
    return new FileChunkIterator(uint8Array, options);
  } else {
    throw new Error(response.statusText);
  }
}

// Generate RequestInit from Request to match tf.util.fetch signature.
const getRequestInitFromRequest = (request: Request) => {
  const init = {
    method: request.method,
    headers: request.headers,
    body: request.body,
    mode: request.mode,
    credentials: request.credentials,
    cache: request.cache,
    redirect: request.redirect,
    referrer: request.referrer,
    integrity: request.integrity,
  };
  return init;
};
