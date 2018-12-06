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

import {ENV} from '@tensorflow/tfjs-core';
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
  let response;
  if (ENV.get('IS_BROWSER')) {
    response = await fetch(url);
    if (response.ok) {
      const blob = await response.blob();
      return new FileChunkIterator(blob, options);
    } else {
      throw new Error(response.statusText);
    }
  } else {
    // TODO(kangyizhang): Provide argument for users to use http.request with
    // headers in node.
    // tslint:disable-next-line:no-require-imports
    const nodeFetch = require('node-fetch');
    if (typeof url !== 'string') {
      throw new Error(
          'URL must be a string. Request objects are not supported ' +
          'in the node.js environment yet.');
    }
    response = await nodeFetch(url);
    if (response.ok) {
      const unitArray = await response.buffer();
      return new FileChunkIterator(unitArray, options);
    } else {
      throw new Error(response.statusText);
    }
  }
}
