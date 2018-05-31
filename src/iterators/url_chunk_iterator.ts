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

import {ByteChunkIterator} from './byte_chunk_iterator';
import {FileChunkIterator} from './file_chunk_iterator';
import {QueueIterator} from './lazy_iterator';

// We wanted multiple inheritance, e.g.
//   class URLIterator extends QueueIterator<Uint8Array>, ByteChunkIterator
// but the TypeScript mixin approach is a bit hacky, so we take this adapter
// approach instead.

export class URLChunkIterator extends ByteChunkIterator {
  private impl: URLChunkIteratorImpl;

  constructor(url: RequestInfo, options = {}) {
    super();
    this.impl = new URLChunkIteratorImpl(url, options);
  }

  async next() {
    return this.impl.next();
  }
}

/**
 * Provide a stream of chunks from a URL.
 *
 * Note this class first downloads the entire file into memory before providing
 * the first element from the stream.  This is because the Fetch API does not
 * yet reliably provide a reader stream for the response body.
 */
class URLChunkIteratorImpl extends QueueIterator<Uint8Array> {
  private blobPromise: Promise<Blob>;
  private fileChunkIterator: FileChunkIterator;

  /**
   * Create a `URLChunkIteratorImpl`.
   *
   * @param url A source URL string, or a `Request` object.
   * @param options Options passed to the underlying `FileChunkIterator`s,
   *   such as {chunksize: 1024}.
   * @returns an Iterator of Uint8Arrays containing sequential chunks of the
   *   input file.
   */
  constructor(protected url: RequestInfo, protected options = {}) {
    super();

    this.blobPromise = fetch(url, options).then(response => {
      if (response.ok) {
        return response.blob();
      } else {
        throw new Error(response.statusText);
      }
    });
  }

  async pump(): Promise<boolean> {
    if (this.fileChunkIterator == null) {
      const blob = await this.blobPromise;
      this.fileChunkIterator = new FileChunkIterator(blob, this.options);
    }
    const chunkResult = await this.fileChunkIterator.next();
    if (chunkResult.done) {
      return false;
    }
    this.outputQueue.push(chunkResult.value);
    return true;
  }
}
