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

import {ByteStream} from './byte_stream';
import {FileReaderStream} from './filereader_stream';
import {QueueStream} from './lazy_iterator';

// We wanted multiple inheritance, e.g.
//   class URLStream extends QueueStream<Uint8Array>, ByteStream
// but the TypeScript mixin approach is a bit hacky, so we take this adapter
// approach instead.

export class URLStream extends ByteStream {
  private impl: URLStreamImpl;

  constructor(url: RequestInfo, options = {}) {
    super();
    this.impl = new URLStreamImpl(url, options);
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
class URLStreamImpl extends QueueStream<Uint8Array> {
  private blobPromise: Promise<Blob>;
  private fileReaderStream: FileReaderStream;

  /**
   * Create a `URLStream`.
   *
   * @param url A source URL string, or a `Request` object.
   * @param options Options passed to the underlying `FileReaderStream`s,
   *   such as {chunksize: 1024}.
   * @returns a Stream of Uint8Arrays containing sequential chunks of the
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
    if (this.fileReaderStream == null) {
      const blob = await this.blobPromise;
      this.fileReaderStream = new FileReaderStream(blob, this.options);
    }
    const chunkResult = await this.fileReaderStream.next();
    if (chunkResult.done) {
      return false;
    }
    this.outputQueue.push(chunkResult.value);
    return true;
  }
}
