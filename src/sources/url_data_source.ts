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

import {DataSource} from '../datasource';
import {ByteChunkIterator} from '../iterators/byte_chunk_iterator';
import {FileChunkIteratorOptions} from '../iterators/file_chunk_iterator';
import {urlChunkIterator} from '../iterators/url_chunk_iterator';
import {isLocalPath} from '../util/source_util';
import {FileDataSource} from './file_data_source';

/*
 * Represents a URL readable as a stream of binary data chunks.
 */
export class URLDataSource extends DataSource {
  /**
   * Create a `URLDataSource`.
   *
   * @param url A source URL string, or a `Request` object.
   * @param options Options passed to the underlying `FileChunkIterator`s,
   *   such as {chunksize: 1024}.
   */
  constructor(
      protected readonly url: RequestInfo,
      protected readonly fileOptions: FileChunkIteratorOptions = {}) {
    super();
  }

  // TODO(soergel): provide appropriate caching options.  Currently this
  // will download the URL anew for each call to iterator().  Since we have
  // to treat the downloaded file as a blob/buffer anyway, we may as well retain
  // it-- but that raises GC issues.  Also we may want a persistent disk cache.
  async iterator(): Promise<ByteChunkIterator> {
    if (isLocalPath(this.url)) {
      return (new FileDataSource(this.url as string, this.fileOptions))
          .iterator();
    } else {
      return urlChunkIterator(this.url, this.fileOptions);
    }
  }
}
