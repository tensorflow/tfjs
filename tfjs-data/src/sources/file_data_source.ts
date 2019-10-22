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

import {env} from '@tensorflow/tfjs-core';
import {DataSource} from '../datasource';
import {ByteChunkIterator} from '../iterators/byte_chunk_iterator';
import {FileChunkIterator, FileChunkIteratorOptions} from '../iterators/file_chunk_iterator';
import {FileElement} from '../types';
import {isLocalPath} from '../util/source_util';

/**
 * Represents a file, blob, or Uint8Array readable as a stream of binary data
 * chunks.
 */
export class FileDataSource extends DataSource {
  /**
   * Create a `FileDataSource`.
   *
   * @param input Local file path, or `File`/`Blob`/`Uint8Array` object to
   *     read. Local file only works in node environment.
   * @param options Options passed to the underlying `FileChunkIterator`s,
   *   such as {chunksize: 1024}.
   */
  constructor(
      protected input: FileElement|string,
      protected readonly options: FileChunkIteratorOptions = {}) {
    super();
  }

  async iterator(): Promise<ByteChunkIterator> {
    if (isLocalPath(this.input) && env().get('IS_NODE')) {
      // tslint:disable-next-line:no-require-imports
      const fs = require('fs');
      this.input = fs.readFileSync((this.input as string).substr(7));
    }
    // TODO(kangyizhang): Add LocalFileChunkIterator to split local streaming
    // with file in browser.
    return new FileChunkIterator(this.input as FileElement, this.options);
  }
}
