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

import {Dataset} from '../dataset';
import {DataSource} from '../datasource';
import {LazyIterator} from '../iterators/lazy_iterator';

/**
 * Represents a potentially large collection of text lines.
 *
 * The results are not batched.
 */
export class TextLineDataset extends Dataset<string> {
  /**
   * Create a `TextLineDataset`.
   *
   * @param input A `DataSource` providing a chunked, UTF8-encoded byte stream.
   */
  constructor(protected readonly input: DataSource) {
    super();
  }

  async iterator(): Promise<LazyIterator<string>> {
    const inputIterator = await this.input.iterator();
    const utf8Iterator = inputIterator.decodeUTF8();
    const lineIterator = utf8Iterator.split('\n').map(line => {
      // Windows/DOS format text file has extra line breaker at the end of line.
      if (line.endsWith('\r')) {
        line = line.slice(0, -1);
      }
      return line;
    });
    return lineIterator;
  }
}
