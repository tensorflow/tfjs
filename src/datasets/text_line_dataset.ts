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
import {DataStream} from '../streams/data_stream';
import {DatasetElement} from '../types';

/**
 * Represents a potentially large collection of text lines.
 *
 * The produced `DatasetElement`s each contain a single string value, with the
 * key given by the `columnName` argument (default 'line').
 *
 * The results are not batched.
 */
export class TextLineDataset extends Dataset {
  /**
   * Create a `TextLineDataset`.
   *
   * @param input A `DataSource` providing a chunked, UTF8-encoded byte stream.
   * @param columnName The key to use in the resulting `DatasetElement`s
   *   (default 'line').
   */
  constructor(
      protected readonly input: DataSource,
      protected readonly columnName = 'line') {
    super();
  }

  getStream(): DataStream<DatasetElement> {
    const readStream = this.input.getStream();
    const utf8Stream = readStream.decodeUTF8();
    const lineStream = utf8Stream.split('\n');
    return lineStream.map(x => ({[this.columnName]: x}));
  }
}
