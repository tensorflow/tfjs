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

import {ByteChunkIterator} from './iterators/byte_chunk_iterator';

/**
 * Represents a data source readable as a stream of binary data chunks.
 *
 * Because `Dataset`s can be read repeatedly (via `Dataset.iterator()`), this
 * provides a means to repeatedly create streams from the underlying data
 * sources.
 */
export abstract class DataSource {
  /**
   * Obtain a new stream of binary data chunks.
   *
   * Starts the new stream from the beginning of the data source, even if other
   * streams have been obtained previously.
   */
  abstract async iterator(): Promise<ByteChunkIterator>;

  // TODO(soergel): consider chainable Dataset construction here
}

// TODO(soergel): consider convenience factory functions here
// in combination with chainable source->dataset above, e.g.:
// tf.data.url(...).asCsvDataset().shuffle().batch()
