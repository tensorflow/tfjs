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
import {IDXIterator} from '../iterators/idx_iterator';
import {LazyIterator} from '../iterators/lazy_iterator';
import {DataElementObject} from '../types';

/**
 * A potentially large collection of Tensors parsed from an IDX source.
 *
 * The produced `DatasetElement`s each contain a single Tensor, with the
 * key given by the `columnName` argument (default 'data').
 *
 * The results are not batched.
 */
export class IDXDataset extends Dataset<DataElementObject> {
  /**
   * Create an `IDXDataset`.
   *
   * @param input A `DataSource` providing a chunked, UTF8-encoded byte stream.
   * @param columnName The key to use in the resulting `DatasetElement`s
   *   (default 'data').
   */
  constructor(
      protected readonly input: DataSource,
      protected readonly columnName = 'data') {
    super();
  }

  async iterator(): Promise<LazyIterator<DataElementObject>> {
    const tensorStream = new IDXIterator(await this.input.iterator());
    return tensorStream.map(
        x => ({[this.columnName]: x} as any as DataElementObject));
  }
}
