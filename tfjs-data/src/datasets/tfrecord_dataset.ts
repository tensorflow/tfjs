/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {TensorContainerObject} from '@tensorflow/tfjs-core';
import {Dataset} from '../dataset';
import {TFRecordDataSource} from '../sources/tfrecord_data_source';
import {LazyIterator} from '../iterators/lazy_iterator';

/**
 * Reads a record from the file.
 *
 * Return a Dataset comprising records from TFRecord file.
 *
 * The results are batched.
 */
/** @doc {heading: 'Data', subheading: 'Classes', namespace: 'data'} */
export class TFRecordDataset extends Dataset<TensorContainerObject> {
  constructor(protected readonly input: TFRecordDataSource) {
    super();
  }

  async iterator(): Promise<LazyIterator<TensorContainerObject>> {
    return this.input.iterator();
  }
}
