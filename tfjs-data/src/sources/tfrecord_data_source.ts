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

import {env} from '@tensorflow/tfjs-core';
import {TFRecordIterator} from '../iterators/tfrecord_iterator';
import {isLocalPath} from '../util/source_util';

export class TFRecordDataSource {
  /**
   * Create a `TFRecordDataSource`.
   *
   * @param input Local file path.
   *     Only works in node environment.
   */
  constructor(protected input: string) {
    if (!env().get('IS_NODE')) {
      throw new Error(
        'tf.data.TFRecord is only supported in node environment.');
    }
    if (!isLocalPath(input)) {
      throw new Error(
        `tf.data.TFRecord is only supported path with prefix 'file://'.`);
    }
  }

  async iterator(): Promise<TFRecordIterator> {
    return new TFRecordIterator(this.input);
  }
}
