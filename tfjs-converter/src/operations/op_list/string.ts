/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
 * =============================================================================
 */

import {OpMapper} from '../types';

export const json: OpMapper[] = [
  {
    'tfOpName': 'StringNGrams',
    'category': 'string',
    'inputs': [
      {'start': 0, 'name': 'data', 'type': 'tensor'},
      {'start': 1, 'name': 'dataSplits', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'separator', 'name': 'separator', 'type': 'string'},
      {'tfName': 'ngram_widths', 'name': 'nGramWidths', 'type': 'number[]'},
      {'tfName': 'left_pad', 'name': 'leftPad', 'type': 'string'},
      {'tfName': 'right_pad', 'name': 'rightPad', 'type': 'string'},
      {'tfName': 'pad_width', 'name': 'padWidth', 'type': 'number'}, {
        'tfName': 'preserve_short_sequences',
        'name': 'preserveShortSequences',
        'type': 'bool'
      }
    ],
    'outputs': ['ngrams', 'ngrams_splits']
  },
  {
    'tfOpName': 'StringSplit',
    'category': 'string',
    'inputs': [
      {'start': 0, 'name': 'input', 'type': 'tensor'},
      {'start': 1, 'name': 'delimiter', 'type': 'tensor'},
    ],
    'attrs': [{'tfName': 'skip_empty', 'name': 'skipEmpty', 'type': 'bool'}],
    'outputs': ['indices', 'values', 'shape']
  },
  {
    'tfOpName': 'StringToHashBucketFast',
    'category': 'string',
    'inputs': [
      {'start': 0, 'name': 'input', 'type': 'tensor'},
    ],
    'attrs': [{'tfName': 'num_buckets', 'name': 'numBuckets', 'type': 'number'}]
  }
];
