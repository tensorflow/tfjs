import {OpMapper} from '../types';

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

export const json: OpMapper[] = [
  {
    'tfOpName': 'DecodeBase64',
    'category': 'string',
    'inputs': [{'start': 0, 'name': 'input', 'type': 'tensor'}]
  },
  {
    'tfOpName': 'EncodeBase64',
    'category': 'string',
    'inputs': [{'start': 0, 'name': 'input', 'type': 'tensor'}],
    'attrs': [{'tfName': 'pad', 'name': 'pad', 'type': 'bool'}]
  }
];
