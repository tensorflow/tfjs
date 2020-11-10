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
import '@tensorflow/tfjs-backend-cpu';
import * as tfc from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {parseAndEvaluateSnippets} from '@tensorflow/tfjs-core/dist/scripts/test_snippets/util';
import * as tfl from '@tensorflow/tfjs-layers';

import * as tfd from '../src/index';

const tf = {
  data: {...tfd},
  ...tfc,
  ...tfl
};
parseAndEvaluateSnippets(tf);
