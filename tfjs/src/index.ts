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
 * =============================================================================
 */

import '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import '@tensorflow/tfjs-core/dist/register_all_gradients';
// tslint:disable-next-line: no-imports-from-dist
import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';

export * from '@tensorflow/tfjs-core';
export * from '@tensorflow/tfjs-layers';
export * from '@tensorflow/tfjs-converter';

// Export data api as tf.data
import * as data from '@tensorflow/tfjs-data';
export {data};

// Import and register backends.
import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';

// Import versions of all sub-packages.
import {version_core} from '@tensorflow/tfjs-core';
import {version_cpu} from '@tensorflow/tfjs-backend-cpu';
import {version_webgl} from '@tensorflow/tfjs-backend-webgl';
import {version_data} from '@tensorflow/tfjs-data';
import {version_layers} from '@tensorflow/tfjs-layers';
import {version_converter} from '@tensorflow/tfjs-converter';
import {version as version_union} from './version';

export const version = {
  'tfjs-core': version_core,
  'tfjs-backend-cpu': version_cpu,
  'tfjs-backend-webgl': version_webgl,
  'tfjs-data': version_data,
  'tfjs-layers': version_layers,
  'tfjs-converter': version_converter,
  'tfjs': version_union
};
