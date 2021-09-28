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

import {env} from '@tensorflow/tfjs-core';

const ENV = env();

/** Whether to keep intermediate tensors. */
ENV.registerFlag('KEEP_INTERMEDIATE_TENSORS', () => false, debugValue => {
  if (debugValue) {
    console.warn(
        'Keep intermediate tensors is ON. This will print the values of all ' +
        'intermediate tensors during model inference. Not all models ' +
        'support this mode. For details, check e2e/benchmarks/ ' +
        'model_config.js. This significantly impacts performance.');
  }
});
