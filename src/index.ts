/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import * as tfc from '@tensorflow/tfjs-core';
import {NodeJSKernelBackend} from './nodejs_kernel_backend';

// tslint:disable-next-line:no-require-imports
import bindings = require('bindings');
import {TFJSBinding} from './tfjs_binding';

export function bindTensorFlowBackend() {
  // TODO(kreeger): This anonymous function should throw an exception if the
  // binding is not installed.
  const nodeBinding = bindings('tfjs_binding.node') as TFJSBinding;

  tfc.ENV.registerBackend(
      'tensorflow', () => new NodeJSKernelBackend(nodeBinding));
  tfc.setBackend('tensorflow');
}

export {version} from './version';
