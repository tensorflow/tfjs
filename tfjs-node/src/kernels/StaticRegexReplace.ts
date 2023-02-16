/**
 * @license
 * Copyright 2023 Google LLC.
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

import {KernelConfig, StaticRegexReplace, StaticRegexReplaceAttrs, StaticRegexReplaceInputs} from '@tensorflow/tfjs';

import {NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const staticRegexReplaceConfig: KernelConfig = {
  kernelName: StaticRegexReplace,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const tensors = args.inputs as unknown as StaticRegexReplaceInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {pattern, rewrite, replaceGlobal} =
      args.attrs as unknown as StaticRegexReplaceAttrs;

    const opAttrs = [
      {name: 'pattern', type: backend.binding.TF_ATTR_STRING, value: pattern},
      {name: 'rewrite', type: backend.binding.TF_ATTR_STRING, value: rewrite},
      {
        name: 'replace_global',
        type: backend.binding.TF_ATTR_BOOL,
        value: replaceGlobal,
      },
    ];

    const inputs = [tensors.x];
    return backend.executeSingleOutput('StaticRegexReplace', opAttrs, inputs);
  }
};
