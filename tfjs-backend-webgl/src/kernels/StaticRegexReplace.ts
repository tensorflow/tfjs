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

import {backend_util, KernelConfig, KernelFunc, NamedAttrMap, StaticRegexReplace, StaticRegexReplaceAttrs, StaticRegexReplaceInputs, TensorInfo} from '@tensorflow/tfjs-core';
import {MathBackendWebGL} from '../backend_webgl';
import {staticRegexReplaceImplCPU} from '../kernel_utils/shared';

export function staticRegexReplace(args: {
  inputs: StaticRegexReplaceInputs,
  backend: MathBackendWebGL,
  attrs: StaticRegexReplaceAttrs,
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;

  if (x.dtype !== 'string') {
    throw new Error('Input must be of datatype string');
  }

  const $x = backend.readSync(x.dataId) as Uint8Array[];

  const stringInput = backend_util.fromUint8ToStringArray($x);
  const output = staticRegexReplaceImplCPU(stringInput, 'string',
                                           attrs as unknown as NamedAttrMap);

  return backend.makeTensorInfo(x.shape, 'string', output);
}

export const staticRegexReplaceConfig: KernelConfig = {
  kernelName: StaticRegexReplace,
  backendName: 'webgl',
  kernelFunc: staticRegexReplace as unknown as KernelFunc,
};
