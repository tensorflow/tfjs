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

import {KernelConfig, StaticRegexReplace, StaticRegexReplaceAttrs} from '@tensorflow/tfjs-core';
import {createSimpleUnaryImpl} from '../utils/unary_impl';
import {unaryKernelFuncFromImpl} from '../utils/unary_utils';

export const staticRegexReplaceImpl = createSimpleUnaryImpl<string,
  string>((x: string, attrs) => {
    const {pattern, replaceGlobal, rewrite} =
      attrs as unknown as StaticRegexReplaceAttrs;
    // TODO(mattSoulanille): Don't create a regex each time.
    return x.replace(new RegExp(pattern, replaceGlobal ? 'g' : ''), rewrite);
});

const staticRegexReplace =
  unaryKernelFuncFromImpl(StaticRegexReplace, staticRegexReplaceImpl);

export const staticRegexReplaceConfig: KernelConfig = {
  kernelName: StaticRegexReplace,
  backendName: 'cpu',
  kernelFunc: staticRegexReplace,
};
