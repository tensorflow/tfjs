/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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
import {Reverse, ReverseAttrs} from '../kernel_names';
import {GradConfig, NamedAttrMap} from '../kernel_registry';
import {reverse} from '../ops/reverse';
import {Tensor} from '../tensor';
import {parseAxisParam} from '../util';

export const reverseGradConfig: GradConfig = {
  kernelName: Reverse,
  gradFunc: (dy: Tensor, saved: Tensor[], attrs: NamedAttrMap) => {
    const {dims} = attrs as {} as ReverseAttrs;
    const axes = parseAxisParam(dims, dy.shape);
    return {x: () => reverse(dy, axes)};
  }
};
