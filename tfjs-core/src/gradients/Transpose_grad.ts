/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {Transpose, TransposeAttrs} from '../kernel_names';
import {GradConfig, NamedAttrMap} from '../kernel_registry';
import * as axis_util from '../ops/axis_util';
import {transpose} from '../ops/transpose';
import {Tensor} from '../tensor';

export const transposeGradConfig: GradConfig = {
  kernelName: Transpose,
  gradFunc: (dy: Tensor, saved: Tensor[], attrs: NamedAttrMap) => {
    const transposeAttrs: TransposeAttrs = attrs as {} as TransposeAttrs;
    const {perm} = transposeAttrs;
    const undoPerm = axis_util.getUndoAxesPermutation(perm);
    return {x: () => transpose(dy, undoPerm)};
  }
};
