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

import {Cumsum, CumsumAttrs} from '../kernel_names';
import {GradConfig, NamedAttrMap} from '../kernel_registry';
import {getAxesPermutation} from '../ops/axis_util';
import {cumsum} from '../ops/cumsum';
import {transpose} from '../ops/transpose';
import {Tensor} from '../tensor';

export const cumsumGradConfig: GradConfig = {
  kernelName: Cumsum,
  inputsToSave: ['x'],
  gradFunc: (dy: Tensor, saved: Tensor[], attrs: NamedAttrMap) => {
    const [x] = saved;
    const {axis, exclusive, reverse}: CumsumAttrs = attrs as {} as CumsumAttrs;

    return {
      x: () => {
        const permutation = getAxesPermutation([axis], x.rank);

        let out = cumsum(dy, axis, exclusive, !reverse);

        if (permutation != null) {
          out = transpose(out, permutation);
        }

        return out;
      }
    };
  }
};
