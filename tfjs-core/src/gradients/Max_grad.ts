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

import {Max, MaxAttrs} from '../kernel_names';
import {GradConfig, NamedAttrMap} from '../kernel_registry';
import * as axis_util from '../ops/axis_util';
import {gradForMinAndMax} from '../ops/reduction_ops_util';
import {transpose} from '../ops/transpose';
import {Tensor} from '../tensor';
import * as util from '../util';

export const maxGradConfig: GradConfig = {
  kernelName: Max,
  inputsToSave: ['x'],
  outputsToSave: [true],
  gradFunc: (dy: Tensor, saved: Tensor[], attrs: NamedAttrMap) => {
    const maxAttrs: MaxAttrs = attrs as {} as MaxAttrs;
    const {reductionIndices} = maxAttrs;
    const [x, y] = saved;
    const origAxes = util.parseAxisParam(reductionIndices, x.shape);
    const permutedAxes = axis_util.getAxesPermutation(origAxes, x.rank);
    const maxGrad = gradForMinAndMax(dy, y, x, origAxes, permutedAxes);
    return {
      x: () => {
        let out = maxGrad['x']();
        if (permutedAxes != null) {
          out = transpose(out);
        }
        return out;
      }
    };
  }
};
