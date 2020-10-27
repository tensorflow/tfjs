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

import {Min, MinAttrs} from '../kernel_names';
import {GradConfig, NamedAttrMap} from '../kernel_registry';
import {Tensor} from '../tensor';
import * as util from '../util';

import {gradForMinAndMax} from './min_max_grad_util';

export const minGradConfig: GradConfig = {
  kernelName: Min,
  inputsToSave: ['x'],
  outputsToSave: [true],
  gradFunc: (dy: Tensor, saved: Tensor[], attrs: NamedAttrMap) => {
    const minAttrs: MinAttrs = attrs as {} as MinAttrs;
    const {axis} = minAttrs;
    const [x, y] = saved;
    const origAxes = util.parseAxisParam(axis, x.shape);
    const minGrad = gradForMinAndMax(dy, y, x, origAxes);
    return {
      x: () => {
        return minGrad['x']();
      }
    };
  }
};
