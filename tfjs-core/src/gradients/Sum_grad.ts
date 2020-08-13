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

import {Sum, SumAttrs} from '../kernel_names';
import {GradConfig, NamedAttrMap} from '../kernel_registry';
import {mul} from '../ops/mul';
import {ones} from '../ops/ones';
import {reshape} from '../ops/reshape';
import {Tensor} from '../tensor';
import {parseAxisParam} from '../util';

export const sumGradConfig: GradConfig = {
  kernelName: Sum,
  inputsToSave: ['x'],
  gradFunc: (dy: Tensor, saved: Tensor[], attrs: NamedAttrMap) => {
    const [x] = saved;
    const expandedDyShape = x.shape.slice();
    const {axis} = attrs as {} as SumAttrs;

    const axes = parseAxisParam(axis, x.shape);
    axes.forEach(axis => {
      expandedDyShape[axis] = 1;
    });
    const expandedDy = reshape(dy, expandedDyShape);
    const derX = mul(expandedDy, ones(x.shape, 'float32'));

    return {x: () => derX};
  }
};
