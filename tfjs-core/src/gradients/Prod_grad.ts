/**
 * @license
 * Copyright 2022 Google Inc. All Rights Reserved.
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

import {Prod, ProdAttrs} from '../kernel_names';
import {GradConfig, NamedAttrMap} from '../kernel_registry';
import {div} from '../ops/div';
import {mul} from '../ops/mul';
import {ones} from '../ops/ones';
import {reshape} from '../ops/reshape';
import {Tensor} from '../tensor';
import {parseAxisParam} from '../util';

export const prodGradConfig: GradConfig = {
  kernelName: Prod,
  inputsToSave: ['x'],
  outputsToSave: [true],
  gradFunc: (dy: Tensor, saved: Tensor[], attrs: NamedAttrMap) => {
    const [x, y] = saved;
    const expandedYShape = x.shape.slice();
    const { axis } = (attrs as {}) as ProdAttrs;
    const axes = parseAxisParam(axis, x.shape);
    axes.forEach((axis) => {
      expandedYShape[axis] = 1;
    });
    const expandedY = reshape(y, expandedYShape);
    const expandedDy = reshape(dy, expandedYShape);
    const xFrac = mul(expandedDy, div(ones(x.shape, "float32"), expandedY));
    return { x: () => mul(x, xFrac) };
  }
};
