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

import {Mean, MeanAttrs} from '../kernel_names';
import {GradConfig, NamedAttrMap} from '../kernel_registry';
import {computeOutAndReduceShapes} from '../ops/axis_util';
import {div} from '../ops/div';
import {mul} from '../ops/mul';
import {ones} from '../ops/ones';
import {reshape} from '../ops/reshape';
import {Tensor} from '../tensor';
import * as util from '../util';

export const meanGradConfig: GradConfig = {
  kernelName: Mean,
  inputsToSave: ['x'],
  gradFunc: (dy: Tensor, saved: Tensor[], attrs: NamedAttrMap) => {
    const [x] = saved;
    const {axis} = attrs as {} as MeanAttrs;
    const axes = util.parseAxisParam(axis, x.shape);
    const shapes = computeOutAndReduceShapes(x.shape, axes);
    const reduceShape = shapes[1];
    const reduceSize = util.sizeFromShape(reduceShape);

    const derX = () => {
      const expandedDyShape = x.shape.slice();
      axes.forEach(axis => {
        expandedDyShape[axis] = 1;
      });
      const expandedDy = reshape(dy, expandedDyShape);
      const res = div(mul(expandedDy, ones(x.shape, 'float32')), reduceSize);
      return res;
    };

    return {x: derX};
  }
};
