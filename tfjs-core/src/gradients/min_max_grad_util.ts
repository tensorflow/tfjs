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

import * as axis_util from '../ops/axis_util';
import {cast} from '../ops/cast';
import {equal} from '../ops/equal';
import {mul} from '../ops/mul';
import {reshape} from '../ops/reshape';
import {Tensor} from '../tensor';

/**
 * Gradient helper function for the min and max operations.
 */
export function gradForMinAndMax<T extends Tensor>(
    dy: T, y: T, xOrig: Tensor, origAxes: number[]) {
  if (y.rank < xOrig.rank) {
    y = reshape(y, axis_util.expandShapeToKeepDim(y.shape, origAxes)) as T;
  }
  if (dy.rank < xOrig.rank) {
    dy = reshape(dy, axis_util.expandShapeToKeepDim(dy.shape, origAxes)) as T;
  }
  return {
    x: () => {
      const dx = mul(dy, cast(equal(xOrig, y), dy.dtype));
      return dx;
    }
  };
}
