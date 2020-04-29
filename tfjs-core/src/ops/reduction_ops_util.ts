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

import {Tensor} from '../tensor';
import * as axis_util from './axis_util';

/**
 * Gradient helper function for the min and max operations.
 */
export function gradForMinAndMax<T extends Tensor>(
    dy: T, y: T, xOrig: Tensor, origAxes: number[], permutedAxes: number[]) {
  if (y.rank < xOrig.rank) {
    y = y.reshape(axis_util.expandShapeToKeepDim(y.shape, origAxes)) as T;
  }
  if (dy.rank < xOrig.rank) {
    dy = dy.reshape(axis_util.expandShapeToKeepDim(dy.shape, origAxes)) as T;
  }
  return {
    x: () => {
      const dx = dy.mul(xOrig.equal(y).cast(dy.dtype));
      return permutedAxes == null ? dx : dx.transpose(permutedAxes);
    }
  };
}
