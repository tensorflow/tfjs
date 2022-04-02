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

import {backend_util} from '../base';
import {Prod, ProdAttrs} from '../kernel_names';
import {GradConfig, NamedAttrMap} from '../kernel_registry';
import {cumprod} from '../ops/cumprod';
import {mul} from '../ops/mul';
import {reshape} from '../ops/reshape';
import {transpose} from '../ops/transpose';
import {Tensor} from '../tensor';

// Gradient for product operation on a single axis.
function prodGradFn_(x: Tensor, dy: Tensor, axis: number): Tensor {
  // The gradient tensor (dy) has a set of axes removed, so we create re-shaped
  // versions (of size 1) for the removed axis; this supports broadcasting over
  // those dimensions.
  const expandedYShape = x.shape.slice();
  expandedYShape[axis] = 1;

  // The actual gradient computation.
  const expandedDy = reshape(dy, expandedYShape);
  const xCumProd = cumprod(x, axis, true, false);
  const xCumRevProd = cumprod(x, axis, true, true);
  const dx = mul(xCumProd, xCumRevProd);
  return mul(expandedDy, dx);
}

// Support gradients when the product is done on many axes at once.
// This done py pushing all the axes on which the product is applied into a
// single axis.
function prodsGradFn_(x: Tensor, dy: Tensor, axis: number[]): Tensor {
  // Move all axes for doing prod over to the end of the tensor.
  const xRank = x.shape.length;
  const finalProdAxis = xRank - axis.length;
  const xPermutation = backend_util.getAxesPermutation(axis, xRank);
  let permutedX = x;
  if (xPermutation != null) {
    permutedX = transpose(x, xPermutation);
  }

  // Reshape all the prod dimensions into a single one, and do compute prod
  // gradients on that.
  const newShape = permutedX.shape.slice();
  const removedShape = newShape.splice(xRank - axis.length, axis.length);
  const endPartShape = removedShape.reduce((p, c) => p * c, 1);
  newShape.push(endPartShape);
  const reshapedPermutedX = permutedX.reshape(newShape);
  let prodGrad = prodGradFn_(reshapedPermutedX, dy, finalProdAxis);

  // Undo the re-shaping now we have the dx vector, and permute back to
  // original axes order.
  prodGrad = prodGrad.reshape(permutedX.shape);
  if (xPermutation != null) {
    const undoPermutation = backend_util.getUndoAxesPermutation(xPermutation);
    prodGrad = transpose(prodGrad, undoPermutation);
  }
  return prodGrad;
}

// Running example:
// [
//   [
//     [3.0, 4.0],
//     [5.0, 6.0],
//     [7.0, 8.0]
//   ],
//   [
//     [3.0, 5.0],
//     [0.0, 6.0],
//     [5.0, 6.0]
//   ]
// ]
//
export const prodGradConfig: GradConfig = {
  kernelName: Prod,
  inputsToSave: ['x'],
  gradFunc: (dy: Tensor|Tensor[], saved: Tensor[], attrs: NamedAttrMap) => {
    const [x] = saved;
    const {axis} = (attrs as {}) as ProdAttrs;
    let axisArr = [] as number[];
    if (axis === undefined || axis === null) {
      axisArr = x.shape.map((_, i) => i);
    } else if (typeof axis === 'number') {
      axisArr = [axis];
    } else {
      axisArr = axis;
    }
    return {x: () => prodsGradFn_(x, dy as Tensor, axisArr)};
  }
};
