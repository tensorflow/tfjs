
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
/**
 * This differs from util.assertShapesMatch in that it allows values of
 * negative one, an undefined size of a dimensinon, in a shape to match
 * anything.
 */

import {Tensor, util} from '@tensorflow/tfjs-core';

/**
 * Used by TensorList and TensorArray to verify if elementShape matches, support
 * negative value as the dim shape.
 * @param shapeA
 * @param shapeB
 * @param errorMessagePrefix
 */
export function assertShapesMatchAllowUndefinedSize(
    shapeA: number|number[], shapeB: number|number[],
    errorMessagePrefix = ''): void {
  // constant shape means unknown rank
  if (typeof shapeA === 'number' || typeof shapeB === 'number') {
    return;
  }
  util.assert(
      shapeA.length === shapeB.length,
      () => errorMessagePrefix + ` Shapes ${shapeA} and ${shapeB} must match`);
  for (let i = 0; i < shapeA.length; i++) {
    const dim0 = shapeA[i];
    const dim1 = shapeB[i];
    util.assert(
        dim0 < 0 || dim1 < 0 || dim0 === dim1,
        () =>
            errorMessagePrefix + ` Shapes ${shapeA} and ${shapeB} must match`);
  }
}

export function fullDefinedShape(elementShape: number|number[]): boolean {
  if (typeof elementShape === 'number' || elementShape.some(dim => dim < 0)) {
    return false;
  }
  return true;
}
/**
 * Generate the output element shape from the list elementShape, list tensors
 * and input param.
 * @param listElementShape
 * @param tensors
 * @param elementShape
 */
export function inferElementShape(
    listElementShape: number|number[], tensors: Tensor[],
    elementShape: number|number[]): number[] {
  let partialShape = mergeElementShape(listElementShape, elementShape);
  const notfullDefinedShape = !fullDefinedShape(partialShape);
  if (notfullDefinedShape && tensors.length === 0) {
    throw new Error(
        `Tried to calculate elements of an empty list` +
        ` with non-fully-defined elementShape: ${partialShape}`);
  }
  if (notfullDefinedShape) {
    tensors.forEach(tensor => {
      partialShape = mergeElementShape(tensor.shape, partialShape);
    });
  }
  if (!fullDefinedShape(partialShape)) {
    throw new Error(`Non-fully-defined elementShape: ${partialShape}`);
  }
  return partialShape as number[];
}

export function mergeElementShape(
    elementShapeA: number|number[], elementShapeB: number|number[]): number|
    number[] {
  if (typeof elementShapeA === 'number') {
    return elementShapeB;
  }
  if (typeof elementShapeB === 'number') {
    return elementShapeA;
  }

  if (elementShapeA.length !== elementShapeB.length) {
    throw new Error(`Incompatible ranks during merge: ${elementShapeA} vs. ${
        elementShapeB}`);
  }

  const result: number[] = [];
  for (let i = 0; i < elementShapeA.length; ++i) {
    const dim0 = elementShapeA[i];
    const dim1 = elementShapeB[i];
    if (dim0 >= 0 && dim1 >= 0 && dim0 !== dim1) {
      throw new Error(`Incompatible shape during merge: ${elementShapeA} vs. ${
          elementShapeB}`);
    }
    result[i] = dim0 >= 0 ? dim0 : dim1;
  }
  return result;
}
