/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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

import * as util from '../util';

export function assertParamsConsistent(shapes: number[][], axis: number) {
  const rank = shapes[0].length;
  shapes.forEach((shape, i) => {
    util.assert(
        shape.length === rank,
        () =>
            `Error in concat${rank}D: rank of tensors[${i}] must be the same ` +
            `as the rank of the rest (${rank})`);
  });

  util.assert(
      axis >= 0 && axis < rank,
      () => `Error in concat${rank}D: axis must be between 0 and ${rank - 1}.`);

  const firstShape = shapes[0];
  shapes.forEach((shape, i) => {
    for (let r = 0; r < rank; r++) {
      util.assert(
          (r === axis) || (shape[r] === firstShape[r]),
          () => `Error in concat${rank}D: Shape of tensors[${i}] (${shape}) ` +
              `does not match the shape of the rest (${firstShape}) ` +
              `along the non-concatenated axis ${i}.`);
    }
  });
}

export function computeOutShape(shapes: number[][], axis: number): number[] {
  const outputShape = shapes[0].slice();
  for (let i = 1; i < shapes.length; i++) {
    outputShape[axis] += shapes[i][axis];
  }
  return outputShape;
}
