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

import {backend_util, TensorInfo, util} from '@tensorflow/tfjs-core';
import {BackendWasm} from '../backend_wasm';
import {transpose} from './Transpose';

/**
 * Compute permutation axes and do a transpose if necessary.
 *
 * Used by reduction ops.
 * @param x input TensorInfo
 * @param axis reduction axes
 * @param backend wasm backend instance
 */
export function permuteAxesAndTranspose(
    x: TensorInfo, axis: number|number[], backend: BackendWasm): {
  transposed: TensorInfo|null,
  axes: number[],
  originalAxes: number[],
  inputWasTransposed: boolean
} {
  const xShape = x.shape;
  const xRank = x.shape.length;

  const originalAxes = util.parseAxisParam(axis, xShape);
  let axes = originalAxes;
  const permutedAxes = backend_util.getAxesPermutation(axes, xRank);
  let xTransposed = null;
  let inputWasTransposed = false;
  if (permutedAxes != null) {
    const newShape: number[] = new Array(xRank);
    for (let i = 0; i < newShape.length; i++) {
      newShape[i] = xShape[permutedAxes[i]];
    }

    axes = backend_util.getInnerMostAxes(axes.length, xRank);
    xTransposed =
        transpose({inputs: {x}, attrs: {perm: permutedAxes}, backend});

    const xId = backend.dataIdMap.get(x.dataId).id;
    const transposedId = backend.dataIdMap.get(xTransposed.dataId).id;
    if (transposedId !== xId) {
      inputWasTransposed = true;
    }
  }

  return {transposed: xTransposed, originalAxes, axes, inputWasTransposed};
}
