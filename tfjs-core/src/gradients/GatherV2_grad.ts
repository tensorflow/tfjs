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

import {GatherV2, GatherV2Attrs} from '../kernel_names';
import {GradConfig, NamedAttrMap} from '../kernel_registry';
import {getUndoAxesPermutation} from '../ops/axis_util';
import {reshape} from '../ops/reshape';
import {stack} from '../ops/stack';
import {transpose} from '../ops/transpose';
import {unsortedSegmentSum} from '../ops/unsorted_segment_sum';
import {Tensor, Tensor1D} from '../tensor';
import {parseAxisParam} from '../util';

export const gatherGradConfig: GradConfig = {
  kernelName: GatherV2,
  inputsToSave: ['x', 'indices'],
  gradFunc: (dy: Tensor, saved: Tensor[], attrs: NamedAttrMap) => {
    const [x, indices] = saved;
    const {axis, batchDims} = attrs as unknown as GatherV2Attrs;

    const parsedAxis = parseAxisParam(axis, x.shape)[0];

    const derXBatch = (x: Tensor, indices: Tensor, dy: Tensor) => {
      return (): Tensor => {
        const paramsShape = x.shape;
        const indicesSize = indices.size;

        const outerShape = paramsShape.slice(0, parsedAxis);
        const outerDims = outerShape.length;
        const innerShape = paramsShape.slice(axis, paramsShape.length).slice(1);
        const innerDims = innerShape.length;

        const outerAxesIndices = arrayRange(0, outerDims);
        const innerAxesIndices =
            arrayRange(outerDims + 1, outerDims + 1 + innerDims);

        const valuesShape = arrayConcat([outerShape, [indicesSize],
                                         innerShape]);

        const values = reshape(dy, valuesShape);
        const reshapedIndices = reshape(indices, [indicesSize]);

        const transposeDims =
            arrayConcat([[outerDims], outerAxesIndices, innerAxesIndices]);
        const valuesTranspose = transpose(values, transposeDims);
        let paramsGrad = unsortedSegmentSum(
            valuesTranspose, reshapedIndices as Tensor1D, x.shape[parsedAxis]);
        const invertTransposeDims = getUndoAxesPermutation(transposeDims);
        paramsGrad = transpose(paramsGrad, invertTransposeDims);
        return paramsGrad;
      };
    };

    if (batchDims === 1) {
      const batchSize = x.shape[0];
      const xBatch = x.split(batchSize, 0);
      const derXBatched = () => {
        const stacked = stack(
          xBatch.map((x, i) => {
            return derXBatch(x, indices.slice(i,1), dy.slice(i,1))();
          }));
        return stacked.reshape(x.shape);
      };
      return {x: derXBatched, indices: () => indices};
    } else {
      return {x: derXBatch(x, indices, dy), indices: () => indices};
    }
  }
};

function arrayRange(start: number, stop: number): number[] {
  const result = [];
  for (let i = start; i < stop; ++i) {
    result.push(i);
  }
  return result;
}

function arrayConcat(arrays: number[][]): number[] {
  const result = [];
  for (let i = 0; i < arrays.length; ++i) {
    for (let j = 0; j < arrays[i].length; ++j) {
      result.push(arrays[i][j]);
    }
  }
  return result;
}
