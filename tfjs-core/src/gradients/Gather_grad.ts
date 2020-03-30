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
import {Gather, GatherAttrs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {GradConfig} from '../kernel_registry';
import {getUndoAxesPermutation} from '../ops/axis_util';
import {unsortedSegmentSum} from '../ops/segment_ops';
import {Tensor1D} from '../tensor';
import {Tensor} from '../tensor';

export const gatherGradConfig: GradConfig = {
  kernelName: Gather,
  inputsToSave: ['x', 'indices'],
  gradFunc: (dy: Tensor, saved: Tensor[], attrs: NamedAttrMap) => {
    const [$x, $indices] = saved;
    const gatherAttrs: GatherAttrs = attrs as {} as GatherAttrs;
    const {axis} = gatherAttrs;
    const derX = () => {
      const paramsShape = $x.shape;
      const indicesSize = $indices.size;

      const outerShape = paramsShape.slice(0, axis);
      const outerDims = outerShape.length;
      const innerShape = paramsShape.slice(axis, paramsShape.length).slice(1);
      const innerDims = innerShape.length;

      const outerAxesIndices = arrayRange(0, outerDims);
      const innerAxesIndices =
          arrayRange(outerDims + 1, outerDims + 1 + innerDims);

      const valuesShape = arrayConcat([outerShape, [indicesSize], innerShape]);

      const values = dy.reshape(valuesShape);
      const reshapedIndices = $indices.reshape([indicesSize]);

      const transposeDims =
          arrayConcat([[outerDims], outerAxesIndices, innerAxesIndices]);
      const valuesTranspose = values.transpose(transposeDims);
      let paramsGrad = unsortedSegmentSum(
          valuesTranspose, reshapedIndices as Tensor1D, $x.shape[axis]);

      const invertTransposeDims = getUndoAxesPermutation(transposeDims);
      paramsGrad = paramsGrad.transpose(invertTransposeDims);

      return paramsGrad;
    };
    return {x: derX, indices: () => $indices};
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
