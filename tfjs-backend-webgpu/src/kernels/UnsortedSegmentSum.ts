/**
 * @license
 * Copyright 2023 Google LLC.
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

import {backend_util, KernelConfig, KernelFunc, TensorInfo, UnsortedSegmentSum, UnsortedSegmentSumAttrs, UnsortedSegmentSumInputs, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {UnsortedSegmentSumProgram} from '../unsorted_segment_sum_webgpu';

import {fill} from './Fill';
import {reshape} from './Reshape';
import {transpose} from './Transpose';

export function unsortedSegmentSum(args: {
  inputs: UnsortedSegmentSumInputs,
  backend: WebGPUBackend,
  attrs: UnsortedSegmentSumAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x, segmentIds} = inputs;
  const {numSegments} = attrs;

  const xRank = x.shape.length;

  const toDispose = [];

  let axis = 0;
  const permutation = backend_util.getAxesPermutation([axis], xRank);
  let permutedX = x;
  if (permutation != null) {
    permutedX = transpose({inputs: {x}, backend, attrs: {perm: permutation}});
    toDispose.push(permutedX);
    axis = backend_util.getInnerMostAxes(1, xRank)[0];
  }

  const outShape = backend_util.segment_util.computeOutShape(
      permutedX.shape, axis, numSegments);
  const inSize = util.sizeFromShape([permutedX.shape[axis]]);
  const a2D =
      reshape({inputs: {x: permutedX}, backend, attrs: {shape: [-1, inSize]}});
  toDispose.push(a2D);

  const dtype = x.dtype;
  const shape = [a2D.shape[0], numSegments];
  const output = fill({backend, attrs: {shape, value: 0, dtype}});
  const program = new UnsortedSegmentSumProgram(a2D.shape, shape, dtype);
  const uniformData = [
    {type: 'int32', data: [numSegments]},
    {type: 'int32', data: [util.sizeFromShape(a2D.shape)]}
  ];
  const segResult = backend.runWebGPUProgram(
      program, [a2D, segmentIds], dtype, uniformData, output);

  const reshaped =
      reshape({inputs: {x: segResult}, backend, attrs: {shape: outShape}});
  toDispose.push(segResult);
  let result = reshaped;
  if (permutation != null) {
    toDispose.push(reshaped);
    const perm = backend_util.getUndoAxesPermutation(permutation);
    result = transpose({inputs: {x: result}, backend, attrs: {perm}});
  }

  toDispose.forEach(t => backend.disposeData(t.dataId));
  return result;
}

export const unsortedSegmentSumConfig: KernelConfig = {
  kernelName: UnsortedSegmentSum,
  backendName: 'webgpu',
  kernelFunc: unsortedSegmentSum as unknown as KernelFunc
};
