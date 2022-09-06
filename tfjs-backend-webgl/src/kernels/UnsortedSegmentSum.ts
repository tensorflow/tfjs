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

import {backend_util, DataType, KernelConfig, KernelFunc, sumOutType, TensorInfo, UnsortedSegmentSum, UnsortedSegmentSumAttrs, UnsortedSegmentSumInputs, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {SegmentOpProgram} from '../segment_gpu';

import {range} from './Range';
import {reshape} from './Reshape';
import {tile} from './Tile';
import {transpose} from './Transpose';

export function unsortedSegmentSum(args: {
  inputs: UnsortedSegmentSumInputs,
  backend: MathBackendWebGL,
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

  const outputDType = sumOutType(x.dtype);

  const segOpCompute =
      (x: TensorInfo, segOpType: 'unsortedSegmentSum', segmentIds: TensorInfo,
       dtype: DataType, numSegments: number): TensorInfo => {
        const batchSize = x.shape[0];
        const inSize = x.shape[1];
        const windowSize =
            backend_util.segment_util.segOpComputeOptimalWindowSize(
                inSize, numSegments);
        const segOpInfo = {windowSize, inSize, batchSize, numSegments};
        const program = new SegmentOpProgram(segOpInfo, segOpType);
        const output = backend.compileAndRun(program, [x, segmentIds], dtype);
        toDispose.push(output);
        // No need to run another GPGPU program.
        if (output.shape[1] === numSegments) {
          return output;
        }
        const rangeInfo = range({
          backend,
          attrs: {start: 0, stop: numSegments, step: 1, dtype: 'float32'}
        });
        const tileInfo = tile({
          inputs: {x: rangeInfo},
          backend,
          attrs: {reps: [inSize / windowSize]}
        });

        toDispose.push(rangeInfo);
        toDispose.push(tileInfo);

        const result =
            segOpCompute(output, segOpType, tileInfo, dtype, numSegments);
        return result;
      };

  const segOpResult = segOpCompute(
      a2D, 'unsortedSegmentSum', segmentIds, outputDType, numSegments);

  const reshaped =
      reshape({inputs: {x: segOpResult}, backend, attrs: {shape: outShape}});

  let result = reshaped;
  if (permutation != null) {
    toDispose.push(reshaped);
    const perm = backend_util.getUndoAxesPermutation(permutation);
    result = transpose({inputs: {x: result}, backend, attrs: {perm}});
  }

  toDispose.forEach(t => backend.disposeIntermediateTensorInfo(t));
  return result;
}

export const unsortedSegmentSumConfig: KernelConfig = {
  kernelName: UnsortedSegmentSum,
  backendName: 'webgl',
  kernelFunc: unsortedSegmentSum as {} as KernelFunc
};
