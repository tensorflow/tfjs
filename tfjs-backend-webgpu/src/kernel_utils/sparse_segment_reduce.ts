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

import {TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {fill} from '../kernels/Fill';
import {SparseSegmentIdCountProgram, SparseSegmentMeanProgram, SparseSegmentSumProgram} from '../sparse_segment_reduce_webgpu';
import {WebGPUProgram} from '../webgpu_program';

export function sparseSegmentReduce(
    input: TensorInfo, indices: TensorInfo, segmentIds: TensorInfo,
    isSum = false, backend: WebGPUBackend): TensorInfo {
  const inputSize = util.sizeFromShape(input.shape);
  const segmentSize = inputSize / input.shape[0];
  const dtype = input.dtype;

  // Note that the current implementation assumes that segmentIds values are
  // sorted.
  const numIndices = util.sizeFromShape(indices.shape);
  const $segmentIds = backend.readSync(segmentIds.dataId) as TypedArray;
  const lastSegmentIdPlusOne =
      numIndices > 0 ? $segmentIds[numIndices - 1] + 1 : 0;
  const outputRows = lastSegmentIdPlusOne;

  let program: WebGPUProgram;
  const outputShape = input.shape.slice();
  outputShape[0] = outputRows;

  const sparseSize = numIndices * segmentSize;
  const sparseSegmentSum =
      fill({backend, attrs: {shape: outputShape, value: 0, dtype}});
  program = new SparseSegmentSumProgram(outputShape, sparseSize, dtype);
  let uniformData = [
    {type: 'int32', data: [segmentSize]}, {type: 'int32', data: [sparseSize]}
  ];
  const $sparseSegmentSum = backend.runWebGPUProgram(
      program, [input, indices, segmentIds], dtype, uniformData,
      sparseSegmentSum);

  if (isSum) {
    return $sparseSegmentSum;
  }

  const sparseSegmentIdCount =
      fill({backend, attrs: {shape: [outputRows], value: 0, dtype: 'int32'}});
  program = new SparseSegmentIdCountProgram(outputRows, segmentIds.shape);
  const $sparseSegmentIdCount = backend.runWebGPUProgram(
      program, [segmentIds], 'int32', null, sparseSegmentIdCount);

  const sparseSegmentMean =
      fill({backend, attrs: {shape: outputShape, value: 0, dtype}});
  program = new SparseSegmentMeanProgram(outputShape, dtype);
  uniformData = [{type: 'int32', data: [segmentSize]}];
  const $sparseSegmentMean = backend.runWebGPUProgram(
      program, [$sparseSegmentSum, $sparseSegmentIdCount], dtype, uniformData,
      sparseSegmentMean);

  backend.disposeData($sparseSegmentSum.dataId);
  backend.disposeData($sparseSegmentIdCount.dataId);
  return $sparseSegmentMean;
}
