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

import {TensorInfo, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {MergeBlocksRemainderSumProgram, MergeBlocksSumProgram, ScanEachBlockSumProgram, ScanMultipleBlocksProgram, ScanSingleBlockProgram} from '../prefix_sum_webgpu';

const THREADS_PER_BLOCK = 128;
const ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;

function singleBlocksPrefixSum(
    input: TensorInfo, inputOffset: number, elementsCount: number,
    backend: WebGPUBackend): TensorInfo {
  const outputSize = elementsCount + 1;
  const program = new ScanSingleBlockProgram(outputSize, elementsCount);
  const uniformData = [{type: 'int32', data: [inputOffset]}];
  return backend.runWebGPUProgram(program, [input], input.dtype, uniformData);
}

function multipleBlocksPrefixSum(
    input: TensorInfo, blocks: number, backend: WebGPUBackend): TensorInfo {
  let program: ScanEachBlockSumProgram|ScanMultipleBlocksProgram|
      MergeBlocksSumProgram;
  const toDispose = [];

  program = new ScanEachBlockSumProgram(blocks);
  const blockSums = backend.runWebGPUProgram(program, [input], input.dtype);
  toDispose.push(blockSums);
  const elementsCount = blocks * ELEMENTS_PER_BLOCK;
  program = new ScanMultipleBlocksProgram(elementsCount);
  const scanBlocks = backend.runWebGPUProgram(program, [input], input.dtype);
  toDispose.push(scanBlocks);

  const blockSumsThreadsNeeded = Math.ceil(blocks / 2);
  let scanBlockSums: TensorInfo;
  if (blockSumsThreadsNeeded > THREADS_PER_BLOCK) {
    scanBlockSums = arbitraryElementsPrefixSum(blockSums, blocks, backend);
  } else {
    scanBlockSums = singleBlocksPrefixSum(blockSums, 0, blocks, backend);
  }
  toDispose.push(scanBlockSums);

  program = new MergeBlocksSumProgram(input.shape[0] + 1, blocks);
  const out = backend.runWebGPUProgram(
      program, [scanBlocks, scanBlockSums], input.dtype);
  toDispose.forEach(t => backend.disposeData(t.dataId));
  return out;
}

function arbitraryElementsPrefixSum(
    input: TensorInfo, elementsCount: number,
    backend: WebGPUBackend): TensorInfo {
  const remainder = elementsCount % ELEMENTS_PER_BLOCK;
  const blocks = Math.floor(elementsCount / ELEMENTS_PER_BLOCK);
  if (remainder === 0) {
    return multipleBlocksPrefixSum(input, blocks, backend);
  } else if (remainder !== 0 && blocks === 0) {
    return singleBlocksPrefixSum(input, 0, remainder, backend);
  } else {
    const toDispose = [];
    const blocksPrefixSum = multipleBlocksPrefixSum(input, blocks, backend);
    toDispose.push(blocksPrefixSum);
    const remainderPrefixSum = singleBlocksPrefixSum(
        input, blocks * ELEMENTS_PER_BLOCK, remainder, backend);
    toDispose.push(remainderPrefixSum);
    const program =
        new MergeBlocksRemainderSumProgram(elementsCount + 1, blocks);
    const out = backend.runWebGPUProgram(
        program, [blocksPrefixSum, remainderPrefixSum], input.dtype);
    toDispose.forEach(t => backend.disposeData(t.dataId));
    return out;
  }
}

// Inclusive scan. This implementation refers
// http://users.umiacs.umd.edu/~ramani/cmsc828e_gpusci/ScanTalk.pdf
export function prefixSum(
    input: TensorInfo, backend: WebGPUBackend): TensorInfo {
  if (input.shape.length !== 1) {
    throw new Error('WebGPU backend: prefixSum input must be a vector');
  }
  const inputSize = util.sizeFromShape(input.shape);
  return arbitraryElementsPrefixSum(input, inputSize, backend);
}
