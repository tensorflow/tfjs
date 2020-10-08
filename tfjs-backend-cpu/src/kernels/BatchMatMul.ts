/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {BatchMatMul, BatchMatMulAttrs, BatchMatMulInputs, buffer, KernelConfig, KernelFunc, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

export function batchMatMul(args: {
  inputs: BatchMatMulInputs,
  attrs: BatchMatMulAttrs,
  backend: MathBackendCPU
}) {
  const {inputs, backend, attrs} = args;
  const {a, b} = inputs;
  const {transposeA, transposeB} = attrs;

  assertNotComplex([a, b], 'matMul');

  const sharedDim = transposeA ? a.shape[1] : a.shape[2];
  const leftDim = transposeA ? a.shape[2] : a.shape[1];
  const rightDim = transposeB ? b.shape[1] : b.shape[2];
  const batchDim = a.shape[0];

  const aValues = backend.data.get(a.dataId).values as TypedArray;
  const bValues = backend.data.get(b.dataId).values as TypedArray;

  const aStrides = util.computeStrides(a.shape);
  const bStrides = util.computeStrides(b.shape);

  const [aBatch, aOuterStep, aInnerStep] = transposeA ?
      [aStrides[0], 1, aStrides[1]] :
      [aStrides[0], aStrides[1], 1];
  const [bInnerStep, bOuterStep, bBatch] = transposeB ?
      [1, bStrides[1], bStrides[0]] :
      [bStrides[1], 1, bStrides[0]];

  const size = leftDim * rightDim;
  const result = buffer([batchDim, leftDim, rightDim], a.dtype);
  const resVals = result.values as TypedArray;
  const blockSize = backend.blockSize;

  for (let b = 0; b < batchDim; b++) {
    for (let i0 = 0; i0 < leftDim; i0 += blockSize) {
      for (let j0 = 0; j0 < rightDim; j0 += blockSize) {
        for (let k0 = 0; k0 < sharedDim; k0 += blockSize) {
          // for when blockSize doesn't evenly divide the input
          const iBlock = Math.min(i0 + blockSize, leftDim);
          const jBlock = Math.min(j0 + blockSize, rightDim);
          const kBlock = Math.min(k0 + blockSize, sharedDim);

          for (let i = i0; i < iBlock; i++) {
            for (let j = j0; j < jBlock; j++) {
              let sum = 0.0;

              for (let k = k0; k < kBlock; k++) {
                sum += aValues[b * aBatch + i * aOuterStep + k * aInnerStep] *
                    bValues[k * bInnerStep + j * bOuterStep + b * bBatch];
              }
              resVals[b * size + (i * rightDim + j)] += sum;
            }
          }
        }
      }
    }
  }

  return backend.makeTensorInfo(
      result.shape, result.dtype, result.values as TypedArray);
}

export const batchMatMulConfig: KernelConfig = {
  kernelName: BatchMatMul,
  backendName: 'cpu',
  kernelFunc: batchMatMul as {} as KernelFunc,
};
