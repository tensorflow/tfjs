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

import {reshape} from './Reshape';

export function batchMatMul(args: {
  inputs: BatchMatMulInputs,
  attrs: BatchMatMulAttrs,
  backend: MathBackendCPU
}) {
  const {inputs, backend, attrs} = args;
  const {a, b} = inputs;
  const {transposeA, transposeB} = attrs;

  assertNotComplex([a, b], 'matMul');

  const aRank = a.shape.length;
  const bRank = b.shape.length;

  const innerShapeA = transposeA ? a.shape[aRank - 2] : a.shape[aRank - 1];
  const innerShapeB = transposeB ? b.shape[bRank - 1] : b.shape[bRank - 2];

  const outerShapeA = transposeA ? a.shape[aRank - 1] : a.shape[aRank - 2];
  const outerShapeB = transposeB ? b.shape[bRank - 2] : b.shape[bRank - 1];

  const outerDimsA = a.shape.slice(0, -2);
  const outerDimsB = b.shape.slice(0, -2);

  util.assert(
      util.arraysEqual(outerDimsA, outerDimsB),
      () => `Error in matMul: outer dimensions (${outerDimsA}) and (` +
          `${outerDimsB}) of Tensors with shapes ${a.shape} and ` +
          `${b.shape} must match.`);

  util.assert(
      innerShapeA === innerShapeB,
      () => `Error in matMul: inner shapes (${innerShapeA}) and (` +
          `${innerShapeB}) of Tensors with shapes ${a.shape} and ` +
          `${b.shape} and transposeA=${transposeA}` +
          ` and transposeB=${transposeB} must match.`);

  const outShape = a.shape.slice(0, -2).concat([outerShapeA, outerShapeB]);

  const batchDimA = util.sizeFromShape(outerDimsA);
  const batchDimB = util.sizeFromShape(outerDimsB);

  const a3dShape = transposeA ? [batchDimA, innerShapeA, outerShapeA] :
                                [batchDimA, outerShapeA, innerShapeA];
  const b3dShape = transposeB ? [batchDimB, outerShapeB, innerShapeB] :
                                [batchDimB, innerShapeB, outerShapeB];

  // The rest of the implementation is designed to operate on rank-3 tensors
  const a3d = reshape({inputs: {x: a}, backend, attrs: {shape: a3dShape}});
  const b3d = reshape({inputs: {x: b}, backend, attrs: {shape: b3dShape}});

  const sharedDim = transposeA ? a3d.shape[1] : a3d.shape[2];
  const leftDim = transposeA ? a3d.shape[2] : a3d.shape[1];
  const rightDim = transposeB ? b3d.shape[1] : b3d.shape[2];
  const batchDim = a3d.shape[0];

  const a3dValues = backend.data.get(a3d.dataId).values as TypedArray;
  const b3dValues = backend.data.get(b3d.dataId).values as TypedArray;

  const a3dStrides = util.computeStrides(a3d.shape);
  const b3dStrides = util.computeStrides(b3d.shape);

  const [aBatch, aOuterStep, aInnerStep] = transposeA ?
      [a3dStrides[0], 1, a3dStrides[1]] :
      [a3dStrides[0], a3dStrides[1], 1];
  const [bInnerStep, bOuterStep, bBatch] = transposeB ?
      [1, b3dStrides[1], b3dStrides[0]] :
      [b3dStrides[1], 1, b3dStrides[0]];

  const size = leftDim * rightDim;
  const result = buffer([batchDim, leftDim, rightDim], a3d.dtype);

  const resVals = result.values as TypedArray;
  const blockSize = backend.blockSize;

  for (let bi = 0; bi < batchDim; bi++) {
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
                sum +=
                    a3dValues[bi * aBatch + i * aOuterStep + k * aInnerStep] *
                    b3dValues[k * bInnerStep + j * bOuterStep + bi * bBatch];
              }
              resVals[bi * size + (i * rightDim + j)] += sum;
            }
          }
        }
      }
    }
  }

  backend.disposeIntermediateTensorInfo(a3d);
  backend.disposeIntermediateTensorInfo(b3d);

  // set correct shape on output.
  return backend.makeTensorInfo(
      outShape, result.dtype, result.values as TypedArray);
}

export const batchMatMulConfig: KernelConfig = {
  kernelName: BatchMatMul,
  backendName: 'cpu',
  kernelFunc: batchMatMul as {} as KernelFunc,
};
