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

import {backend_util, TensorInfo, upcastType, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {mapActivationToShaderProgram} from '../kernel_utils/kernel_funcs_utils';
import {MatMulPackedProgram} from '../mulmat_packed_gpu';

import {multiply} from './Multiply';
import {reshape} from './Reshape';
import {sum} from './Sum';
import {transpose} from './Transpose';

// Empirically determined minimal shared dimension in matmul before we forward
// to a.mul(b).sum() in order to take advantage of GPU parallelism. See
// https://github.com/tensorflow/tfjs-core/pull/1379 for benchmarks.
export const MATMUL_SHARED_DIM_THRESHOLD = 1000;

type BatchMatMulConfig = {
  a: TensorInfo,
  b: TensorInfo,
  transposeA: boolean,
  transposeB: boolean,
  backend: MathBackendWebGL,
  bias?: TensorInfo,
  preluActivationWeights?: TensorInfo,
  leakyreluAlpha?: number,
  activation?: backend_util.Activation
};

export function batchMatMulImpl({
  a,
  b,
  transposeA,
  transposeB,
  backend,
  bias = null,
  preluActivationWeights = null,
  leakyreluAlpha = 0,
  activation = null
}: BatchMatMulConfig): TensorInfo {
  const aRank = a.shape.length;
  const bRank = b.shape.length;

  const innerShapeA = transposeA ? a.shape[aRank - 2] : a.shape[aRank - 1];
  const innerShapeB = transposeB ? b.shape[bRank - 1] : b.shape[bRank - 2];

  const outerShapeA = transposeA ? a.shape[aRank - 1] : a.shape[aRank - 2];
  const outerShapeB = transposeB ? b.shape[bRank - 2] : b.shape[bRank - 1];

  const outerDimsA = a.shape.slice(0, -2);
  const outerDimsB = b.shape.slice(0, -2);

  const batchDimA = util.sizeFromShape(outerDimsA);
  const batchDimB = util.sizeFromShape(outerDimsB);

  const batchDimsCompatible =
      batchDimA === batchDimB || batchDimA === 1 || batchDimB === 1;

  util.assert(
      aRank >= 2 && bRank >= 2 && batchDimsCompatible,
      () => `Error in matMul: the input batch dimensions must either be the ` +
          `same or at least one input batch dimension must be 1. Got input ` +
          `batch dimensions of (${outerDimsA}) and (${outerDimsB}).`);

  const outShapeOuterDims =
      batchDimA > batchDimB ? a.shape.slice(0, -2) : b.shape.slice(0, -2);
  const outShape = outShapeOuterDims.concat([outerShapeA, outerShapeB]);

  util.assert(
      innerShapeA === innerShapeB,
      () => `Error in matMul: inner shapes (${innerShapeA}) and (` +
          `${innerShapeB}) of Tensors with shapes ${a.shape} and ` +
          `${b.shape} and transposeA=${transposeA}` +
          ` and transposeB=${transposeB} must match.`);

  const a3dShape: [number, number, number] = transposeA ?
      [batchDimA, innerShapeA, outerShapeA] :
      [batchDimA, outerShapeA, innerShapeA];
  const b3dShape: [number, number, number] = transposeB ?
      [batchDimB, outerShapeB, innerShapeB] :
      [batchDimB, innerShapeB, outerShapeB];

  // The rest of the implementation is designed to operate on rank-3 tensors
  const a3d = reshape({inputs: {x: a}, backend, attrs: {shape: a3dShape}});
  const b3d = reshape({inputs: {x: b}, backend, attrs: {shape: b3dShape}});

  const intermediates: TensorInfo[] = [a3d, b3d];

  const batchDim = Math.max(batchDimA, batchDimB);
  const sharedDim = transposeA ? a3d.shape[1] : a3d.shape[2];

  const hasBias = bias != null;
  const hasPreluActivationWeights = preluActivationWeights != null;
  const hasLeakyreluAlpha = activation === 'leakyrelu';
  const fusedActivation = activation != null ?
      mapActivationToShaderProgram(activation, true) :
      null;
  const containsFusedOps = hasBias || hasPreluActivationWeights ||
      hasLeakyreluAlpha || fusedActivation != null;
  let out: TensorInfo;

  // Since the matrices are vectors, it is faster to call mul().sum()
  // because sum() is O(sqrt(N)) due to divide-and-conquer.
  if ((outerShapeA === 1 || outerShapeB === 1) &&
      sharedDim > MATMUL_SHARED_DIM_THRESHOLD && containsFusedOps === false) {
    let aVec = a3d;
    let bVec = b3d;
    if (transposeA) {
      aVec = transpose({inputs: {x: a3d}, backend, attrs: {perm: [0, 2, 1]}});
      intermediates.push(aVec);
    }
    if (transposeB) {
      bVec = transpose({inputs: {x: b3d}, backend, attrs: {perm: [0, 2, 1]}});
      intermediates.push(bVec);
    }

    const shouldReshapeA = outerShapeB !== 1;
    const shouldReshapeB = outerShapeB === 1;

    let aVec3d = aVec;
    if (shouldReshapeA) {
      aVec3d = reshape({
        inputs: {x: aVec},
        backend,
        attrs: {shape: [batchDim, sharedDim, 1]}
      });

      intermediates.push(aVec3d);
    }

    const axis = outerShapeB === 1 ? 2 : 1;

    let bVec3d = bVec;
    if (shouldReshapeB) {
      bVec3d = reshape({
        inputs: {x: bVec},
        backend,
        attrs: {shape: [batchDim, 1, sharedDim]}
      });

      intermediates.push(bVec3d);
    }

    const product = multiply({inputs: {a: aVec3d, b: bVec3d}, backend});
    out = sum({inputs: {x: product}, backend, attrs: {axis, keepDims: true}});
    intermediates.push(product);
  } else {
    const dtype = upcastType(a.dtype, b.dtype);

    const program = new MatMulPackedProgram(
        a3dShape, b3dShape, [batchDim, outerShapeA, outerShapeB], transposeA,
        transposeB, hasBias, fusedActivation, hasPreluActivationWeights,
        hasLeakyreluAlpha);

    const inputs: TensorInfo[] = [a3d, b3d];
    if (bias != null) {
      inputs.push(bias);
    }
    if (hasPreluActivationWeights) {
      inputs.push(preluActivationWeights);
    }
    if (hasLeakyreluAlpha) {
      const $leakyreluAlpha = backend.makeTensorInfo(
          [], 'float32',
          util.createScalarValue(leakyreluAlpha as {} as 'float32', 'float32'));
      inputs.push($leakyreluAlpha);
      intermediates.push($leakyreluAlpha);
    }

    out = backend.runWebGLProgram(program, inputs, dtype);
  }

  const outReshaped =
      reshape({inputs: {x: out}, backend, attrs: {shape: outShape}});
  intermediates.push(out);
  for (const i of intermediates) {
    backend.disposeIntermediateTensorInfo(i);
  }
  return outReshaped;
}
