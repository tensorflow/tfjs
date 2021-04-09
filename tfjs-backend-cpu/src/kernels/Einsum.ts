/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {backend_util, Einsum, EinsumAttrs, EinsumInputs, KernelConfig, KernelFunc, Tensor, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

import {multiply} from './Multiply';
import {reshape} from './Reshape';
import {sum} from './Sum';
import {transpose} from './Transpose';

export function einsum(
    args: {inputs: EinsumInputs, backend: MathBackendCPU, attrs: EinsumAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {equation} = attrs;
  const tensors = inputs as Tensor[];

  const {allDims, summedDims, idDims} =
      backend_util.decodeEinsumEquation(equation, tensors.length);
  backend_util.checkEinsumDimSizes(allDims.length, idDims, tensors);
  const {path, steps} = backend_util.getEinsumComputePath(summedDims, idDims);

  const nSteps = steps.length;
  let out: TensorInfo|null = null;
  let numDimsRemaining = allDims.length;
  const tensorsToDispose: TensorInfo[] = [];
  for (let i = 0; i < nSteps; ++i) {
    for (const idTerm of steps[i]) {
      const {permutationIndices: perm, expandDims: dimsToExpand} =
          backend_util.getEinsumPermutation(numDimsRemaining, idDims[idTerm]);
      let x: TensorInfo;
      if (backend_util.isIdentityPermutation(perm)) {
        x = tensors[idTerm];
      } else {
        x = transpose({inputs: {x: tensors[idTerm]}, backend, attrs: {perm}});
        tensorsToDispose.push(x);
      }
      const targetShape: number[] = x.shape.slice();
      for (let k = 0; k < dimsToExpand.length; ++k) {
        targetShape.splice(dimsToExpand[k], 0, 1);
      }

      if (!util.arraysEqual(x.shape, targetShape)) {
        x = reshape({inputs: {x}, backend, attrs: {shape: targetShape}});
        tensorsToDispose.push(x);
      }
      if (out === null) {
        out = x;
      } else {
        // tslint:disable-next-line: no-unnecessary-type-assertion
        out = multiply({inputs: {a: x, b: out}, backend}) as TensorInfo;
        tensorsToDispose.push(out);
      }
    }
    if (i < nSteps - 1) {
      if (path[i] >= 0) {
        out = sum({
          inputs: {x: out},
          backend,
          attrs: {
            axis: path[i] - (allDims.length - numDimsRemaining),
            keepDims: false
          }
        });
        tensorsToDispose.push(out);
      }
      numDimsRemaining--;
    }
  }

  // Clean up intermediate tensors.
  for (const tensorInfo of tensorsToDispose) {
    if (tensorInfo === out) {
      continue;
    }
    backend.disposeIntermediateTensorInfo(tensorInfo);
  }

  return out;
}

export const einsumConfig: KernelConfig = {
  kernelName: Einsum,
  backendName: 'cpu',
  kernelFunc: einsum as {} as KernelFunc
};
