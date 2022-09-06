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

import {backend_util, KernelConfig, KernelFunc, Prod, ProdAttrs, ProdInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {permuteAxesAndTranspose} from './kernel_utils';

import {CppDType} from './types';

let wasmProd: (
    xId: number, reduceSize: number,
    dtype: number, outId: number) => void;

function setup(backend: BackendWasm): void {
  wasmProd = backend.wasm.cwrap(Prod, null /*void*/, [
    'number',
    'number',
    'number',
    'number'
  ]);
}

function prod(args: {
  backend: BackendWasm,
  inputs: ProdInputs,
  attrs: ProdAttrs
}): TensorInfo {
  const {backend, inputs, attrs} = args;
  const {axis, keepDims} = attrs;
  const {x} = inputs;
  const xId = backend.dataIdMap.get(x.dataId).id;
  let inputId = xId;
  let input = x;

  const {transposed, axes, originalAxes, inputWasTransposed} =
      permuteAxesAndTranspose(x, axis, backend);

  let reductionAxes = axes;
  if (inputWasTransposed) {
    const transposedId = backend.dataIdMap.get(transposed.dataId).id;
    if (transposedId !== xId) {
      // transpose was not a no-op. We will need to dispose of this
      // once we are done.
      input = transposed;
      inputId = transposedId;
      reductionAxes = backend_util.getInnerMostAxes(
          reductionAxes.length, input.shape.length);
    }
  }

  backend_util.assertAxesAreInnerMostDims(
      'prod', reductionAxes, input.shape.length);
  const [outShape, reduceShape] =
      backend_util.computeOutAndReduceShapes(input.shape, reductionAxes);
  const reduceSize = util.sizeFromShape(reduceShape);

  const out = backend.makeOutput(outShape, input.dtype);
  if (util.sizeFromShape(input.shape) !== 0) {
    const outId = backend.dataIdMap.get(out.dataId).id;
    wasmProd(inputId, reduceSize, CppDType[out.dtype], outId);
  }

  if (inputWasTransposed) {
    // dispose of the transposed tensor.
    backend.disposeData(transposed.dataId);
  }

  if (keepDims) {
    // reshape
    const newShape = backend_util.expandShapeToKeepDim(out.shape, originalAxes);
    out.shape = newShape;
  }

  return out;
}

export const prodConfig: KernelConfig = {
  kernelName: Prod,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: prod as {} as KernelFunc
};
