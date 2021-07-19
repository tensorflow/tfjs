/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {ExpandDims, ExpandDimsAttrs, ExpandDimsInputs, KernelConfig, KernelFunc, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';
import {reshape} from './Reshape';

export function expandDims(args: {
  inputs: ExpandDimsInputs,
  attrs: ExpandDimsAttrs,
  backend: BackendWasm
}) {
  const {inputs, attrs, backend} = args;
  const {input} = inputs;
  const {dim} = attrs;

  const inputRank = input.shape.length;
  const newShape = input.shape.slice();
  let $dim = dim;
  if (dim < 0) {
    // Negative value is counted from the tail of rank.
    util.assert(
        -(inputRank + 1) <= dim,
        () => `Axis must be in the interval [${- (inputRank + 1)}, ${
            inputRank}]`);
    $dim = inputRank + dim + 1;
  }
  newShape.splice($dim, 0, 1);

  return reshape({inputs: {x: input}, backend, attrs: {shape: newShape}});
}

export const expandDimsConfig: KernelConfig = {
  kernelName: ExpandDims,
  backendName: 'wasm',
  kernelFunc: expandDims as {} as KernelFunc,
};
