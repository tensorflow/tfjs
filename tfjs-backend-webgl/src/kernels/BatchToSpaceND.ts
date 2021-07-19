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

import {backend_util, BatchToSpaceND, BatchToSpaceNDAttrs, BatchToSpaceNDInputs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';

import {reshape} from './Reshape';
import {slice} from './Slice';
import {transpose} from './Transpose';

export const batchToSpaceND = (args: {
  inputs: BatchToSpaceNDInputs,
  backend: MathBackendWebGL,
  attrs: BatchToSpaceNDAttrs
}): TensorInfo => {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {blockShape, crops} = attrs;

  util.assert(
      x.shape.length <= 4,
      () => 'batchToSpaceND for rank > 4 with a WebGL backend not ' +
          'implemented yet');
  const prod = blockShape.reduce((a, b) => a * b);

  const reshaped = backend_util.getReshaped(x.shape, blockShape, prod);
  const permuted = backend_util.getPermuted(reshaped.length, blockShape.length);
  const reshapedPermuted =
      backend_util.getReshapedPermuted(x.shape, blockShape, prod);
  const sliceBeginCoords =
      backend_util.getSliceBeginCoords(crops, blockShape.length);
  const sliceSize =
      backend_util.getSliceSize(reshapedPermuted, crops, blockShape.length);

  const toDispose = [];

  const reshapedIntermediate =
      reshape({inputs: {x}, backend, attrs: {shape: reshaped}});
  const transposedIntermediate = transpose(
      {inputs: {x: reshapedIntermediate}, backend, attrs: {perm: permuted}});
  const reshapedIntermediate2 = reshape({
    inputs: {x: transposedIntermediate},
    backend,
    attrs: {shape: reshapedPermuted}
  });
  const sliced = slice({
    inputs: {x: reshapedIntermediate2},
    backend,
    attrs: {begin: sliceBeginCoords, size: sliceSize}
  });

  toDispose.push(reshapedIntermediate);
  toDispose.push(transposedIntermediate);
  toDispose.push(reshapedIntermediate2);

  toDispose.forEach(t => backend.disposeIntermediateTensorInfo(t));

  return sliced;
};

export const batchToSpaceNDConfig: KernelConfig = {
  kernelName: BatchToSpaceND,
  backendName: 'webgl',
  kernelFunc: batchToSpaceND as {} as KernelFunc
};
