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

import {backend_util, KernelConfig, KernelFunc, Softmax, SoftmaxAttrs, SoftmaxInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

import {exp} from './Exp';
import {max} from './Max';
import {div} from './RealDiv';
import {reshape} from './Reshape';
import {sub} from './Sub';
import {sum} from './Sum';

export function softmax(
    args:
        {inputs: SoftmaxInputs, backend: MathBackendCPU, attrs: SoftmaxAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {logits} = inputs;
  const {dim} = attrs;

  const logitsRank = logits.shape.length;

  let $dim = dim;
  if ($dim === -1) {
    $dim = logitsRank - 1;
  }
  if ($dim !== logitsRank - 1) {
    throw Error(
        'Softmax along a non-last dimension is not yet supported. ' +
        `Logits was rank ${logitsRank} and dim was ${$dim}`);
  }

  const axes = util.parseAxisParam([$dim], logits.shape);
  const maxLogit = max({
    inputs: {x: logits},
    backend,
    attrs: {reductionIndices: axes, keepDims: false}
  });
  const expandedShape = backend_util.expandShapeToKeepDim(maxLogit.shape, axes);

  const maxLogitReshaped =
      reshape({inputs: {x: maxLogit}, backend, attrs: {shape: expandedShape}});
  const a =
      sub({inputs: {a: logits, b: maxLogitReshaped}, backend}) as TensorInfo;
  const b = exp({inputs: {x: a}, backend}) as TensorInfo;
  const sumExp =
      sum({inputs: {x: b}, backend, attrs: {axis: axes, keepDims: false}});
  const sumReshaped =
      reshape({inputs: {x: sumExp}, backend, attrs: {shape: expandedShape}});

  const result = div({inputs: {a: b, b: sumReshaped}, backend}) as TensorInfo;

  backend.disposeIntermediateTensorInfo(maxLogit);
  backend.disposeIntermediateTensorInfo(maxLogitReshaped);
  backend.disposeIntermediateTensorInfo(a);
  backend.disposeIntermediateTensorInfo(b);
  backend.disposeIntermediateTensorInfo(sumExp);
  backend.disposeIntermediateTensorInfo(sumReshaped);

  return result;
}

export const softmaxConfig: KernelConfig = {
  kernelName: Softmax,
  backendName: 'cpu',
  kernelFunc: softmax as {} as KernelFunc
};
