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

import {MathBackendWebGL} from '../backend_webgl';

import {exp} from './Exp';
import {max} from './Max';
import {realDiv} from './RealDiv';
import {reshape} from './Reshape';
import {sub} from './Sub';
import {sum} from './Sum';

export function softmax(args: {
  inputs: SoftmaxInputs,
  backend: MathBackendWebGL,
  attrs: SoftmaxAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {logits} = inputs;
  const {dim} = attrs;

  const axes = util.parseAxisParam([dim], logits.shape);

  const maxLogit = max({
    inputs: {x: logits},
    backend,
    attrs: {reductionIndices: axes, keepDims: false}
  });

  const expandedShape = backend_util.expandShapeToKeepDim(maxLogit.shape, axes);

  const maxLogitsReshaped =
      reshape({inputs: {x: maxLogit}, backend, attrs: {shape: expandedShape}});
  const a =
      sub({inputs: {a: logits, b: maxLogitsReshaped}, backend}) as TensorInfo;
  const b = exp({inputs: {x: a}, backend}) as TensorInfo;
  const sumExp =
      sum({inputs: {x: b}, backend, attrs: {axis: axes, keepDims: false}});
  const sumExpReshaped =
      reshape({inputs: {x: sumExp}, backend, attrs: {shape: expandedShape}});

  const res =
      realDiv({inputs: {a: b, b: sumExpReshaped}, backend}) as TensorInfo;

  backend.disposeIntermediateTensorInfo(maxLogit);
  backend.disposeIntermediateTensorInfo(maxLogitsReshaped);
  backend.disposeIntermediateTensorInfo(a);
  backend.disposeIntermediateTensorInfo(b);
  backend.disposeIntermediateTensorInfo(sumExp);
  backend.disposeIntermediateTensorInfo(sumExpReshaped);

  return res;
}

export const softmaxConfig: KernelConfig = {
  kernelName: Softmax,
  backendName: 'webgl',
  kernelFunc: softmax as {} as KernelFunc
};
