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

import {ArgMin, ArgMinAttrs, ArgMinInputs, backend_util, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {argMinMaxReduce} from '../kernel_utils/arg_min_max';
import {transpose} from './Transpose';

export function argMin(
    args:
        {inputs: ArgMinInputs, backend: MathBackendWebGL, attrs: ArgMinAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {axis} = attrs;

  let axes = util.parseAxisParam(axis, x.shape);
  const permutedAxes = backend_util.getAxesPermutation(axes, x.shape.length);
  let $x = x;
  const intermediateTensorInfos = [];
  if (permutedAxes != null) {
    $x = transpose({inputs: {x}, backend, attrs: {perm: permutedAxes}});
    intermediateTensorInfos.push($x);
    axes = backend_util.getInnerMostAxes(axes.length, $x.shape.length);
  }

  backend_util.assertAxesAreInnerMostDims('argMin', [axes[0]], $x.shape.length);

  const out = argMinMaxReduce(backend, $x, axes[0], 'min');

  intermediateTensorInfos.forEach(
      t => backend.disposeIntermediateTensorInfo(t));
  return out;
}

export const argMinConfig: KernelConfig = {
  kernelName: ArgMin,
  backendName: 'webgl',
  kernelFunc: argMin as {} as KernelFunc
};
