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

import {ArgMin, ArgMinAttrs, ArgMinInputs, backend_util, KernelConfig, KernelFunc, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';
import {transpose} from './Transpose';

export function argMin(
    args: {inputs: ArgMinInputs, backend: MathBackendCPU, attrs: ArgMinAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {axis} = attrs;

  assertNotComplex(x, 'argMin');

  let axes = util.parseAxisParam(axis, x.shape);
  const permutedAxes = backend_util.getAxesPermutation(axes, x.shape.length);
  let $x = x;
  const intermediateTensorInfos = [];
  if (permutedAxes != null) {
    $x = transpose({inputs: {x}, backend, attrs: {perm: permutedAxes}});
    intermediateTensorInfos.push($x);
    axes = backend_util.getInnerMostAxes(axes.length, $x.shape.length);
  }

  axes = [axes[0]];
  backend_util.assertAxesAreInnerMostDims('argMin', axes, $x.shape.length);
  const [outShape, reduceShape] =
      backend_util.computeOutAndReduceShapes($x.shape, axes);

  const outSize = util.sizeFromShape(outShape);
  const vals = util.makeZerosTypedArray(outSize, 'int32');
  const reduceSize = util.sizeFromShape(reduceShape);

  const aVals = backend.data.get($x.dataId).values as TypedArray;
  for (let i = 0; i < vals.length; ++i) {
    const offset = i * reduceSize;
    let min = aVals[offset];
    let minIndex = 0;
    for (let j = 0; j < reduceSize; ++j) {
      const value = aVals[offset + j];
      if (value < min) {
        min = value;
        minIndex = j;
      }
    }
    vals[i] = minIndex;
  }

  intermediateTensorInfos.forEach(
      t => backend.disposeIntermediateTensorInfo(t));

  return backend.makeTensorInfo(outShape, 'int32', vals);
}

export const argMinConfig: KernelConfig = {
  kernelName: ArgMin,
  backendName: 'cpu',
  kernelFunc: argMin as {} as KernelFunc
};
