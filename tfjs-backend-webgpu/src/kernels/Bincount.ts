/**
 * @license
 * Copyright 2022 Google LLC.
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

import {Bincount, BincountAttrs, BincountInputs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {BincountProgram} from '../bincount_webgpu';

import {fill} from './Fill';
import {slice} from './Slice';

export function bincount(
    args:
        {inputs: BincountInputs, backend: WebGPUBackend, attrs: BincountAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x, weights} = inputs;
  const {size} = attrs;

  const xSize = util.sizeFromShape(x.shape);
  const weightsSize = util.sizeFromShape(weights.shape);
  const hasWeights = weightsSize > 0;
  const outputSize: [number] = [Math.max(xSize, size)];
  const dtype = weights.dtype;
  const program = new BincountProgram(outputSize, hasWeights);

  const bincountInputs: TensorInfo[] = hasWeights ? [x, weights] : [x];
  const uniformData = [{type: 'int32', data: [size]}];
  let result: TensorInfo;
  result = fill({backend, attrs: {shape: outputSize, value: 0, dtype}});
  result = backend.runWebGPUProgram(
      program, bincountInputs, dtype, uniformData, result);

  if (size === xSize) {
    return result;
  }

  let out: TensorInfo;
  out = slice({inputs: {x: result}, backend, attrs: {begin: 0, size}});
  backend.disposeData(result.dataId);

  return out;
}

export const bincountConfig: KernelConfig = {
  kernelName: Bincount,
  backendName: 'webgpu',
  kernelFunc: bincount as {} as KernelFunc
};
