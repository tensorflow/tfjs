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

import {DenseBincount, DenseBincountAttrs, DenseBincountInputs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {BincountProgram} from '../bincount_webgpu';

import {fill} from './Fill';

export function denseBincount(args: {
  inputs: DenseBincountInputs,
  backend: WebGPUBackend,
  attrs: DenseBincountAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x, weights} = inputs;
  const {size, binaryOutput} = attrs;

  const xRankOne = x.shape.length === 1;
  const weightsSize = util.sizeFromShape(weights.shape);
  const hasWeights = weightsSize > 0;
  const dtype = weights.dtype;
  const xSize: [number]|[number, number] =
      xRankOne ? [x.shape[0]] : [x.shape[0], x.shape[1]];
  const outputSize: [number]|[number, number] =
      xRankOne ? [size] : [x.shape[0], size];

  const output = fill({backend, attrs: {shape: outputSize, value: 0, dtype}});
  const program = new BincountProgram(xSize, hasWeights, binaryOutput);
  const uniformData = [{type: 'int32', data: [size]}];
  const bincountInputs: TensorInfo[] = hasWeights ? [x, weights] : [x];
  const res = backend.runWebGPUProgram(
      program, bincountInputs, dtype, uniformData, output);

  return res;
}

export const denseBincountConfig: KernelConfig = {
  kernelName: DenseBincount,
  backendName: 'webgpu',
  kernelFunc: denseBincount as unknown as KernelFunc
};
