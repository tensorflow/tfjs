/**
 * @license
 * Copyright 2023 Google LLC.
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

import {KernelConfig, KernelFunc, Multinomial, MultinomialAttrs, MultinomialInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {MultinomialProgram} from '../multinomial_webgpu';

import {softmax} from './Softmax';

export function multinomial(args: {
  inputs: MultinomialInputs,
  backend: WebGPUBackend,
  attrs: MultinomialAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {logits} = inputs;
  const {numSamples, seed, normalized} = attrs;

  const probs = normalized ?
      logits :
      softmax(
          {inputs: {logits}, backend, attrs: {dim: logits.shape.length - 1}});
  const batchSize = probs.shape[0];
  const numOutcomes = probs.shape[1];
  const program = new MultinomialProgram(batchSize, numSamples);
  const uniformData =
      [{type: 'float32', data: [seed]}, {type: 'int32', data: [numOutcomes]}];
  const res = backend.runWebGPUProgram(program, [probs], 'int32', uniformData);
  if (!normalized) {
    backend.disposeData(probs.dataId);
  }
  return res;
}

export const multinomialConfig: KernelConfig = {
  kernelName: Multinomial,
  backendName: 'webgpu',
  kernelFunc: multinomial as unknown as KernelFunc
};
