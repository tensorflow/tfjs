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

import {KernelConfig, KernelFunc, Multinomial, MultinomialAttrs, MultinomialInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {MultinomialProgram} from '../multinomial_gpu';

import {softmax} from './Softmax';

export function multinomial(args: {
  inputs: MultinomialInputs,
  backend: MathBackendWebGL,
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
  const program = new MultinomialProgram(batchSize, numOutcomes, numSamples);
  const customSetup = program.getCustomSetupFunc(seed);

  const res = backend.runWebGLProgram(program, [probs], 'int32', customSetup);
  if (!normalized) {
    backend.disposeIntermediateTensorInfo(probs);
  }
  return res;
}

export const multinomialConfig: KernelConfig = {
  kernelName: Multinomial,
  backendName: 'webgl',
  kernelFunc: multinomial as {} as KernelFunc
};
