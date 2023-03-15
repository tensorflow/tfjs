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

import {BackendWasm} from '../backend_wasm';
import {softmax} from './Softmax';

let wasmMultinomial: (
    probabilitiesId: number, batchSize: number, numEvents: number,
    numSamples: number, seed: number, outId: number) => void;

function setup(backend: BackendWasm) {
  wasmMultinomial = backend.wasm.cwrap(Multinomial, null, [
    'number',  // probabilitiesId
    'number',  // batchSize
    'number',  // numEvents
    'number',  // numSamples
    'number',  // seed
    'number',  // outId
  ]);
}

export function multinomial(args: {
  inputs: MultinomialInputs,
  attrs: MultinomialAttrs,
  backend: BackendWasm,
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {logits} = inputs;
  const {numSamples, seed, normalized} = attrs;

  if (logits.dtype !== 'float32') {
    throw new Error(
        `Tensor logits must have dtype float32, got ${logits.dtype}`);
  }

  const probabilities = normalized ? logits : softmax({
    inputs: {logits},
    backend,
    attrs: {dim: logits.shape.length - 1},
  });

  const [batchSize, numEvents] = probabilities.shape;
  const out = backend.makeOutput([batchSize, numSamples], 'int32');

  wasmMultinomial(
      backend.dataIdMap.get(probabilities.dataId).id,
      batchSize,
      numEvents,
      numSamples,
      seed,
      backend.dataIdMap.get(out.dataId).id,
  );
  if (!normalized) {
    backend.disposeData(probabilities.dataId);
  }
  return out;
}

export const multinomialConfig: KernelConfig = {
  kernelName: Multinomial,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: multinomial as unknown as KernelFunc
};
