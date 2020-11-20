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

import {KernelConfig, KernelFunc, Multinomial, MultinomialAttrs, MultinomialInputs, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';
import * as seedrandom from 'seedrandom';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

import {softmax} from './Softmax';

export function multinomial(args: {
  inputs: MultinomialInputs,
  backend: MathBackendCPU,
  attrs: MultinomialAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {logits} = inputs;
  const {numSamples, seed, normalized} = attrs;

  assertNotComplex(logits, 'multinomial');

  const probabilities = normalized ?
      logits :
      softmax({inputs: {logits}, backend, attrs: {dim: -1}});

  const batchSize = probabilities.shape[0];
  const numEvents = probabilities.shape[1];
  const probVals = backend.data.get(probabilities.dataId).values as TypedArray;
  const resShape = [batchSize, numSamples];
  const resVals =
      util.makeZerosTypedArray(util.sizeFromShape(resShape), 'int32');

  for (let b = 0; b < batchSize; ++b) {
    const offset = b * numEvents;
    // The cdf won't include the last event. It will be implicit if no other
    // event happened.
    const cdf = new Float32Array(numEvents - 1);
    cdf[0] = probVals[offset];
    for (let event = 1; event < cdf.length; ++event) {
      cdf[event] = cdf[event - 1] + probVals[offset + event];
    }

    const random = seedrandom.alea(seed.toString());
    const outOffset = b * numSamples;
    for (let sampleId = 0; sampleId < numSamples; ++sampleId) {
      const r = random();

      // Assume last event happened by default.
      resVals[outOffset + sampleId] = cdf.length;

      for (let event = 0; event < cdf.length; event++) {
        if (r < cdf[event]) {
          resVals[outOffset + sampleId] = event;
          break;
        }
      }
    }
  }

  if (!normalized) {
    backend.disposeIntermediateTensorInfo(probabilities);
  }

  return backend.makeTensorInfo(resShape, 'int32', resVals);
}

export const multinomialConfig: KernelConfig = {
  kernelName: Multinomial,
  backendName: 'cpu',
  kernelFunc: multinomial as {} as KernelFunc
};
