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

import * as tfc from '@tensorflow/tfjs-core';

/** Smoke tests run in PR and nightly builds. */
export const SMOKE = '#SMOKE';
/** Layers tests run in layers-related PR builds. */
export const LAYERS = '#LAYERS';
/** Layers tests run in layers-related PR builds. */
export const REGRESSION = '#REGRESSION';

/** Testing tags. */
export const TAGS = [SMOKE, LAYERS, REGRESSION];

/** Testing backends. */
export const BACKENDS = ['cpu', 'webgl'];

/** Testing models for CUJ: create -> save -> predict. */
export const MODELS = [
  'mlp', 'cnn', 'depthwise_cnn', 'simple_rnn', 'gru', 'bidirectional_lstm',
  'time_distributed_lstm', 'one_dimensional', 'functional_merge'
];

/** Local server address for testing browser to access local files. */
export const LOCAL_SERVER = 'http://127.0.0.1:8080';

/**
 * Create a list of input tensors.
 * @param inputsData An array with each element being the value to create a
 *    tensor.
 * @param inputsShapes An array with each element being the shape to create a
 *    tensor.
 */
export function createInputTensors(
    inputsData: tfc.TypedArray[], inputsShapes: number[][]) {
  const xs = [];
  for (let i = 0; i < inputsData.length; i++) {
    const input = tfc.tensor(inputsData[i], inputsShapes[i]);
    xs.push(input);
  }

  return xs;
}
