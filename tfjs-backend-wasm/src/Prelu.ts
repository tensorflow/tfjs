/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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

import {registerKernel, util} from '@tensorflow/tfjs-core';
import {TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from './backend_wasm';

interface PreluInputs {
  x: TensorInfo;
  weights: TensorInfo;
}

registerKernel('Prelu', 'wasm', ({inputs, storage}) => {
  const {x, weights} = inputs as {} as PreluInputs;
  const backend = storage as BackendWasm;
  const xId = backend.dataIdMap.get(x.dataId).id;
  const weightsId = backend.dataIdMap.get(weights.dataId).id;

  const out = backend.makeOutput(x.shape, 'float32');
  const outId = backend.dataIdMap.get(out.dataId).id;
  const xSize = util.sizeFromShape(x.shape);
  backend.wasm.tfjs.prelu(xId, xSize, weightsId, outId);
  return out;
});
