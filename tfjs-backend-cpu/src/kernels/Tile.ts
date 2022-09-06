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

import {KernelConfig, KernelFunc, TensorInfo, Tile, TileAttrs, TileInputs} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';
import {tileImpl} from './Tile_impl';

export function tile(
    args: {inputs: TileInputs, backend: MathBackendCPU, attrs: TileAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {reps} = attrs;

  assertNotComplex(x, 'tile');
  const outBuf = tileImpl(backend.bufferSync(x), reps);

  return backend.makeTensorInfo(outBuf.shape, outBuf.dtype, outBuf.values);
}

export const tileConfig: KernelConfig = {
  kernelName: Tile,
  backendName: 'cpu',
  kernelFunc: tile as {} as KernelFunc
};
