/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {buffer, KernelConfig, KernelFunc, TensorInfo, Tile, TileAttrs, TileInputs, TypedArray, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {tileImplCPU} from '../kernel_utils/shared';
import {TileProgram} from '../tile_webgpu';

export function tile(
    params: {inputs: TileInputs, backend: WebGPUBackend, attrs: TileAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = params;
  const {x} = inputs;
  const {reps} = attrs;

  // tile gpu program cannot handle rank >= 5 case.
  if (backend.shouldExecuteOnCPU([x]) || x.dtype === 'string' ||
      x.shape.length >= 5) {
    // Even thought string tensor is always on CPU, just to be consistent on how
    // to access tensor data.
    const data = backend.readSync(x.dataId);
    const value = x.dtype === 'string' ?
        (data as Uint8Array[]).map(d => util.decodeString(d)) :
        data as TypedArray;
    const buf = buffer(x.shape, x.dtype, value);
    const outBuf = tileImplCPU(buf, reps);
    return backend.makeTensorInfo(outBuf.shape, outBuf.dtype, outBuf.values);
  }

  const program = new TileProgram(x.shape, reps);
  const output = backend.runWebGPUProgram(program, [x], x.dtype);

  return output;
}

export const tileConfig: KernelConfig = {
  kernelName: Tile,
  backendName: 'webgpu',
  kernelFunc: tile as {} as KernelFunc,
};
