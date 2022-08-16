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

import {TensorInfo, TypedArray} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {UnaryOpType} from '../unary_op_util';
import {UnaryOpProgram} from '../unary_op_webgpu';

export function int(input: TensorInfo, backend: WebGPUBackend): TensorInfo {
  if (backend.shouldExecuteOnCPU([input])) {
    const values = backend.tensorMap.get(input.dataId).values as TypedArray;
    const resultValues = Int32Array.from(values);
    return backend.makeTensorInfo(input.shape, 'int32', resultValues);
  }
  const program = new UnaryOpProgram(input.shape, UnaryOpType.TO_INT);
  const output = backend.runWebGPUProgram(program, [input], 'int32');
  return {dataId: output.dataId, shape: output.shape, dtype: output.dtype};
}
