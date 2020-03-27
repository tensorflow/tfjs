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

import {env} from '../../../environment';
import {TensorInfo} from '../../../kernel_registry';
import {MathBackendWebGL} from '../backend_webgl';
import {TransposeProgram} from '../transpose_gpu';
import {TransposePackedProgram} from '../transpose_packed_gpu';

export function transposeImpl(
    x: TensorInfo, perm: number[], backend: MathBackendWebGL): TensorInfo {
  const program = env().getBool('WEBGL_PACK_ARRAY_OPERATIONS') ?
      new TransposePackedProgram(x.shape, perm) :
      new TransposeProgram(x.shape, perm);
  return backend.runWebGLProgram(program, [x], x.dtype);
}
