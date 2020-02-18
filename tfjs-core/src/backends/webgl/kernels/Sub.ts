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
import {NamedTensorInfoMap, registerKernel, TensorInfo} from '../../../kernel_registry';
import {upcastType} from '../../../types';
import {MathBackendWebGL} from '../backend_webgl';
import * as binaryop_gpu from '../binaryop_gpu';
import {BinaryOpProgram} from '../binaryop_gpu';
import {BinaryOpPackedProgram} from '../binaryop_packed_gpu';

interface SubInputs extends NamedTensorInfoMap {
  a: TensorInfo;
  b: TensorInfo;
}

export const subImpl =
    (a: TensorInfo, b: TensorInfo, backend: MathBackendWebGL): TensorInfo => {
      const dtype = upcastType(a.dtype, b.dtype);
      let program = new BinaryOpProgram(binaryop_gpu.SUB, a.shape, b.shape);
      if (env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
        program = new BinaryOpPackedProgram(binaryop_gpu.SUB, a.shape, b.shape);
      }
      const output = backend.runWebGLProgram(program, [a, b], dtype);
      return output;
    };

registerKernel({
  kernelName: 'Sub',
  backendName: 'webgl',
  kernelFunc: ({inputs, backend}) => {
    const {a, b} = inputs as SubInputs;
    const webglBackend = backend as MathBackendWebGL;
    const out = subImpl(a, b, webglBackend);

    return {dataId: out.dataId, shape: out.shape, dtype: out.dtype};
  }
})
