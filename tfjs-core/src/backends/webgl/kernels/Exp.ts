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
import {MathBackendWebGL} from '../backend_webgl';
import * as unaryop_gpu from '../unaryop_gpu';
import {UnaryOpProgram} from '../unaryop_gpu';
import {UnaryOpPackedProgram} from '../unaryop_packed_gpu';

interface ExpInputs extends NamedTensorInfoMap {
  x: TensorInfo;
}

export const expImpl =
    (x: TensorInfo, backend: MathBackendWebGL): TensorInfo => {
      let program = new UnaryOpProgram(x.shape, unaryop_gpu.EXP);
      if (env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
        program = new UnaryOpPackedProgram(x.shape, unaryop_gpu.EXP);
      }
      const output = backend.runWebGLProgram(program, [x], x.dtype);
      return output;
    };

registerKernel({
  kernelName: 'Exp',
  backendName: 'webgl',
  kernelFunc: ({inputs, backend}) => {
    const {x} = inputs as ExpInputs;
    const webglBackend = backend as MathBackendWebGL;
    const out = expImpl(x, webglBackend);

    return {dataId: out.dataId, shape: out.shape, dtype: out.dtype};
  }
});
