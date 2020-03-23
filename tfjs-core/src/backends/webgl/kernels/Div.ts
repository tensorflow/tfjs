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

import {assertAndGetBroadcastShape} from '../../../../src/ops/broadcast_util';
import {env} from '../../../environment';
import {BinaryInputs, Div} from '../../../kernel_names';
import {KernelConfig, TensorInfo} from '../../../kernel_registry';
import {MathBackendWebGL} from '../backend_webgl';
import * as binaryop_gpu from '../binaryop_gpu';
import {BinaryOpProgram} from '../binaryop_gpu';
import * as binaryop_packed_gpu from '../binaryop_packed_gpu';
import {BinaryOpPackedProgram} from '../binaryop_packed_gpu';

export const divImpl =
    (a: TensorInfo, b: TensorInfo, out: TensorInfo,
     backend: MathBackendWebGL): TensorInfo => {
      let program = new BinaryOpProgram(binaryop_gpu.DIV, a.shape, b.shape);
      if (env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
        program = new BinaryOpPackedProgram(
            binaryop_packed_gpu.DIV, a.shape, b.shape, true);
      }
      const output = backend.runWebGLProgram(program, [a, b], 'float32', out);
      return output;
    };

export const divConfig: KernelConfig = {
  kernelName: Div,
  backendName: 'webgl',
  kernelFunc: ({inputs, backend}) => {
    const {a, b} = inputs as BinaryInputs;

    const webglBackend = backend as MathBackendWebGL;

    const outShape = assertAndGetBroadcastShape(a.shape, b.shape);
    const outTensorInfo = webglBackend.makeTensorInfo(outShape, a.dtype);

    const out = divImpl(a, b, outTensorInfo, webglBackend);

    return {dataId: out.dataId, shape: out.shape, dtype: out.dtype};
  }
};
