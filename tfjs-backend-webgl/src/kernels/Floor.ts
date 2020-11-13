
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

import {Floor, KernelConfig, TypedArray} from '@tensorflow/tfjs-core';
import {MathBackendWebGL} from '../backend_webgl';
import {unaryKernelFunc} from '../kernel_utils/kernel_funcs_utils';
import {floorImplCPU} from '../kernel_utils/shared';

// import { UnaryOpPackedProgram } from '../unaryop_packed_gpu';

const FLOOR = `return floor(x);`;

const floorWebgl = unaryKernelFunc(FLOOR);

export const floorConfig: KernelConfig = {
  kernelName: Floor,
  backendName: 'webgl',
  kernelFunc: ({inputs, attrs, backend}) => {
    const {x} = inputs;
    const webglBackend = backend as MathBackendWebGL;

    const shouldExecuteOnCPU = webglBackend.shouldExecuteOnCPU([x]);

    if (shouldExecuteOnCPU) {
      const outValues = floorImplCPU(
          webglBackend.texData.get(x.dataId).values as TypedArray, x.dtype);

      const out = webglBackend.makeTensorInfo(x.shape, x.dtype);
      const outData = webglBackend.texData.get(out.dataId);
      outData.values = outValues;
      return out;
    }

    // if (env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
    //   const program = new UnaryOpPackedProgram(x.shape, FLOOR);

    //   return this.packedUnaryOp(x, unary_op.FLOOR, x.dtype) as T;
    // }

    return floorWebgl({inputs, attrs, backend});
  },
};
