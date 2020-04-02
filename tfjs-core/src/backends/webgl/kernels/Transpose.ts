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

import {TypedArray} from '../../../../src/types';
import {transposeImpl as cpuTranspose} from '../../../backends/cpu/kernels/Transpose_impl';
import {Transpose, TransposeAttrs, TransposeInputs} from '../../../kernel_names';
import {KernelConfig, TensorInfo} from '../../../kernel_registry';
import {MathBackendWebGL} from '../backend_webgl';
import {transposeImpl} from './Transpose_impl';

export const transposeConfig: KernelConfig = {
  kernelName: Transpose,
  backendName: 'webgl',
  kernelFunc: ({inputs, attrs, backend}) => {
    const {x} = inputs as TransposeInputs;
    const {perm} = attrs as {} as TransposeAttrs;
    const webglBackend = backend as MathBackendWebGL;

    const xRank = x.shape.length;

    const newShape: number[] = new Array(xRank);
    for (let i = 0; i < newShape.length; i++) {
      newShape[i] = x.shape[perm[i]];
    }

    let out: TensorInfo;
    if (webglBackend.shouldExecuteOnCPU([x])) {
      const xTexData = webglBackend.texData.get(x.dataId);
      const values = xTexData.values as TypedArray;
      const outValues = cpuTranspose(values, x.shape, x.dtype, perm, newShape);

      out = webglBackend.makeTensorInfo(newShape, x.dtype);
      const outData = webglBackend.texData.get(out.dataId);
      outData.values = outValues;
    } else {
      out = transposeImpl(x, perm, webglBackend);
    }
    return out;
  }
};
