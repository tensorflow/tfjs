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

import {maxImpl as cpuMax} from '../../../backends/cpu/kernels/Max_impl';
import {Max, MaxAttrs, MaxInputs} from '../../../kernel_names';
import {KernelConfig} from '../../../kernel_registry';
import * as axis_util from '../../../ops/axis_util';
import {TypedArray} from '../../../types';
import * as util from '../../../util';
import {MathBackendWebGL} from '../backend_webgl';
import {maxImpl} from './Max_impl';

export const maxConfig: KernelConfig = {
  kernelName: Max,
  backendName: 'webgl',
  kernelFunc: ({inputs, attrs, backend}) => {
    const {x} = inputs as MaxInputs;
    const {reductionIndices} = attrs as {} as MaxAttrs;
    const webglBackend = backend as MathBackendWebGL;

    axis_util.assertAxesAreInnerMostDims(
        'max', reductionIndices, x.shape.length);

    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, reductionIndices);

    let out;
    if (webglBackend.shouldExecuteOnCPU([x])) {
      const xTexData = webglBackend.texData.get(x.dataId);
      const values = xTexData.values as TypedArray;
      const outValues =
          cpuMax(values, util.sizeFromShape(reduceShape), outShape, x.dtype);

      out = webglBackend.makeTensorInfo(outShape, x.dtype);
      const outData = webglBackend.texData.get(out.dataId);
      outData.values = outValues;
    } else {
      out = maxImpl(x, reduceShape, outShape, webglBackend);
    }

    return out;
  }
};
