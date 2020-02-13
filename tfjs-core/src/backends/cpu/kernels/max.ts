/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {NamedAttrMap, NamedTensorInfoMap, registerKernel, TensorInfo} from '../../../kernel_registry';
import * as axis_util from '../../../ops/axis_util';
import {sizeFromShape} from '../../../util';
import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';
import {max} from './max_impl';

interface MaxInputs extends NamedTensorInfoMap {
  x: TensorInfo;
}

interface MaxAttrs extends NamedAttrMap {
  axes: number[];
}

registerKernel({
  kernelName: 'Max',
  backendName: 'cpu',
  kernelFunc: ({inputs, attrs, backend}) => {
    const {x} = inputs as MaxInputs;
    const {axes} = attrs as MaxAttrs;
    const cpuBackend = backend as MathBackendCPU;

    assertNotComplex(x, 'max');

    axis_util.assertAxesAreInnerMostDims('max', axes, x.shape.length);

    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);

    const xVals = cpuBackend.data.get(x.dataId).values as Float32Array;
    const outValues = new Float32Array(sizeFromShape(outShape));
    const result = max(xVals, reduceShape, outValues);

    const dataId = cpuBackend.write(result, outShape, x.dtype);
    return {dataId, shape: outShape, dtype: x.dtype};
  }
});
