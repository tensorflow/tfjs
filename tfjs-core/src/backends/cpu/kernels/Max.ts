/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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

import {Max, MaxAttrs, MaxInputs} from '../../../kernel_names';
import {KernelConfig} from '../../../kernel_registry';
import * as axis_util from '../../../ops/axis_util';
import {TypedArray} from '../../../types';
import * as util from '../../../util';
import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';
import {maxImpl} from './Max_impl';

export const maxConfig: KernelConfig = {
  kernelName: Max,
  backendName: 'cpu',
  kernelFunc: ({inputs, attrs, backend}) => {
    const {x} = inputs as MaxInputs;
    const {reductionIndices} = attrs as {} as MaxAttrs;
    const cpuBackend = backend as MathBackendCPU;
    console.log('max cpu kernel func', x, reductionIndices);

    const origAxes = util.parseAxisParam(reductionIndices, x.shape);
    let axes = origAxes;
    const permutedAxes = axis_util.getAxesPermutation(axes, x.shape.length);
    if (permutedAxes != null) {
      console.log('TRANSPOSE');
    }

    assertNotComplex(x, 'max');
    axis_util.assertAxesAreInnerMostDims('max', axes, x.shape.length);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);

    const reduceSize = util.sizeFromShape(reduceShape);

    const aVals = cpuBackend.data.get(x.dataId).values as TypedArray;
    const result = maxImpl(aVals, reduceSize, outShape, x.dtype);

    const dataId = cpuBackend.write(result, outShape, x.dtype);
    return {dataId, shape: outShape, dtype: x.dtype};
  }
};
