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
import {parseAxisParam, sizeFromShape} from '../../../util';
import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

import {max} from './max_impl';

interface SoftmaxInputs extends NamedTensorInfoMap {
  x: TensorInfo;
}

interface SoftmaxAttrs extends NamedAttrMap {
  dim: number;
}

registerKernel({
  kernelName: 'Softmax',
  backendName: 'cpu',
  kernelFunc: ({inputs, attrs, backend}) => {
    console.log('BLERG KERNEL FUNC');
    const {logits} = inputs as SoftmaxInputs;
    const {dim} = attrs as SoftmaxAttrs;
    const cpuBackend = backend as MathBackendCPU;
    assertNotComplex(logits, 'softmax');

    const axes = parseAxisParam([dim], logits.shape);

    const [maxLogitOutShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(logits.shape, axes);
    const values = cpuBackend.data.get(logits.dataId).values as Float32Array;
    const outValues = new Float32Array(sizeFromShape(maxLogitOutShape));
    const maxLogit = max(values, reduceShape, outValues);

    const expandedShape =
        axis_util.expandShapeToKeepDim(maxLogitOutShape, axes);

    console.log('MAX LOGIT');
    console.log(maxLogit);

    const dataId = cpuBackend.write(maxLogit, maxLogitOutShape, logits.dtype);
    return {dataId, shape: maxLogitOutShape, dtype: logits.dtype};
  }
});
