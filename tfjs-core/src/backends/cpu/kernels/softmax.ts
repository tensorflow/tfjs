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

import {div} from './div_impl';
import {exp} from './exp_impl';
import {max} from './max_impl';
import {sub} from './sub_impl';
import {sum} from './sum_impl';

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
    const {logits} = inputs as SoftmaxInputs;
    const {dim} = attrs as SoftmaxAttrs;
    const cpuBackend = backend as MathBackendCPU;
    assertNotComplex(logits, 'softmax');

    const axes = parseAxisParam([dim], logits.shape);

    const [reduceOutShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(logits.shape, axes);
    const logitsValues =
        cpuBackend.data.get(logits.dataId).values as Float32Array;
    const maxLogit =
        max(logitsValues, reduceShape,
            new Float32Array(sizeFromShape(reduceOutShape)));

    const expandedShape = axis_util.expandShapeToKeepDim(reduceOutShape, axes);

    console.log('MAX LOGIT');
    console.log(maxLogit);

    const a =
        sub(logitsValues, logits.shape, maxLogit, expandedShape,
            new Float32Array(sizeFromShape(logits.shape)), logits.shape);

    console.log('subtract');
    console.log(a);

    const b = exp(a, new Float32Array(sizeFromShape(logits.shape)));
    console.log('exp');
    console.log(b);

    const sumExp =
        sum(b, reduceShape, new Float32Array(sizeFromShape(reduceOutShape)));
    console.log('sumexp');
    console.log(sumExp);

    const out =
        div(b, logits.shape, sumExp, expandedShape,
            new Float32Array(sizeFromShape(logits.shape)), logits.shape);

    const dataId = cpuBackend.write(out, logits.shape, logits.dtype);
    return {dataId, shape: logits.shape, dtype: logits.dtype};
  }
});
