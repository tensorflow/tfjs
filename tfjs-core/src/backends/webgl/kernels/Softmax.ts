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

import {NamedAttrMap, NamedTensorInfoMap, registerKernel, TensorInfo} from '../../../kernel_registry';
import * as axis_util from '../../../ops/axis_util';
// import {parseAxisParam, sizeFromShape} from '../../../util';
import {parseAxisParam} from '../../../util';
import {MathBackendWebGL} from '../backend_webgl';

import {divImpl} from './Div';
import {expImpl} from './Exp';
import {maxImpl} from './Max';
import {subImpl} from './Sub';
import {sumImpl} from './Sum';

interface SoftmaxInputs extends NamedTensorInfoMap {
  x: TensorInfo;
}

interface SoftmaxAttrs extends NamedAttrMap {
  dim: number;
}

registerKernel({
  kernelName: 'Softmax',
  backendName: 'webgl',
  kernelFunc: ({inputs, attrs, backend}) => {
    const {logits} = inputs as SoftmaxInputs;
    const {dim} = attrs as SoftmaxAttrs;
    const webglBackend = backend as MathBackendWebGL;

    const axes = parseAxisParam([dim], logits.shape);

    const [, reduceShape] =
        axis_util.computeOutAndReduceShapes(logits.shape, axes);

    const max = maxImpl(logits, reduceShape, webglBackend);

    const subtracted = subImpl(logits, max, webglBackend);
    const exponentiated = expImpl(subtracted, webglBackend);

    const summed = sumImpl(exponentiated, reduceShape, webglBackend);

    const out = divImpl(exponentiated, summed, webglBackend);

    webglBackend.disposeData(max.dataId);
    webglBackend.disposeData(subtracted.dataId);
    webglBackend.disposeData(exponentiated.dataId);
    webglBackend.disposeData(summed.dataId);

    return {dataId: out.dataId, shape: out.shape, dtype: out.dtype};
  }
});
