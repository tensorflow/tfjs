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

import {Softmax, SoftmaxAttrs, SoftmaxInputs} from '../../../kernel_names';
// import {backend_util} from '../../..';
import {KernelConfig} from '../../../kernel_registry';
import * as axis_util from '../../../ops/axis_util';
import {reduceOutShapeFromInShape} from '../../../ops/reduce_util';
// import {parseAxisParam, sizeFromShape} from '../../../util';
import {parseAxisParam} from '../../../util';
import {MathBackendWebGL} from '../backend_webgl';

// import {divImpl} from './Div';
// import {expImpl} from './Exp';
import {maxImpl} from './Max';

// import {subImpl} from './Sub';
// import {sumImpl} from './Sum';

export const softmaxConfig: KernelConfig = {
  kernelName: Softmax,
  backendName: 'webgl',
  kernelFunc: ({inputs, attrs, backend}) => {
    const {logits} = inputs as SoftmaxInputs;
    const {dim} = attrs as {} as SoftmaxAttrs;
    const webglBackend = backend as MathBackendWebGL;

    const axes = parseAxisParam([dim], logits.shape);

    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(logits.shape, axes);

    const maxOutShape = reduceOutShapeFromInShape(logits.shape, reduceShape);

    const maxOutTensorInfo =
        webglBackend.makeTensorInfo(maxOutShape, logits.dtype);
    const max =
        maxImpl(logits, reduceShape, outShape, maxOutTensorInfo, webglBackend);

    // const subtracted = subImpl(logits, max, webglBackend);
    // const exponentiated = expImpl(subtracted, webglBackend);

    // const sumOutTensorInfo =
    //     webglBackend.makeTensorInfo(outShape, logits.dtype);
    // const summed = sumImpl(
    //     exponentiated, reduceShape, outShape, sumOutTensorInfo,
    //     webglBackend);

    // const divOutTensorInfo = webglBackend.makeTensorInfo(
    //     backend_util.assertAndGetBroadcastShape(
    //         exponentiated.shape, summed.shape),
    //     exponentiated.dtype);

    // const out = divImpl(exponentiated, summed, divOutTensorInfo,
    // webglBackend);

    // webglBackend.disposeData(max.dataId);
    // webglBackend.disposeData(subtracted.dataId);
    // webglBackend.disposeData(exponentiated.dataId);
    // webglBackend.disposeData(summed.dataId);

    // return {dataId: out.dataId, shape: out.shape, dtype: out.dtype};
    return {dataId: max.dataId, shape: max.shape, dtype: max.dtype};
  }
};
