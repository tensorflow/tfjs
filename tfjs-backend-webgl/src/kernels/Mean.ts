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

import {Mean, MeanAttrs, MeanInputs} from '@tensorflow/tfjs-core';
import {backend_util, KernelConfig, scalar, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {divImpl} from './Div_impl';
import {sumImpl} from './Sum_impl';
import {transposeImpl} from './Transpose_impl';

export const meanConfig: KernelConfig = {
  kernelName: Mean,
  backendName: 'webgl',
  kernelFunc: ({inputs, attrs, backend}) => {
    const {x} = inputs as MeanInputs;
    const {reductionIndices, keepDims} = attrs as {} as MeanAttrs;
    const webglBackend = backend as MathBackendWebGL;

    const origAxes = util.parseAxisParam(reductionIndices, x.shape);
    let axes = origAxes;
    const [sumOutShape, reduceShape] =
        backend_util.computeOutAndReduceShapes(x.shape, axes);
    const reduceSize = util.sizeFromShape(reduceShape);

    const reduceSizeScalar = scalar(reduceSize);
    const res = divImpl(x, reduceSizeScalar, webglBackend);

    const permutedAxes = backend_util.getAxesPermutation(axes, x.shape.length);
    let sumInput = res;
    if (permutedAxes != null) {
      sumInput = transposeImpl(res, permutedAxes, webglBackend);
      axes = backend_util.getInnerMostAxes(axes.length, x.shape.length);
    }

    let outShape = sumOutShape;
    if (keepDims) {
      outShape = backend_util.expandShapeToKeepDim(sumOutShape, origAxes);
    }

    const value = sumImpl(sumInput, reduceShape, outShape, webglBackend);

    webglBackend.disposeData(res.dataId);
    webglBackend.disposeData(reduceSizeScalar.dataId);
    if (permutedAxes != null) {
      webglBackend.disposeData(sumInput.dataId);
    }

    return value;
  }
};
