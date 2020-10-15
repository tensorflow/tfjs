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

import {backend_util, KernelConfig, Mean, MeanAttrs, MeanInputs, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';

import {meanImpl} from './Mean_impl';
import {transposeImpl, transposeImplCPU} from './Transpose_impl';

export const meanConfig: KernelConfig = {
  kernelName: Mean,
  backendName: 'webgl',
  kernelFunc: ({inputs, attrs, backend}) => {
    const {x} = inputs as MeanInputs;
    const {keepDims, axis} = attrs as {} as MeanAttrs;
    const webglBackend = backend as MathBackendWebGL;

    const xRank = x.shape.length;
    const origAxes = util.parseAxisParam(axis, x.shape);

    let axes = origAxes;
    const permutedAxes = backend_util.getAxesPermutation(axes, xRank);
    const meanInputIsTransposed = permutedAxes != null;
    const shouldExecuteOnCPU = webglBackend.shouldExecuteOnCPU([x]);

    const intermediates: TensorInfo[] = [];

    let meanInput = x;
    if (meanInputIsTransposed) {
      if (shouldExecuteOnCPU) {
        const xTexData = webglBackend.texData.get(meanInput.dataId);
        const values = xTexData.values as TypedArray;

        const newShape: number[] = new Array(xRank);
        for (let i = 0; i < newShape.length; i++) {
          newShape[i] = x.shape[permutedAxes[i]];
        }
        const meanInputValues =
            transposeImplCPU(values, x.shape, x.dtype, permutedAxes, newShape);

        meanInput = webglBackend.makeTensorInfo(newShape, x.dtype);
        const meanInputData = webglBackend.texData.get(meanInput.dataId);
        meanInputData.values = meanInputValues;
      } else {
        meanInput = transposeImpl(x, permutedAxes, webglBackend);
      }

      intermediates.push(meanInput);
      axes = backend_util.getInnerMostAxes(axes.length, xRank);
    }

    backend_util.assertAxesAreInnerMostDims('sum', axes, xRank);
    const [meanOutShape, reduceShape] =
        backend_util.computeOutAndReduceShapes(meanInput.shape, axes);

    let outShape = meanOutShape;
    if (keepDims) {
      // rather than reshape at the end, set the target shape here.
      outShape = backend_util.expandShapeToKeepDim(meanOutShape, origAxes);
    }

    const out = meanImpl(meanInput, reduceShape, outShape, webglBackend);
    for (const i of intermediates) {
      webglBackend.disposeIntermediateTensorInfo(i);
    }

    return out;
  }
};
