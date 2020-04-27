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

import {Max, MaxAttrs, MaxInputs} from '@tensorflow/tfjs-core';
import {backend_util, KernelConfig, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';

import {maxImpl, maxImplCPU} from './Max_impl';
import {transposeImpl} from './Transpose_impl';

export const maxConfig: KernelConfig = {
  kernelName: Max,
  backendName: 'webgl',
  kernelFunc: ({inputs, attrs, backend}) => {
    const {x} = inputs as MaxInputs;
    const {reductionIndices} = attrs as {} as MaxAttrs;
    const webglBackend = backend as MathBackendWebGL;

    const xRank = x.shape.length;

    const origAxes = util.parseAxisParam(reductionIndices, x.shape);
    let axes = origAxes;
    const permutedAxes = backend_util.getAxesPermutation(axes, xRank);
    const maxInputIsTransposed = permutedAxes != null;

    let maxInput = x;
    if (maxInputIsTransposed) {
      maxInput = transposeImpl(x, permutedAxes, webglBackend);
      axes = backend_util.getInnerMostAxes(axes.length, xRank);
    }

    backend_util.assertAxesAreInnerMostDims('max', axes, xRank);
    const [outShape, reduceShape] =
        backend_util.computeOutAndReduceShapes(maxInput.shape, axes);

    let out;
    if (webglBackend.shouldExecuteOnCPU([x])) {
      const xTexData = webglBackend.texData.get(maxInput.dataId);
      const values = xTexData.values as TypedArray;
      const outValues = maxImplCPU(
          values, util.sizeFromShape(reduceShape), outShape, x.dtype);

      out = webglBackend.makeTensorInfo(outShape, x.dtype);
      const outData = webglBackend.texData.get(out.dataId);
      outData.values = outValues;
    } else {
      out = maxImpl(maxInput, reduceShape, outShape, webglBackend);
    }

    if (maxInputIsTransposed) {
      webglBackend.disposeData(maxInput.dataId);
    }

    return out;
  }
};
