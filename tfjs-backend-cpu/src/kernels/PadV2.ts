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

import {KernelConfig, KernelFunc, NumericDataType, PadV2, PadV2Attrs, PadV2Inputs, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

export function padV2(
    args: {inputs: PadV2Inputs, backend: MathBackendCPU, attrs: PadV2Attrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {paddings, constantValue} = attrs;

  assertNotComplex(x, 'pad');

  const outShape = paddings.map(
      (p, i) => p[0] /* beforePad */ + x.shape[i] + p[1] /* afterPad */);

  const start = paddings.map(p => p[0]);

  const xVals = backend.data.get(x.dataId).values as TypedArray;
  const xSize = util.sizeFromShape(x.shape);
  const xRank = x.shape.length;
  const xStrides = util.computeStrides(x.shape);

  const resultSize = util.sizeFromShape(outShape);
  const resultRank = outShape.length;
  const resultStrides = util.computeStrides(outShape);
  const resVals =
      util.getTypedArrayFromDType(x.dtype as NumericDataType, resultSize);

  if (constantValue !== 0) {
    resVals.fill(constantValue);
  }

  for (let i = 0; i < xSize; i++) {
    const coords = util.indexToLoc(i, xRank, xStrides);
    const outCoords = coords.map((c, i) => c + start[i]);
    const outIndex = util.locToIndex(outCoords, resultRank, resultStrides);

    resVals[outIndex] = xVals[i];
  }

  const outId = backend.write(resVals, outShape, x.dtype);

  return {dataId: outId, shape: outShape, dtype: x.dtype};
}

export const padV2Config: KernelConfig = {
  kernelName: PadV2,
  backendName: 'cpu',
  kernelFunc: padV2 as {} as KernelFunc
};
