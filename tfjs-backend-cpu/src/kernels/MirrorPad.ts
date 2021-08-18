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

import {KernelConfig, KernelFunc, MirrorPad, MirrorPadAttrs, MirrorPadInputs, NumericDataType, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

export function mirrorPad(args: {
  inputs: MirrorPadInputs,
  backend: MathBackendCPU,
  attrs: MirrorPadAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {paddings, mode} = attrs;

  assertNotComplex(x, 'mirrorPad');

  const outShape = paddings.map(
      (p, i) => p[0] /* beforePad */ + x.shape[i] + p[1] /* afterPad */);

  const start = paddings.map(p => p[0]);
  const end = paddings.map((p, i) => p[0] + x.shape[i]);
  const offset = mode === 'reflect' ? 0 : 1;

  const xVals = backend.data.get(x.dataId).values as TypedArray;
  const xRank = x.shape.length;
  const xStrides = util.computeStrides(x.shape);

  const resultSize = util.sizeFromShape(outShape);
  const resultRank = outShape.length;
  const resultStrides = util.computeStrides(outShape);
  const resVals =
      util.getTypedArrayFromDType(x.dtype as NumericDataType, resultSize);

  for (let i = 0; i < resultSize; i++) {
    let coords = util.indexToLoc(i, resultRank, resultStrides);
    for (let i = 0; i < resultRank; i++) {
      if (coords[i] < start[i]) {
        coords[i] = start[i] * 2 - coords[i] - offset;
      } else if (coords[i] >= end[i]) {
        coords[i] = (end[i] - 1) * 2 - coords[i] + offset;
      }
    }
    coords = coords.map((c, i) => c - start[i]);

    const inIndex = util.locToIndex(coords, xRank, xStrides);

    resVals[i] = xVals[inIndex];
  }

  const outId = backend.write(resVals, outShape, x.dtype);

  return {dataId: outId, shape: outShape, dtype: x.dtype};
}

export const mirrorPadConfig: KernelConfig = {
  kernelName: MirrorPad,
  backendName: 'cpu',
  kernelFunc: mirrorPad as {} as KernelFunc
};
