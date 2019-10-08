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

import {DataId, DataInfo, registerKernel} from '../../kernel_registry';
import {buffer} from '../../ops/array_ops';
import * as broadcast_util from '../../ops/broadcast_util';
import {BackendValues, DataType, TypedArray, upcastType} from '../../types';

export interface CPUStorage {
  readSync(dataId: DataId): BackendValues;
  read(dataId: DataId): Promise<BackendValues>;
  newData(dtype: DataType, values: BackendValues): DataId;
}

function broadcastedBinaryComplexOp(
    a: DataInfo, b: DataInfo, storage: CPUStorage,
    op: (aReal: number, aImag: number, bReal: number, bImag: number) => {
      real: number,
      imag: number
    }): DataInfo {
  const newShape = broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
  const realResult = buffer(newShape, 'float32');
  const imagResult = buffer(newShape, 'float32');

  const aVals = storage.readSync(a.dataId) as TypedArray;
  const bVals = storage.readSync(b.dataId) as TypedArray;
  const aBroadcastDims = broadcast_util.getBroadcastDims(a.shape, newShape);
  const bBroadcastDims = broadcast_util.getBroadcastDims(b.shape, newShape);

  const realVals = realResult.values;
  const imagVals = imagResult.values;

  if (aBroadcastDims.length + bBroadcastDims.length === 0) {
    for (let i = 0; i < realVals.length; i++) {
      const aIdx = i % aVals.length;
      const bIdx = i % bVals.length;

      const result =
          op(aVals[aIdx * 2], aVals[aIdx * 2 + 1], bVals[bIdx * 2],
             bVals[bIdx * 2 + 1]);

      realVals[i] = result.real;
      imagVals[i] = result.imag;
    }
  } else {
    const aRealBuf = bufferSync(this.data.get(a.dataId).complexTensors.real);
    const bRealBuf = bufferSync(this.data.get(b.dataId).complexTensors.real);
    const aRank = a.shape.length;
    const bRank = b.shape.length;
    for (let i = 0; i < realVals.length; i++) {
      const loc = realResult.indexToLoc(i);

      const aLoc = loc.slice(-aRank);
      aBroadcastDims.forEach(d => aLoc[d] = 0);
      const aIndex = aRealBuf.locToIndex(aLoc);

      const bLoc = loc.slice(-bRank);
      bBroadcastDims.forEach(d => bLoc[d] = 0);
      const bIndex = bRealBuf.locToIndex(bLoc);

      const opResult =
          op(aVals[aIndex * 2], aVals[aIndex * 2 + 1], bVals[bIndex * 2],
             bVals[bIndex * 2 + 1]);

      realVals[i] = opResult.real;
      imagVals[i] = opResult.imag;
    }
  }
  return this.complex(realResult.toTensor(), imagResult.toTensor());
}

function broadcastedBinaryOp(
    a: DataInfo, b: DataInfo, outDtype: DataType, storage: CPUStorage,
    op: (a: number, b: number) => number): DataInfo {
  const outShape = broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
  const out = buffer(outShape, outDtype);
  const aVals = storage.readSync(a.dataId) as TypedArray;
  const bVals = storage.readSync(b.dataId) as TypedArray;

  const aBroadcastDims = broadcast_util.getBroadcastDims(a.shape, outShape);
  const bBroadcastDims = broadcast_util.getBroadcastDims(b.shape, outShape);

  const outVals = out.values;
  if (aBroadcastDims.length + bBroadcastDims.length === 0) {
    for (let i = 0; i < outVals.length; ++i) {
      outVals[i] = op(aVals[i % aVals.length], bVals[i % bVals.length]);
    }
  } else {
    const aBuf = buffer(a.shape, a.dtype, aVals);
    const bBuf = buffer(b.shape, b.dtype, bVals);
    const oBuf = out;
    const aRank = a.shape.length;
    const bRank = b.shape.length;
    for (let i = 0; i < outVals.length; ++i) {
      const loc = oBuf.indexToLoc(i);

      const aLoc = loc.slice(-aRank);
      aBroadcastDims.forEach(d => aLoc[d] = 0);
      const aIndex = aBuf.locToIndex(aLoc);

      const bLoc = loc.slice(-bRank);
      bBroadcastDims.forEach(d => bLoc[d] = 0);
      const bIndex = bBuf.locToIndex(bLoc);

      outVals[i] = op(aVals[aIndex], bVals[bIndex]);
    }
  }
  const outId = storage.newData(outDtype, out.values as BackendValues);
  return {dataId: outId, shape: outShape, dtype: outDtype};
}

registerKernel('Mul', 'cpu', ({inputs, storage}) => {
  const {a, b} = inputs;

  if (a.dtype === 'complex64' || b.dtype === 'complex64') {
    return broadcastedBinaryComplexOp(
        a.cast('complex64'), b.cast('complex64'), storage as CPUStorage,
        (aReal, aImag, bReal, bImag) => {
          return {
            real: aReal * bReal - aImag * bImag,
            imag: aReal * bImag + aImag * bReal
          };
        });
  }

  const outDtype = upcastType(a.dtype, b.dtype);
  return broadcastedBinaryOp(
      a, b, outDtype, storage as CPUStorage,
      (aVal: number, bVal: number) => aVal * bVal);
});
