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

function broadcastedBinaryOp(
    a: DataInfo, b: DataInfo, outDtype: DataType, storage: CPUStorage,
    op: (a: number, b: number) => number) {
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
  const outDtype = upcastType(a.dtype, b.dtype);
  return broadcastedBinaryOp(
      a, b, outDtype, storage as CPUStorage,
      (aVal: number, bVal: number) => aVal * bVal);
});
