/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

/** An implementation of topk shared between webgl and cpu backend. */

import {tensor} from '../ops/tensor_ops';
import {Tensor} from '../tensor';
import {getTypedArrayFromDType} from '../util';

export function topkImpl<T extends Tensor>(
    x: T, k: number, sorted: boolean): [T, T] {
  // Reshape into a 2d tensor [batch, lastDim] and compute topk along lastDim.
  const lastDim = x.shape[x.shape.length - 1];
  const x2d = x.reshape([-1, lastDim]);
  const [batch, size] = x2d.shape;
  const allVals = x.dataSync();
  const allTopKVals = getTypedArrayFromDType(x.dtype, batch * k);
  const allTopKIndices = getTypedArrayFromDType('int32', batch * k);

  for (let b = 0; b < batch; b++) {
    const offset = b * size;
    const vals = allVals.subarray(offset, offset + size);
    const valAndInd: Array<{value: number, index: number}> = [];
    for (let i = 0; i < vals.length; i++) {
      valAndInd.push({value: vals[i], index: i});
    }
    valAndInd.sort((a, b) => b.value - a.value);

    const outOffset = b * k;
    const topKVals = allTopKVals.subarray(outOffset, outOffset + k);
    const topKIndices = allTopKIndices.subarray(outOffset, outOffset + k);
    for (let i = 0; i < k; i++) {
      topKVals[i] = valAndInd[i].value;
      topKIndices[i] = valAndInd[i].index;
    }
  }
  // Reshape back to the original input shape, except that the last
  // dimension is k.
  const outputShape = x.shape.slice();
  outputShape[outputShape.length - 1] = k;
  return [
    tensor(allTopKVals, outputShape, x.dtype) as T,
    tensor(allTopKIndices, outputShape, 'int32') as T
  ];
}
