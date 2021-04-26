/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {DataType, TypedArray, util} from '@tensorflow/tfjs-core';

export function sparseFillEmptyRowsImpl(
    indices: TypedArray, indicesShape: number[], indicesDType: DataType,
    values: number[], denseShape: number[], defaultValue: number):
    [TypedArray, number[], number[], boolean[], number[]] {
  const N = indicesShape[0];
  const denseRows = denseShape[0];

  const emptyRowIndicator: boolean[] = new Array(denseRows);
  const reverseIndexMap: number[] = new Array(N);

  const rank = indicesShape[1];

  if (denseRows === 0) {
    if (N !== 0) {
      throw new Error(`Received SparseTensor with denseShape[0] = 0 but
         indices.shape[0] = ${N}`);
    }
    const outputIndices =
        util.getArrayFromDType(indicesDType, 0 * rank) as TypedArray;
    const outputValues: number[] = [];
    return [
      outputIndices, [0, rank], outputValues, emptyRowIndicator, reverseIndexMap
    ];
  }

  let rowsAreOrdered = true;
  let lastIndicesRow = 0;
  const csrOffset: number[] = new Array(denseRows).fill(0);

  for (let i = 0; i < N; ++i) {
    // indices is a 2d tensor with shape of [N, rank]
    const row = indices[i * rank + 0];
    if (row < 0) {
      throw new Error(`indices(${i}, 0) is invalid: ${row} < 0`);
    }
    if (row >= denseRows) {
      throw new Error(`indices(${i}, 0) is invalid: ${row} >= ${denseRows}`);
    }
    ++csrOffset[row];
    rowsAreOrdered = rowsAreOrdered && (row >= lastIndicesRow);
    lastIndicesRow = row;
  }

  let allRowsFull = true;
  for (let row = 0; row < denseRows; ++row) {
    // csrOffset here describes the number of elements in this dense row
    const rowEmpty = (csrOffset[row] === 0);
    emptyRowIndicator[row] = rowEmpty;
    allRowsFull = allRowsFull && !rowEmpty;
    // In filled version, each row has at least one element.
    csrOffset[row] = Math.max(csrOffset[row], 1);
    // Update csrOffset to represent the number of elements up to and
    // including denseRows + 1:
    //  csrOffset[0] == #{elements of row 0}
    //  csrOffset[1] == #{elements of row 1} + #{elements of row 0}
    //  ..
    //  csrOffset[i] == starting index for elements in row i + 1.
    if (row > 0) {
      csrOffset[row] += csrOffset[row - 1];
    }
  }

  if (allRowsFull && rowsAreOrdered) {
    const outputIndices: TypedArray = indices;
    const outputValues: number[] = values;
    for (let i = 0; i < N; ++i) {
      reverseIndexMap[i] = i;
    }
    return [
      outputIndices, [N, rank], outputValues, emptyRowIndicator, reverseIndexMap
    ];
  } else {
    const nFull = csrOffset[denseRows - 1];
    const outputIndices =
        util.getArrayFromDType(indicesDType, nFull * rank) as TypedArray;
    const outputValues: number[] = new Array(nFull);
    const filledCount: number[] = new Array(denseRows).fill(0);

    // Fill in values for rows that are not missing
    for (let i = 0; i < N; ++i) {
      // indices is a 2d tensor with shape of [N, rank]
      const row = indices[i * rank + 0];
      const offset = filledCount[row];
      const outputI = ((row === 0) ? 0 : csrOffset[row - 1]) + offset;
      filledCount[row]++;  // Increment the filled count for this row.
      for (let j = 0; j < rank; ++j) {
        // indices and outputIndices are 2d tensors with shape of [N, rank]
        outputIndices[outputI * rank + j] = indices[i * rank + j];
      }
      outputValues[outputI] = values[i];
      // We'll need this reverse index map to backprop correctly.
      reverseIndexMap[i] = outputI;
    }

    // Fill in values for rows that are missing
    for (let row = 0; row < denseRows; ++row) {
      const rowCount = filledCount[row];
      if (rowCount === 0) {  // We haven't filled this row
        const startingIndex = (row === 0) ? 0 : csrOffset[row - 1];
        // Remaining index values were set to zero already.
        // Just need to set the row index in the right location.
        // outputIndices is a 2d tensor with shape of [N, rank]

        outputIndices[startingIndex * rank + 0] = row;
        for (let col = 1; col < rank; ++col) {
          outputIndices[startingIndex * rank + col] = 0;
        }
        outputValues[startingIndex] = defaultValue;
      }
    }
    return [
      outputIndices, [N, rank], outputValues, emptyRowIndicator, reverseIndexMap
    ];
  }
}
