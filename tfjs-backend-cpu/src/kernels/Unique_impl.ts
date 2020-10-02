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

import {BackendValues, DataType, DataValues, TensorBuffer, TypedArray, util} from '@tensorflow/tfjs-core';

export function uniqueImpl(
    values: BackendValues, axis: number, shape: number[], dtype: DataType): {
  outputValues: BackendValues,
  outputShape: number[],
  indices: BackendValues
} {
  // Normalize and validate axis.
  const $axis = util.parseAxisParam(axis, shape)[0];

  // Calculate the new shape that is suitable for extracting data along the
  // given axis.
  //
  // The rank is 3.
  // The size of the 1st dimension is the size of all the axes < the given axis.
  // The size of the 2nd dimension is the same as the size of the given axis.
  // The size of the 3rd dimension is the size of all the axes > the given axis.
  //
  // For example, for a 4D tensor with shape=[2, 3, 5, 4] and axis=2, the
  // newShape would be: [2*3, 5, 4].
  const newShape = [1, shape[0], 1];
  for (let i = 0; i < $axis; i++) {
    newShape[0] *= shape[i];
  }
  newShape[1] = shape[$axis];
  for (let i = $axis + 1; i < shape.length; i++) {
    newShape[2] *= shape[i];
  }

  // A map from unique elements (their string representations) to their values
  // in "indices" (below).
  const uniqueElements: {[key: string]: number} = {};
  // The indices of each unique element in the original tensor along the given
  // axis. It is 1D and has the same size as the given axis.
  const indices = new Int32Array(shape[$axis]);
  // Create a buffer so we can easily extract value at a given location.
  const inputBuffer = new TensorBuffer(newShape, dtype, values as TypedArray);
  // The indices along the given axis that have unique elements. This is a
  // de-duped version of "indices" above.
  const uniqueIndices: number[] = [];
  const is1DTensor = newShape[0] === 1 && newShape[2] === 1;
  for (let i = 0; i < shape[$axis]; i++) {
    // Extract values along the axis.
    let element: string;
    if (is1DTensor) {
      // Fast path for 1D tensor input.
      element = values[i].toString();
    } else {
      const axisValues = [];
      for (let m = 0; m < newShape[0]; m++) {
        for (let n = 0; n < newShape[2]; n++) {
          axisValues.push(inputBuffer.get(m, i, n));
        }
      }
      element = axisValues.toString();
    }

    // Dedup and update various indices.
    if (uniqueElements[element] !== undefined) {
      indices[i] = uniqueElements[element];
    } else {
      const uniqueIndex = Object.keys(uniqueElements).length;
      uniqueElements[element] = uniqueIndex;
      indices[i] = uniqueIndex;
      uniqueIndices.push(i);
    }
  }

  // Now we know where each of the unique elements are located along the axis
  // (uniqueIndices). Extract them from input buffer and store them in the
  // output buffer.
  const outputTmpShape = newShape.slice();
  outputTmpShape[1] = Object.keys(uniqueElements).length;
  const outputBuffer = new TensorBuffer(outputTmpShape, dtype);
  uniqueIndices.forEach((uniqueElementIndex, i) => {
    for (let m = 0; m < newShape[0]; m++) {
      for (let n = 0; n < newShape[2]; n++) {
        outputBuffer.set(inputBuffer.get(m, uniqueElementIndex, n), m, i, n);
      }
    }
  });

  // The output shape can be calculated from the input shape with the size of
  // the given axis replaced by the number of unique elements along that axis.
  const outputShape = shape.slice();
  outputShape[$axis] = outputTmpShape[1];

  // Convert strings back to bytes if input is a string tensor.
  let outputValues: BackendValues = outputBuffer.values as BackendValues;
  if (dtype === 'string' && util.isString(outputValues[0])) {
    outputValues =
        (outputValues as DataValues as string[]).map(d => util.encodeString(d));
  }

  return {
    outputValues,
    outputShape,
    indices,
  };
}
