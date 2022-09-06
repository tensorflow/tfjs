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

import {BackendValues, DataType, TensorBuffer, TypedArray, util} from '@tensorflow/tfjs-core';

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
  //
  // Note that this is not the final output shape. This will be the shape for an
  // intermediate TensorBuffer (see inputBuffer below) to allow us to extract
  // values along the given axis. To demonstrate how it works, consider the
  // following example:
  //
  // Input: a 3D tensor, with shape [1, 2, 3]
  // [
  //   [
  //      [1,2,3],
  //      [4,5,6]
  //   ]
  // ]
  // Axis: 2 (the last axis).
  // Along axis 2, we expect to extract 3 tensors: [1,4], [2,5], [3,6].
  //
  // For this example, newShape would be: [2, 3, 1], where 2 is calculated from
  // 1*2. The re-shaped data would look like:
  //
  // [
  //   [
  //     [1], [2], [3]
  //   ],
  //   [
  //     [4], [5], [6]
  //   ]
  // ]
  //
  // Then, we can construct a 3-level nested loop by the following dimension
  // order to extract the values along the axis (dimension1):
  // i: dimension1       // 0,1,2 (newShape[1])
  //   m: dimension0     // 0,1   (newShape[0])
  //     n: dimension2   // 0     (newShape[2])
  //
  //                       m, i, n
  //                      ---------
  // Iteration 0: data at [0, 0, 0] => "1"
  // Iteration 1: data at [1, 0, 0] => "4"
  // We got [1,4].
  // Iteration 2: data at [0, 1, 0] => "2"
  // Iteration 3: data at [1, 1, 0] => "5"
  // We got [2,5].
  // Iteration 4: data at [0, 2, 0] => "3"
  // Iteration 5: data at [1, 2, 0] => "6"
  // We got [3,6].
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
      element = axisValues.join(',');
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

  return {
    outputValues: outputBuffer.values as BackendValues,
    outputShape,
    indices,
  };
}
