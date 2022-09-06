/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

function validateIndices(
    indices: TypedArray, indicesShape: number[], numParams: number) {
  indices.forEach((index: number, i: number) => {
    if (index < 0 || index >= numParams) {
      const locString =
          util.indexToLoc(
                  i, indicesShape.length, util.computeStrides(indicesShape))
              .join(',');
      throw new Error(
          `indices[${locString}] = ${index} is not in [0, ${numParams})`);
    }
  });
}

function validateSplits(
    paramsNestedSplits: TypedArray[], numParamsDenseValues: number) {
  // Validate
  for (let dim = 0; dim < paramsNestedSplits.length; ++dim) {
    const splits = paramsNestedSplits[dim];
    const lastSplit = (dim === paramsNestedSplits.length - 1) ?
        numParamsDenseValues :
        paramsNestedSplits[dim + 1].length;
    if (splits.length === 0) {
      throw new Error('Ragged splits may not be empty');
    }
    if (splits[0] < 0) {
      throw new Error('Ragged splits must be non-negative');
    }
    if (splits[splits.length - 1] > lastSplit) {
      throw new Error('Ragged splits must not point past values');
    }
    for (let i = 1; i < splits.length; ++i) {
      if (splits[i - 1] > splits[i]) {
        throw new Error('Ragged splits must be sorted in ascending order');
      }
    }
  }
}

// Construct the `splits` output tensors, encoded using a nested vector.
// Also find the slices of values that need to be copied, and store them
// in `valueSlices`.  The total number of values that will be copied (which
// we need for allocating the output values tensor) is stored in `numValues`.
function makeSplits(
    indices: TypedArray, indicesShape: number[],
    paramsNestedSplits: TypedArray[], numParamsDenseValues: number) {
  const valueSlices: Array<[number, number]> = [];
  let numValues = 0;

  const numSplits = indicesShape.length - 1 + paramsNestedSplits.length;
  const outSplits = new Array(numSplits).fill(null).map(() => [0]);

  validateSplits(paramsNestedSplits, numParamsDenseValues);

  // Add `splits` that come from all but the last dimension of the dense
  // Tensor `indices`.  In particular, for each dimension D, we add a
  // splits tensor whose values are:
  //   range(reduceProd(splits.shape[:D]) + 1) * splits.shape[D+1]
  // E.g., if indices.shape=[2, 3, 4] then we will add splits tensors:
  //   [0, 3, 6]                    # length=2+1, stride=3
  //   [0, 4, 8, 12, 16, 20, 24]    # length=2*3+1, stride=4
  let nrows = 1;
  for (let dim = 0; dim < indicesShape.length - 1; ++dim) {
    nrows *= indicesShape[dim];
    const rowLength = indicesShape[dim + 1];
    for (let i = 1; i < nrows + 1; ++i) {
      outSplits[dim].push(i * rowLength);
    }
  }

  // Add `splits` that come from `paramsNestedSplits`.  Starting with the
  // outermost ragged dimension (i.e., the first `splits` tensor), we work
  // our way in, finding the range of values that should be copied.  As we
  // go, we update the output `splits` for each dimension with the appropriate
  // values.  In particular, the *lengths* of the slices from `param_splits`
  // should be copied to generate corresponding slice lengths in the output
  // splits.  E.g., if we are copying a ragged row with length 4, then we
  // should add a new split point to outSplits that is 4 greater than the
  // previous split point in outSplits.
  for (let i = 0; i < indices.length; ++i) {
    let start = indices[i];
    let limit = indices[i] + 1;

    // Copy splits.
    for (let dim = 0; dim < paramsNestedSplits.length; ++dim) {
      const splits = paramsNestedSplits[dim];
      const outDim = dim + indicesShape.length - 1;
      if (outDim >= 0) {
        const outSplitsOutDim = outSplits[outDim];
        const delta =
            outSplitsOutDim[outSplitsOutDim.length - 1] - splits[start];
        for (let j = start; j < limit; ++j) {
          outSplits[outDim].push(splits[j + 1] + delta);
        }
      }
      start = splits[start];
      limit = splits[limit];
    }
    if (limit !== start) {
      valueSlices.push([start, limit]);
      numValues += limit - start;
    }
  }

  return {outSplits, valueSlices, numValues};
}

function getSplits(outSplits: number[][]) {
  const splitsOut: TypedArray[] = [];
  for (let i = 0; i < outSplits.length; ++i) {
    const numSplits = outSplits[i].length;
    const splits = util.getArrayFromDType('int32', numSplits) as TypedArray;
    splitsOut.push(splits);

    outSplits[i].forEach((value, j: number) => splits[j] = value);
  }

  return splitsOut;
}

function computeFlatOuterDims(orig: number[], numOutDims: number) {
  const outDims = orig.slice(0, numOutDims);
  while (outDims.length < numOutDims) {
    outDims.push(1);
  }

  for (let inDim = numOutDims; inDim < orig.length; inDim++) {
    outDims[numOutDims - 1] *= orig[inDim];
  }

  return outDims;
}
// For each slice in `(start, limit)` in `valueSlices`, append
// `paramsDenseValues[start,...,limit] to `values`.  `valueSize` indicates
// the number of scalars contained in each value paramsDenseValues[i].
function writeValueSlices(
    paramsDenseValues: TypedArray, paramsDenseValuesShape: number[],
    valueSlices: Array<[number, number]>, valueSize: number, values: TypedArray,
    valuesShape: number[]) {
  const denseM = computeFlatOuterDims(paramsDenseValuesShape, 2)[1];
  const valuesM = computeFlatOuterDims(valuesShape, 2)[1];

  let outPos = 0;
  for (const slice of valueSlices) {
    for (let i = slice[0]; i < slice[1]; ++i) {
      for (let j = 0; j < valueSize; ++j) {
        values[outPos * valuesM + j] = paramsDenseValues[i * denseM + j];
      }
      ++outPos;
    }
  }
}

function getValues(
    paramsDenseValues: TypedArray, paramsDenseValuesShape: number[],
    paramsDenseValuesDType: DataType, valueSlices: Array<[number, number]>,
    numValues: number): [TypedArray, number[]] {
  const valuesShape = paramsDenseValuesShape.slice();
  valuesShape[0] = numValues;

  const valuesOut = util.getArrayFromDType(
                        paramsDenseValuesDType,
                        util.sizeFromShape(valuesShape)) as TypedArray;

  const numElements = paramsDenseValues.length;
  const valueSize =
      numElements === 0 ? 0 : (numElements / paramsDenseValuesShape[0]);
  writeValueSlices(
      paramsDenseValues, paramsDenseValuesShape, valueSlices, valueSize,
      valuesOut, valuesShape);

  return [valuesOut, valuesShape];
}
export function raggedGatherImpl(
    paramsNestedSplits: TypedArray[], paramsNestedSplitsShapes: number[][],
    paramsDenseValues: TypedArray, paramsDenseValuesShape: number[],
    paramsDenseValuesDType: DataType, indices: TypedArray,
    indicesShape: number[],
    outputRaggedRank: number): [TypedArray[], TypedArray, number[]] {
  if (paramsNestedSplits.length === 0) {
    throw new Error('paramsNestedSplits must be non empty');
  }

  if (paramsNestedSplitsShapes[0].length === 0) {
    throw new Error('Split tensors must not be scalars');
  }
  const numParams = paramsNestedSplitsShapes[0][0] - 1;
  validateIndices(indices, indicesShape, numParams);

  if (paramsDenseValuesShape.length === 0) {
    throw new Error('params.rank must be nonzero');
  }
  const numParamsDenseValues = paramsDenseValuesShape[0];

  // Calculate the `splits`, and store the value slices that we need to
  // copy in `valueSlices`.
  const {outSplits, valueSlices, numValues} = makeSplits(
      indices, indicesShape, paramsNestedSplits, numParamsDenseValues);

  // Write the output tensors.
  const outputNestedSplits = getSplits(outSplits);
  const outputDenseValues = getValues(
      paramsDenseValues, paramsDenseValuesShape, paramsDenseValuesDType,
      valueSlices, numValues);

  return [outputNestedSplits, outputDenseValues[0], outputDenseValues[1]];
}
