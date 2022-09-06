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

export enum RowPartitionType {
  FIRST_DIM_SIZE,
  VALUE_ROWIDS,
  ROW_LENGTHS,
  ROW_SPLITS,
  ROW_LIMITS,
  ROW_STARTS
}

export function combineRaggedTensorToTensorShapes(
    raggedRank: number, shape: number[], valueShape: number[]) {
  // Test for consistency of valueShape and shape specified.
  // If shape is unspecified and valueShape is specified, then copy
  // over the size from the valueShape dimension.

  let outputShape: number[] = new Array();
  if (valueShape == null && shape == null) {
    return outputShape;
  }

  if (shape == null) {
    // Here, value_shape must be of known size.
    while (outputShape.length < raggedRank + valueShape.length) {
      outputShape.push(-1);
    }
  } else {
    outputShape = shape.slice();
  }
  if (valueShape == null) {
    return outputShape;
  }
  // At this point, valueShape and output_shape have known ranks.
  if (raggedRank + valueShape.length !== outputShape.length) {
    throw new Error(
        `rt input.shape and shape=${shape} are incompatible: rt input.rank = ${
            raggedRank +
            valueShape.length}, but shape.rank = ${outputShape.length}`);
  }

  for (let i = 1; i < valueShape.length; ++i) {
    const valueDim = valueShape[i];
    const outputShapeDimIndex =
        outputShape[outputShape.length - valueShape.length + i];
    const outputShapeDim = outputShape[outputShapeDimIndex];

    if (valueDim >= 0) {
      if (outputShapeDim >= 0) {
        if (outputShapeDim !== valueDim) {
          throw new Error(`rt input.shape and shape=${
              shape} are incompatible: rt input.shape[${i + raggedRank}] = ${
              valueDim} but shape[${i + raggedRank}] = ${outputShapeDim}`);
        }
      } else {
        outputShape[outputShapeDimIndex] = valueDim;
      }
    }
  }
  return outputShape;
}

export function getRowPartitionTypesHelper(rowPartitionTypeStrings: string[]) {
  const stringToType = {
    'FIRST_DIM_SIZE': RowPartitionType.FIRST_DIM_SIZE,
    'VALUE_ROWIDS': RowPartitionType.VALUE_ROWIDS,
    'ROW_LENGTHS': RowPartitionType.ROW_LENGTHS,
    'ROW_SPLITS': RowPartitionType.ROW_SPLITS,
    'ROW_LIMITS': RowPartitionType.ROW_LIMITS,
    'ROW_STARTS': RowPartitionType.ROW_STARTS
  };

  const result: RowPartitionType[] = [];
  for (const typeStr of rowPartitionTypeStrings) {
    if (typeStr in stringToType) {
      result.push(stringToType[typeStr as keyof typeof stringToType]);
    } else {
      break;
    }
  }

  return result;
}

export function getRaggedRank(rowPartitionTypes: RowPartitionType[]) {
  if (rowPartitionTypes.length === 0) {
    return 0;
  }
  if (rowPartitionTypes[0] === RowPartitionType.FIRST_DIM_SIZE) {
    return rowPartitionTypes.length - 1;
  }
  return rowPartitionTypes.length;
}

export function validateDefaultValueShape(
    defaultValueShape: number[], valueShape: number[]) {
  if (defaultValueShape == null || valueShape == null) {
    return;
  }

  const defaultNDims = defaultValueShape.length;
  const valuesNDims = valueShape.length;
  if (defaultNDims >= valuesNDims) {
    throw new Error(`defaultValue.shape=${
        defaultValueShape} and ragged tensor flatValues.shape=${
        valueShape}, are incompatible: defaultValue.rank = ${
        defaultNDims} must be less than ragged tensor input flatValues.rank = ${
        valuesNDims})`);
  }
  for (let i = 0; i < Math.min(defaultNDims, valuesNDims - 1); ++i) {
    const defaultDim = defaultValueShape[i];
    const valueDim = valueShape[i + 1];
    if (defaultDim >= 0 && valueDim >= 0 && defaultDim !== 1 &&
        defaultDim !== valueDim) {
      throw new Error(`defaultValue.shape=${
          defaultValueShape}, and ragged tensor input flatValues.shape=${
          valueShape} are incompatible: defaultValue.shape[${
          i - defaultValueShape.length}] = ${
          defaultDim} but ragged tensor input.flatValues.shape[${
          i - defaultValueShape.length}] = ${valueDim}`);
    }
  }
}
