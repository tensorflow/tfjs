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

import {backend_util, broadcastTo, DataType, reshape, tidy, TypedArray, util} from '@tensorflow/tfjs-core';

import RowPartitionType = backend_util.RowPartitionType;
// Based on
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/ragged_tensor_to_tensor_op.cc
class RaggedTensorToTensorOp {
  private readonly rowPartitionTypes: RowPartitionType[];
  private readonly raggedRank: number;
  constructor(
      private shape: TypedArray, private shapeShape: number[],
      private values: TypedArray, private valuesShape: number[],
      private valuesDType: DataType, private defaultValue: TypedArray,
      private defaultValueShape: number[],
      private readonly rowPartitionValues: TypedArray[],
      private readonly rowPartitionValuesShapes: number[][],
      rowPartitionTypeStrings: string[]) {
    this.rowPartitionTypes =
        backend_util.getRowPartitionTypesHelper(rowPartitionTypeStrings);
    this.raggedRank = backend_util.getRaggedRank(this.rowPartitionTypes);
  }

  private getRowPartitionTypeByDimension(dimension: number) {
    if (this.rowPartitionTypes[0] === RowPartitionType.FIRST_DIM_SIZE) {
      return this.rowPartitionTypes[dimension + 1];
    } else {
      return this.rowPartitionTypes[dimension];
    }
  }

  // Returns the relationship between dimension and dimension + 1.
  private getRowPartitionTensor(dimension: number) {
    if (this.rowPartitionTypes[0] === RowPartitionType.FIRST_DIM_SIZE) {
      return this.rowPartitionValues[dimension + 1];
    } else {
      return this.rowPartitionValues[dimension];
    }
  }

  private getMaxWidth(dimension: number) {
    const rowPartitionTensor = this.getRowPartitionTensor(dimension - 1);
    switch (this.getRowPartitionTypeByDimension(dimension - 1)) {
      case RowPartitionType.VALUE_ROWIDS:
        return RaggedTensorToTensorOp.getMaxWidthValueRowID(rowPartitionTensor);
      case RowPartitionType.ROW_SPLITS:
        return RaggedTensorToTensorOp.getMaxWidthRowSplit(rowPartitionTensor);
      default:
        throw new Error(`Cannot handle partition type ${
            RowPartitionType[this.getRowPartitionTypeByDimension(
                dimension - 1)]}`);
    }
  }

  static getMaxWidthRowSplit(rowSplit: TypedArray) {
    const tensorLength = rowSplit.length;
    if (tensorLength === 0 || tensorLength === 1) {
      return 0;
    }
    let maxWidth = 0;
    for (let i = 0; i < tensorLength - 1; ++i) {
      const currentWidth = rowSplit[i + 1] - rowSplit[i];
      if (currentWidth > maxWidth) {
        maxWidth = currentWidth;
      }
    }
    return maxWidth;
  }

  static getMaxWidthValueRowID(valueRowIds: TypedArray) {
    const indexLength = valueRowIds.length;
    if (indexLength === 0) {
      return 0;
    }
    let firstEqualIndex = 0;
    let firstEqualIndexValue = valueRowIds[0];
    let maxWidth = 0;
    for (let i = 1; i < indexLength; ++i) {
      const value = valueRowIds[i];
      if (value !== firstEqualIndexValue) {
        firstEqualIndexValue = value;
        maxWidth = Math.max(i - firstEqualIndex, maxWidth);
        firstEqualIndex = i;
      }
    }
    return Math.max(indexLength - firstEqualIndex, maxWidth);
  }

  private tensorShapeFromTensor(
      t: TypedArray, tShape: number[], isPartial = true) {
    if (tShape.length === 0) {
      if (t[0] === -1) {
        return [];
      }
      throw new Error(
          `The only valid scalar shape tensor is the fully unknown shape specified as -1.`);
    }
    // MakePartialShape/MakeShapeHelper.
    return makeShape(t, isPartial);
  }

  private calculateOutputSize(firstDim: number) {
    const valueShape = this.valuesShape;
    const defaultValueShape = this.defaultValueShape;

    backend_util.validateDefaultValueShape(defaultValueShape, valueShape);

    const shape = this.tensorShapeFromTensor(this.shape, this.shapeShape);
    const outputShape = backend_util.combineRaggedTensorToTensorShapes(
        this.raggedRank, shape, valueShape);

    const result = outputShape;

    if (result[0] < 0) {
      result[0] = firstDim;
    }
    for (let i = 1; i <= this.raggedRank; ++i) {
      if (result[i] < 0) {
        result[i] = this.getMaxWidth(i);
      }
    }

    return result;
  }

  /**
   * The outputIndex represents the index in the output tensor
   * where the first element of a particular dimension would be written.
   * If it is -1, it indicates that the index is out of scope.
   * Example, given firstDimension = 10, firstDimensionOutput = 6,
   * and outputIndexMultiplier = 100:
   * result = [0 100 200 300 400 500 -1 -1 -1 -1]
   * If firstDimensionOutput = 11 instead, then:
   * result = [0 100 200 300 400 500 600 700 800 900]
   */
  private calculateFirstParentOutputIndex(
      firstDimension: number, outputIndexMultiplier: number,
      firstDimensionOutput: number) {
    const minDimension = Math.min(firstDimension, firstDimensionOutput);
    const result: number[] = [];
    let currentOutputIndex = 0;
    for (let i = 0; i < minDimension;
         ++i, currentOutputIndex += outputIndexMultiplier) {
      result.push(currentOutputIndex);
    }
    for (let i = minDimension; i < firstDimension; ++i) {
      result.push(-1);
    }
    util.assert(
        result.length === firstDimension,
        () => 'Final length of result must be equal to firstDimension.');

    return result;
  }

  private calculateOutputIndexRowSplit(
      rowSplit: TypedArray, parentOutputIndex: number[],
      outputIndexMultiplier: number, outputSize: number) {
    const rowSplitSize = rowSplit.length;
    const result: number[] = [];
    for (let i = 0; i < rowSplitSize - 1; ++i) {
      const rowLength = rowSplit[i + 1] - rowSplit[i];
      let realLength = Math.min(outputSize, rowLength);
      let parentOutputIndexCurrent = parentOutputIndex[i];

      if (parentOutputIndexCurrent === -1) {
        realLength = 0;
      }
      for (let j = 0; j < realLength; ++j) {
        result.push(parentOutputIndexCurrent);
        parentOutputIndexCurrent += outputIndexMultiplier;
      }
      for (let j = 0; j < rowLength - realLength; ++j) {
        result.push(-1);
      }
    }
    if (rowSplitSize > 0 && result.length !== rowSplit[rowSplitSize - 1]) {
      throw new Error('Invalid row split size.');
    }

    return result;
  }

  // Calculate the output index of the first element of a list.
  // The parentOutputIndex is the same computation for the previous list.
  // -1 indicates an element or list that is out of range.
  // The outputIndexMultiplier is the number of output indices one moves
  // forward for each column.
  // E.g., given:
  // valueRowIds:[0 1 2 2 2 3 5 5 6]
  // parentOutputIndex:[1000 1100 2000 2100 -1 3000 4000]
  // outputIndexMultiplier: 10
  // outputSize: 2
  // You get:
  // result = [1000 1100 2000 2010 -1 2100 -1 -1 3000]
  // result[0] = parentOutputIndex[valueRowIds[0]]
  // result[1] = parentOutputIndex[valueRowIds[1]]
  // result[2] = parentOutputIndex[valueRowIds[2]]
  // result[3] = parentOutputIndex[valueRowIds[2] + 10]
  // result[4] = -1 because it is the third element the size is 2.
  // result[5] = parentOutputIndex[valueRowIds[3]]
  // result[6] = -1 because parentOutputIndex[valueRowIds[6]] == -1
  // result[7] = -1 because parentOutputIndex[valueRowIds[6]] == -1
  // result[8] = parentOutputIndex[valueRowIds[7]]
  private calculateOutputIndexValueRowID(
      valueRowIds: TypedArray, parentOutputIndex: number[],
      outputIndexMultiplier: number, outputSize: number) {
    const indexSize = valueRowIds.length;
    const result: number[] = [];
    if (indexSize === 0) {
      return [];
    }

    let currentOutputColumn = 0;
    let currentValueRowId = valueRowIds[0];

    if (currentValueRowId >= parentOutputIndex.length) {
      throw new Error(
          `Got currentValueRowId=${currentValueRowId}, which is not less than ${
              parentOutputIndex.length}`);
    }

    let currentOutputIndex = parentOutputIndex[currentValueRowId];
    result.push(currentOutputIndex);
    for (let i = 1; i < indexSize; ++i) {
      const nextValueRowId = valueRowIds[i];
      if (nextValueRowId === currentValueRowId) {
        if (currentOutputIndex >= 0) {
          ++currentOutputColumn;
          if (currentOutputColumn < outputSize) {
            currentOutputIndex += outputIndexMultiplier;
          } else {
            currentOutputIndex = -1;
          }
        }
      } else {
        currentOutputColumn = 0;
        currentValueRowId = nextValueRowId;

        if (nextValueRowId >= parentOutputIndex.length) {
          throw new Error(
              `Got nextValueRowId=${nextValueRowId} which is not less than ${
                  parentOutputIndex.length}`);
        }

        currentOutputIndex = parentOutputIndex[nextValueRowId];
      }
      result.push(currentOutputIndex);
    }

    if (result.length !== valueRowIds.length) {
      throw new Error('Invalid row ids.');
    }

    return result;
  }

  private calculateOutputIndex(
      dimension: number, parentOutputIndex: number[],
      outputIndexMultiplier: number, outputSize: number) {
    const rowPartitionTensor = this.getRowPartitionTensor(dimension);
    const partitionType = this.getRowPartitionTypeByDimension(dimension);
    switch (partitionType) {
      case RowPartitionType.VALUE_ROWIDS:
        return this.calculateOutputIndexValueRowID(
            rowPartitionTensor, parentOutputIndex, outputIndexMultiplier,
            outputSize);
      case RowPartitionType.ROW_SPLITS:
        if (rowPartitionTensor.length - 1 > parentOutputIndex.length) {
          throw new Error(`Row partition size is greater than output size: ${
              rowPartitionTensor.length - 1} > ${parentOutputIndex.length}`);
        }
        return this.calculateOutputIndexRowSplit(
            rowPartitionTensor, parentOutputIndex, outputIndexMultiplier,
            outputSize);
      default:
        throw new Error(
            `Unsupported partition type: ${RowPartitionType[partitionType]}`);
    }
  }

  private getFirstDimensionSize() {
    const firstPartitionTensor = this.rowPartitionValues[0];
    if (this.rowPartitionTypes.length === 0) {
      throw new Error('No row_partition_types given.');
    }
    const firstPartitionType = this.rowPartitionTypes[0];
    switch (firstPartitionType) {
      case RowPartitionType.FIRST_DIM_SIZE:
        return firstPartitionTensor[0];
      case RowPartitionType.VALUE_ROWIDS:
        throw new Error('Cannot handle VALUE_ROWIDS in first dimension.');
      case RowPartitionType.ROW_SPLITS:
        return this.rowPartitionValuesShapes[0][0] - 1;
      default:
        throw new Error(
            `Cannot handle type ${RowPartitionType[firstPartitionType]}`);
    }
  }

  compute(): [number[], TypedArray] {
    const firstPartitionTensor = this.rowPartitionValues[0];
    if (firstPartitionTensor.length <= 0) {
      throw new Error(
          'Invalid first partition input. ' +
          'Tensor requires at least one element.');
    }
    const firstDimension = this.getFirstDimensionSize();
    const outputSize = this.calculateOutputSize(firstDimension);
    const multiplier: number[] = new Array(this.raggedRank + 1);

    multiplier[multiplier.length - 1] = 1;
    for (let i = multiplier.length - 2; i >= 0; --i) {
      multiplier[i] = multiplier[i + 1] * outputSize[i + 1];
    }
    // Full size of the tensor.
    const outputShape: number[] = makeShape(outputSize, false);
    const outputTensor =
        util.getArrayFromDType(
            this.valuesDType, util.sizeFromShape(outputShape)) as TypedArray;

    const fullSize = multiplier[0] * outputSize[0];
    if (fullSize > 0) {
      let outputIndex = this.calculateFirstParentOutputIndex(
          firstDimension, multiplier[0], outputSize[0]);
      for (let i = 1; i <= this.raggedRank; ++i) {
        const newOutputIndex = this.calculateOutputIndex(
            i - 1, outputIndex, multiplier[i], outputSize[i]);
        outputIndex = newOutputIndex;
      }

      this.setOutput(this.raggedRank, outputIndex, outputTensor, outputShape);
    }

    return [outputShape, outputTensor];
  }
  setOutput(
      raggedRank: number, outputIndex: number[], outputTensor: TypedArray,
      outputShape: number[]) {
    if (outputTensor.length === 0) {
      return;
    }

    const valuesBase = this.values;
    const outputBase = outputTensor;

    let elementShape = outputShape.slice();
    elementShape = elementShape.slice(raggedRank + 1);
    const valueElementSize = util.sizeFromShape(elementShape);
    const outputIndexSize = outputIndex.length;

    // Broadcast the default value to value_element_size.  (We can skip this
    // if defaultValueTensor.size == 1, since we use fill when that's true.)
    let defaultValue = this.defaultValue;
    if (defaultValue.length !== valueElementSize && defaultValue.length !== 1) {
      const srcShape = this.defaultValueShape;
      tidy(() => {
        const defaultValueTensor = reshape(defaultValue, srcShape);
        const bCastDefault = broadcastTo(defaultValueTensor, elementShape);
        defaultValue = bCastDefault.dataSync();
      });
    }

    // Loop through the outputIndex array, finding contiguous regions that
    // should be copied.  Once we find the end of a contiguous region, copy it
    // and add any necessary padding (with defaultValue).
    let srcStart = 0;  // Start of contiguous region (in values)
    let dstStart = 0;  // Destination for contiguous region (in output)
    let dstEnd = 0;    // Destination for contiguous region (in output)
    for (let srcI = 0; srcI <= outputIndexSize; ++srcI) {
      // dstI is the destination where the value at srcI should be copied.
      let dstI = srcI < outputIndexSize ? outputIndex[srcI] : -1;

      // If we're still in a contiguous region, then update dstEnd go to the
      // next srcI.
      if (dstI === dstEnd) {
        ++dstEnd;
        continue;
      }

      // We found the end of contiguous region.  This can be because we found
      // a gap (dstI > dstEnd), or a source value that shouldn't be copied
      // because it's out-of-bounds (dstI == -1), or the end of the tensor
      // (dstI === -1).
      if (dstStart < dstEnd) {
        // Copy the contiguous region.
        const src = valuesBase.subarray(srcStart * valueElementSize);
        const dst = outputBase.subarray(dstStart * valueElementSize);
        const nVals = (dstEnd - dstStart) * valueElementSize;
        copyArray(dst, src, nVals);
      }

      // Add any necessary padding (w/ defaultValue).
      if (srcI >= outputIndexSize) {
        // We reached the end of values: pad to the end of output.
        const outputSize = outputTensor.length;
        dstI = Math.floor(outputSize / valueElementSize);
      }
      if (dstI > dstEnd) {
        if (this.defaultValue.length === 1) {
          outputBase
              .subarray(dstEnd * valueElementSize, dstI * valueElementSize)
              .fill(this.defaultValue[0]);
          dstEnd = dstI;
        } else {
          while (dstI > dstEnd) {
            const dst = outputBase.slice(dstEnd * valueElementSize);
            copyArray(dst, defaultValue, valueElementSize);
            ++dstEnd;
          }
        }
      }

      // Update indices.
      if (dstI < 0) {
        // srcI should be skipped -- leave it out of the contiguous region.
        srcStart = srcI + 1;
        dstStart = dstEnd;
      } else {
        // srcI should be copied -- include it in the contiguous region.
        srcStart = srcI;
        dstStart = dstEnd;
        dstEnd = dstStart + 1;
      }
    }
  }
}

function copyArray(dst: TypedArray, src: TypedArray, size: number) {
  for (let i = 0; i < size; i++) {
    dst[i] = src[i];
  }
}

function makeShape(shape: number[]|TypedArray, isPartial: boolean) {
  const out: number[] = [];
  for (let dim of shape) {
    if (dim < 0) {
      if (!isPartial) {
        throw new Error(`Dimension ${dim} must be >= 0`);
      }
      if (dim < -1) {
        throw new Error(`Dimension ${dim} must be >= -1`);
      }
      dim = -1;
    }
    out.push(dim);
  }

  return out;
}

export function raggedTensorToTensorImpl(
    shape: TypedArray, shapesShape: number[], values: TypedArray,
    valuesShape: number[], valuesDType: DataType, defaultValue: TypedArray,
    defaultValueShape: number[], rowPartitionValues: TypedArray[],
    rowPartitionValuesShapes: number[][],
    rowPartitionTypes: string[]): [number[], TypedArray] {
  return new RaggedTensorToTensorOp(
             shape, shapesShape, values, valuesShape, valuesDType, defaultValue,
             defaultValueShape, rowPartitionValues, rowPartitionValuesShapes,
             rowPartitionTypes)
      .compute();
}
