/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

import * as seedrandom from 'seedrandom';

import {ENV} from '../../environment';
import * as util from '../../util';
import * as broadcast_util from '../broadcast_util';
import * as concat_util from '../concat_util';
import {Conv2DInfo} from '../conv_util';
import {NDArrayMath} from '../math';
// tslint:disable-next-line:max-line-length
import {Array1D, Array2D, Array3D, Array4D, DataType, DataTypeMap, NDArray, Rank, Scalar} from '../ndarray';
import * as types from '../types';
import {SumTypes, SumTypesMap} from '../types';

import * as axis_util from './../axis_util';
import {MathBackend} from './backend';
import {MatrixOrientation} from './types/matmul';

export class MathBackendCPU implements MathBackend {
  private data: {[dataId: number]: DataTypeMap[DataType]} = {};
  private canvas: HTMLCanvasElement;

  constructor() {
    if (typeof document !== 'undefined') {
      this.canvas = document.createElement('canvas');
    }
  }

  register(dataId: number, shape: number[], dtype: DataType): void {
    this.data[dataId] = null;
  }
  write<D extends DataType>(dataId: number, values: DataTypeMap[D]): void {
    if (values == null) {
      throw new Error('MathBackendCPU.write(): values can not be null');
    }
    this.throwIfNoData(dataId);
    this.data[dataId] = values;
  }
  writePixels(
      dataId: number,
      pixels: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      numChannels: number): void {
    if (pixels == null) {
      throw new Error('MathBackendCPU.writePixels(): pixels can not be null');
    }
    this.throwIfNoData(dataId);
    let vals: Uint8ClampedArray;
    if (pixels instanceof ImageData) {
      vals = pixels.data;
    } else if (pixels instanceof HTMLCanvasElement) {
      vals = pixels.getContext('2d')
                 .getImageData(0, 0, pixels.width, pixels.height)
                 .data;
    } else if (
        pixels instanceof HTMLImageElement ||
        pixels instanceof HTMLVideoElement) {
      if (this.canvas == null) {
        throw new Error(
            'Can\'t read pixels from HTMLImageElement outside ' +
            'the browser.');
      }
      this.canvas.width = pixels.width;
      this.canvas.height = pixels.height;
      this.canvas.getContext('2d').drawImage(
          pixels, 0, 0, pixels.width, pixels.height);
      vals = this.canvas.getContext('2d')
                 .getImageData(0, 0, pixels.width, pixels.height)
                 .data;
    } else {
      throw new Error(
          `pixels is of unknown type: ${(pixels as {}).constructor.name}`);
    }
    let values: Int32Array;
    if (numChannels === 4) {
      values = new Int32Array(vals);
    } else {
      const numPixels = pixels.width * pixels.height;
      values = new Int32Array(numPixels * numChannels);
      for (let i = 0; i < numPixels; i++) {
        for (let channel = 0; channel < numChannels; ++channel) {
          values[i * numChannels + channel] = vals[i * 4 + channel];
        }
      }
    }
    this.data[dataId] = values;
  }
  async read<D extends DataType>(dataId: number): Promise<DataTypeMap[D]> {
    this.throwIfNoData(dataId);
    return this.data[dataId];
  }
  readSync<D extends DataType>(dataId: number): DataTypeMap[D] {
    this.throwIfNoData(dataId);
    return this.data[dataId];
  }
  disposeData(dataId: number): void {
    delete this.data[dataId];
  }
  async time(query: () => NDArray): Promise<number> {
    const start = performance.now();
    query();
    return performance.now() - start;
  }
  private throwIfNoData(dataId: number) {
    if (!(dataId in this.data)) {
      throw new Error(
          `No data found for NDArray with data id ${dataId}. ` +
          `Use dl.ENV.math instead of constructing your own NDArrayMath. ` +
          `If you need to construct your own math, make sure this array is ` +
          `allocated after the math construction`);
    }
  }

  clone<T extends NDArray>(x: T): T {
    return NDArray.make(x.shape, {values: new Float32Array(x.dataSync())}) as T;
  }

  slice1D(x: Array1D, begin: number, size: number): Array1D {
    const newVals = x.dataSync().slice(begin, begin + size);
    return Array1D.new(newVals);
  }

  slice2D(x: Array2D, begin: [number, number], size: [number, number]):
      Array2D {
    const result = Array2D.zeros(size);
    const [startI, startJ] = begin;

    for (let i = 0; i < size[0]; ++i) {
      for (let j = 0; j < size[1]; ++j) {
        const val = x.get(i + startI, j + startJ);
        result.set(val, i, j);
      }
    }
    return result;
  }

  slice3D(x: Array3D, begin: [number, number, number], size: [
    number, number, number
  ]): Array3D {
    const result = Array3D.zeros(size);
    const [startI, startJ, startK] = begin;

    for (let i = 0; i < size[0]; ++i) {
      for (let j = 0; j < size[1]; ++j) {
        for (let k = 0; k < size[2]; ++k) {
          const val = x.get(i + startI, j + startJ, k + startK);
          result.set(val, i, j, k);
        }
      }
    }
    return result;
  }
  slice4D(x: Array4D, begin: [number, number, number, number], size: [
    number, number, number, number
  ]): Array4D {
    const result = Array4D.zeros(size);
    const [startI, startJ, startK, startL] = begin;

    for (let i = 0; i < size[0]; ++i) {
      for (let j = 0; j < size[1]; ++j) {
        for (let k = 0; k < size[2]; ++k) {
          for (let l = 0; l < size[3]; ++l) {
            const val = x.get(i + startI, j + startJ, k + startK, l + startL);
            result.set(val, i, j, k, l);
          }
        }
      }
    }
    return result;
  }

  reverse4D(x: Array4D, axis: number[]): Array4D {
    const result = NDArray.like(x);

    // Reverse axis only if the axis has dim != 1
    const revAxis = (i: number) => axis.indexOf(i) !== -1 && x.shape[i] !== 1;

    // naive O(n) reverse implementation
    for (let b = 0; b < x.shape[0]; ++b) {
      for (let r = 0; r < x.shape[1]; ++r) {
        for (let c = 0; c < x.shape[2]; ++c) {
          for (let d = 0; d < x.shape[3]; ++d) {
            const b0 = revAxis(0) ? x.shape[0] - b - 1 : b;
            const r0 = revAxis(1) ? x.shape[1] - r - 1 : r;
            const c0 = revAxis(2) ? x.shape[2] - c - 1 : c;
            const d0 = revAxis(3) ? x.shape[3] - d - 1 : d;
            const val = x.get(b0, r0, c0, d0);
            result.set(val, b, r, c, d);
          }
        }
      }
    }

    return result;
  }

  concat1D(a: Array1D, b: Array1D): Array1D {
    const outShape = concat_util.computeOutShape(a.shape, b.shape, 0);
    const result = Array1D.zeros(outShape as [number]);

    // Use built-in TypedArray.set() method for speed.
    const aVals = a.dataSync();
    const bVals = b.dataSync();
    const vals = result.dataSync();
    vals.set(aVals, 0);
    vals.set(bVals, a.size);

    return result;
  }

  concat2D(a: Array2D, b: Array2D, axis: number): Array2D {
    const outShape = concat_util.computeOutShape(a.shape, b.shape, axis);
    const result = Array2D.zeros(outShape as [number, number]);

    if (axis === 0) {
      // Use built-in TypedArray.set() method for speed.
      const aVals = a.dataSync();
      const bVals = b.dataSync();
      const vals = result.dataSync();
      vals.set(aVals, 0);
      vals.set(bVals, a.size);
      return result;
    }

    for (let i = 0; i < outShape[0]; ++i) {
      for (let j = 0; j < outShape[1]; ++j) {
        const index: [number, number] = [i, j];
        let value: number;
        if (index[axis] < a.shape[axis]) {
          value = a.get(i, j);
        } else {
          index[axis] -= a.shape[axis];
          const [i2, j2] = index;
          value = b.get(i2, j2);
        }

        result.set(value, i, j);
      }
    }
    return result;
  }

  concat3D(a: Array3D, b: Array3D, axis: number): Array3D {
    const outShape = concat_util.computeOutShape(a.shape, b.shape, axis);

    const result = Array3D.zeros(outShape as [number, number, number]);

    if (axis === 0) {
      // Use built-in TypedArray.set() method for speed.
      const aVals = a.dataSync();
      const bVals = b.dataSync();
      const vals = result.dataSync();
      vals.set(aVals, 0);
      vals.set(bVals, a.size);
      return result;
    }

    for (let i = 0; i < outShape[0]; ++i) {
      for (let j = 0; j < outShape[1]; ++j) {
        for (let k = 0; k < outShape[2]; ++k) {
          // Shader begins.
          const index: [number, number, number] = [i, j, k];
          let value: number;
          if (index[axis] < a.shape[axis]) {
            value = a.get(i, j, k);
          } else {
            index[axis] -= a.shape[axis];
            const [i2, j2, k2] = index;
            value = b.get(i2, j2, k2);
          }

          result.set(value, i, j, k);
        }
      }
    }

    return result;
  }

  concat4D(a: Array4D, b: Array4D, axis: number): Array4D {
    const outShape = concat_util.computeOutShape(a.shape, b.shape, axis);
    const result = Array4D.zeros(outShape as [number, number, number, number]);

    if (axis === 0) {
      // Use built-in TypedArray.set() method for speed.
      const aVals = a.dataSync();
      const bVals = b.dataSync();
      const vals = result.dataSync();
      vals.set(aVals, 0);
      vals.set(bVals, a.size);
      return result;
    }

    for (let i = 0; i < outShape[0]; ++i) {
      for (let j = 0; j < outShape[1]; ++j) {
        for (let k = 0; k < outShape[2]; ++k) {
          for (let l = 0; l < outShape[3]; ++l) {
            // Shader begins.
            const index: [number, number, number, number] = [i, j, k, l];
            let value: number;
            if (index[axis] < a.shape[axis]) {
              value = a.get(i, j, k, l);
            } else {
              index[axis] -= a.shape[axis];
              const [i2, j2, k2, l2] = index;
              value = b.get(i2, j2, k2, l2);
            }

            result.set(value, i, j, k, l);
          }
        }
      }
    }

    return result;
  }

  neg<T extends NDArray>(x: T): T {
    return this.multiply(Scalar.new(-1), x) as T;
  }

  add<D extends DataType>(a: NDArray<D>, b: NDArray<D>): NDArray<D> {
    return this.broadcastedBinaryOp(
               a, b, types.upcastType(a.dtype, b.dtype),
               (aValue, bValue) => aValue + bValue) as NDArray<D>;
  }

  subtract<D extends DataType>(a: NDArray<D>, b: NDArray<D>): NDArray<D> {
    return this.broadcastedBinaryOp(
               a, b, types.upcastType(a.dtype, b.dtype),
               (aValue, bValue) => aValue - bValue) as NDArray<D>;
  }

  pow<T extends NDArray>(a: T, b: NDArray<'int32'>): T {
    return this.broadcastedBinaryOp(
               a, b, a.dtype, (aValue, bValue) => Math.pow(aValue, bValue)) as
        T;
  }

  matMul(
      a: Array2D, b: Array2D, aOrientation = MatrixOrientation.REGULAR,
      bOrientation = MatrixOrientation.REGULAR): Array2D {
    const sharedDim =
        (aOrientation === MatrixOrientation.REGULAR) ? a.shape[1] : a.shape[0];

    const leftDim =
        (aOrientation === MatrixOrientation.REGULAR) ? a.shape[0] : a.shape[1];
    const rightDim =
        (bOrientation === MatrixOrientation.REGULAR) ? b.shape[1] : b.shape[0];

    const normalGetter = (matrix: Array2D, i: number, j: number) =>
        matrix.get(i, j);
    const transposedGetter = (matrix: Array2D, i: number, j: number) =>
        matrix.get(j, i);

    const aGetter = (aOrientation === MatrixOrientation.REGULAR) ?
        normalGetter :
        transposedGetter;
    const bGetter = (bOrientation === MatrixOrientation.REGULAR) ?
        normalGetter :
        transposedGetter;
    const values = new Float32Array(leftDim * rightDim);
    let index = 0;
    for (let i = 0; i < leftDim; ++i) {
      for (let j = 0; j < rightDim; ++j) {
        let sum = 0;
        for (let k = 0; k < sharedDim; ++k) {
          // TODO: optimize CPU matmul.
          sum += aGetter(a, i, k) * bGetter(b, k, j);
        }
        values[index++] = sum;
      }
    }
    return Array2D.new([leftDim, rightDim], values);
  }

  multiply<D extends DataType>(a: NDArray<D>, b: NDArray<D>): NDArray<D> {
    return this.broadcastedBinaryOp(
               a, b, types.upcastType(a.dtype, b.dtype),
               (aValue, bValue) => aValue * bValue) as NDArray<D>;
  }

  divide(a: NDArray, b: NDArray): NDArray<'float32'> {
    return this.broadcastedBinaryOp(
               a, b, 'float32', (aValue, bValue) => aValue / bValue) as
        NDArray<'float32'>;
  }

  sum<D extends DataType>(x: NDArray<D>, axes: number[]): NDArray<SumTypes[D]> {
    axis_util.assertAxesAreInnerMostDims('sum', axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const resultDtype = SumTypesMap[x.dtype] as keyof SumTypes;
    const result = NDArray.zeros(outShape, resultDtype);
    const reduceSize = util.sizeFromShape(reduceShape);
    const vals = result.dataSync();

    const aVals = x.dataSync();
    for (let i = 0; i < vals.length; ++i) {
      const offset = i * reduceSize;
      let sum = 0;
      for (let j = 0; j < reduceSize; ++j) {
        sum += aVals[offset + j];
      }
      vals[i] = sum;
    }
    return result as NDArray<SumTypes[D]>;
  }

  argMin(x: NDArray, axes: number[]): NDArray<'int32'> {
    axis_util.assertAxesAreInnerMostDims('argMin', axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const result = NDArray.zeros(outShape, 'int32');
    const reduceSize = util.sizeFromShape(reduceShape);
    const vals = result.dataSync();

    const aVals = x.dataSync();
    for (let i = 0; i < vals.length; ++i) {
      const offset = i * reduceSize;
      let min = aVals[offset];
      let minIndex = 0;
      for (let j = 0; j < reduceSize; ++j) {
        const value = aVals[offset + j];
        if (isNaN(value)) {
          minIndex = util.NAN_INT32;
          break;
        }
        if (value < min) {
          min = value;
          minIndex = j;
        }
      }
      vals[i] = minIndex;
    }
    return result;
  }

  argMax(x: NDArray, axes: number[]): NDArray<'int32'> {
    axis_util.assertAxesAreInnerMostDims('argMax', axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const result = NDArray.zeros(outShape, 'int32');
    const reduceSize = util.sizeFromShape(reduceShape);
    const vals = result.dataSync();

    const aVals = x.dataSync();
    for (let i = 0; i < vals.length; ++i) {
      const offset = i * reduceSize;
      let max = aVals[offset];
      let maxIndex = 0;
      for (let j = 0; j < reduceSize; ++j) {
        const value = aVals[offset + j];
        if (isNaN(value)) {
          maxIndex = util.NAN_INT32;
          break;
        }
        if (value > max) {
          max = value;
          maxIndex = j;
        }
      }
      vals[i] = maxIndex;
    }
    return result;
  }

  equal(a: NDArray, b: NDArray): NDArray<'bool'> {
    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      if (util.isValNaN(aVal, a.dtype) || util.isValNaN(bVal, b.dtype)) {
        return util.getNaN('bool');
      } else {
        return (aVal === bVal) ? 1 : 0;
      }
    });
  }

  notEqual(a: NDArray, b: NDArray): NDArray<'bool'> {
    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      if (util.isValNaN(aVal, a.dtype) || util.isValNaN(bVal, b.dtype)) {
        return util.getNaN('bool');
      } else {
        return (aVal !== bVal) ? 1 : 0;
      }
    });
  }

  less(a: NDArray, b: NDArray): NDArray<'bool'> {
    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      if (util.isValNaN(aVal, a.dtype) || util.isValNaN(bVal, b.dtype)) {
        return util.getNaN('bool');
      } else {
        return (aVal < bVal) ? 1 : 0;
      }
    });
  }

  lessEqual(a: NDArray, b: NDArray): NDArray<'bool'> {
    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      if (util.isValNaN(aVal, a.dtype) || util.isValNaN(bVal, b.dtype)) {
        return util.getNaN('bool');
      } else {
        return (aVal <= bVal) ? 1 : 0;
      }
    });
  }

  greater(a: NDArray, b: NDArray): NDArray<'bool'> {
    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      if (util.isValNaN(aVal, a.dtype) || util.isValNaN(bVal, b.dtype)) {
        return util.getNaN('bool');
      } else {
        return (aVal > bVal) ? 1 : 0;
      }
    });
  }

  greaterEqual(a: NDArray, b: NDArray): NDArray<'bool'> {
    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      if (util.isValNaN(aVal, a.dtype) || util.isValNaN(bVal, b.dtype)) {
        return util.getNaN('bool');
      } else {
        return (aVal >= bVal) ? 1 : 0;
      }
    });
  }

  logicalAnd(a: NDArray, b: NDArray): NDArray<'bool'> {
    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      if (util.isValNaN(aVal, a.dtype) || util.isValNaN(bVal, b.dtype)) {
        return util.getNaN('bool');
      } else {
        return aVal && bVal;
      }
    });
  }

  logicalOr(a: NDArray, b: NDArray): NDArray<'bool'> {
    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      if (util.isValNaN(aVal, a.dtype) || util.isValNaN(bVal, b.dtype)) {
        return util.getNaN('bool');
      } else {
        return aVal || bVal;
      }
    });
  }

  where<D extends DataType>(
      condition: NDArray, a: NDArray, b: NDArray, dtype: D): NDArray<D> {
    const values = condition.dataSync();
    const aValues = a.dataSync();
    const bValues = b.dataSync();
    const result = NDArray.zeros(a.shape, dtype);
    const newValues = result.dataSync();
    let index = 0;
    const offset = condition.rank > 1 || a.rank === 1 ? 1 : a.shape[1];
    for (let i = 0; i < values.length; i++) {
      for (let j = 0; j < offset; j++) {
        if (values[i] === 1) {
          newValues[index++] = aValues[i];
        } else {
          newValues[index++] = bValues[i];
        }
      }
    }
    return result;
  }

  topKValues<D extends DataType, T extends NDArray<D>>(x: T, k: number):
      Array1D<D> {
    return this.topK(x, k).values as Array1D<D>;
  }

  topKIndices(x: NDArray, k: number): Array1D<'int32'> {
    return this.topK(x, k).indices;
  }

  private topK<D extends DataType, T extends NDArray<D>>(x: T, k: number):
      {values: Array1D<D>, indices: Array1D<'int32'>} {
    const values = x.dataSync();
    const valuesAndIndices: Array<{value: number, index: number}> = [];
    for (let i = 0; i < values.length; i++) {
      valuesAndIndices.push({value: values[i], index: i});
    }
    valuesAndIndices.sort((a, b) => {
      return b.value - a.value;
    });

    const topkValues = util.getTypedArrayFromDType(x.dtype, k);
    const topkIndices = new Int32Array(k);
    for (let i = 0; i < k; i++) {
      topkValues[i] = valuesAndIndices[i].value;
      topkIndices[i] = valuesAndIndices[i].index;
    }
    return {
      values: Array1D.new<D>(topkValues),
      indices: Array1D.new<'int32'>(topkIndices)
    };
  }

  min<D extends DataType>(x: NDArray<D>, axes: number[]): NDArray<D> {
    axis_util.assertAxesAreInnerMostDims('min', axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const result = NDArray.zeros(outShape, x.dtype);
    const reduceSize = util.sizeFromShape(reduceShape);
    const vals = result.dataSync();

    const aVals = x.dataSync();
    for (let i = 0; i < vals.length; ++i) {
      const offset = i * reduceSize;
      let min = aVals[0];
      for (let j = 0; j < reduceSize; ++j) {
        const value = aVals[offset + j];
        if (isNaN(value)) {
          min = Number.NaN;
          break;
        }
        if (value < min) {
          min = value;
        }
      }
      vals[i] = min;
    }
    return result;
  }

  minimum<D extends DataType>(a: NDArray<D>, b: NDArray<D>): NDArray<D> {
    return this.broadcastedBinaryOp(
        a, b, a.dtype, (aVal, bVal) => Math.min(aVal, bVal));
  }

  max<D extends DataType>(x: NDArray<D>, axes: number[]): NDArray<D> {
    axis_util.assertAxesAreInnerMostDims('max', axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const result = NDArray.zeros(outShape, x.dtype);
    const reduceSize = util.sizeFromShape(reduceShape);
    const vals = result.dataSync();

    const aVals = x.dataSync();
    for (let i = 0; i < vals.length; ++i) {
      const offset = i * reduceSize;
      let max = aVals[offset];
      for (let j = 0; j < reduceSize; ++j) {
        const value = aVals[offset + j];
        if (isNaN(value)) {
          max = Number.NaN;
          break;
        }
        if (value > max) {
          max = value;
        }
      }
      vals[i] = max;
    }
    return result;
  }

  maximum<D extends DataType>(a: NDArray<D>, b: NDArray<D>): NDArray<D> {
    return this.broadcastedBinaryOp(
        a, b, a.dtype, (aVal, bVal) => Math.max(aVal, bVal));
  }

  ceil<T extends NDArray>(x: T): T {
    const values = x.dataSync();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      newValues[i] = Math.ceil(values[i]);
    }
    return NDArray.make(x.shape, {values: newValues}) as T;
  }

  floor<T extends NDArray>(x: T): T {
    const values = x.dataSync();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      newValues[i] = Math.floor(values[i]);
    }
    return NDArray.make(x.shape, {values: newValues}) as T;
  }

  exp<T extends NDArray>(x: T): T {
    const values = x.dataSync();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      newValues[i] = Math.exp(values[i]);
    }
    return NDArray.make(x.shape, {values: newValues}) as T;
  }

  log<T extends NDArray>(x: T): T {
    const values = x.dataSync();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      const value = values[i];
      newValues[i] = Math.log(value);
    }
    return NDArray.make(x.shape, {values: newValues}) as T;
  }

  sqrt<T extends NDArray>(x: T): T {
    const values = x.dataSync();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      const value = values[i];
      newValues[i] = Math.sqrt(value);
    }
    return NDArray.make(x.shape, {values: newValues}) as T;
  }

  square<T extends NDArray>(x: T): T {
    const values = x.dataSync();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      const value = values[i];
      newValues[i] = value * value;
    }
    return NDArray.make(x.shape, {values: newValues}) as T;
  }

  relu<T extends NDArray>(x: T): T {
    const res = NDArray.zeros(x.shape, x.dtype);
    const resVals = res.dataSync();
    const inVals = x.dataSync();
    for (let i = 0; i < inVals.length; ++i) {
      const val = inVals[i];
      if (util.isValNaN(val, x.dtype)) {
        resVals[i] = util.getNaN(res.dtype);
      } else {
        resVals[i] = Math.max(0, inVals[i]);
      }
    }
    return res as T;
  }

  elu<T extends NDArray>(x: T): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      const v = values[i];
      if (v >= 0) {
        resultValues[i] = v;
      } else {
        resultValues[i] = (Math.exp(v) - 1);
      }
    }
    return NDArray.make(x.shape, {values: resultValues}) as T;
  }

  eluDer<T extends NDArray>(x: T): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      const v = values[i];
      if (v >= 0) {
        resultValues[i] = 1;
      } else {
        resultValues[i] = Math.exp(v);
      }
    }
    return NDArray.make(x.shape, {values: resultValues}) as T;
  }

  selu<T extends NDArray>(x: T): T {
    // Stable and Attracting Fixed Point (0, 1) for Normalized Weights.
    // see: https://arxiv.org/abs/1706.02515
    const scaleAlpha = 1.7580993408473768599402175208123;
    const scale = 1.0507009873554804934193349852946;

    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      const v = values[i];
      if (v >= 0) {
        resultValues[i] = scale * v;
      } else {
        resultValues[i] = scaleAlpha * (Math.exp(v) - 1);
      }
    }
    return NDArray.make(x.shape, {values: resultValues}) as T;
  }

  leakyRelu<T extends NDArray>(x: T, alpha: number) {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (v >= 0) {
        resultValues[i] = v;
      } else {
        resultValues[i] = alpha * v;
      }
    }
    return NDArray.make(x.shape, {values: resultValues}) as T;
  }

  prelu<T extends NDArray>(x: T, alpha: T) {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    const alphas = alpha.dataSync();
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (v >= 0) {
        resultValues[i] = v;
      } else {
        resultValues[i] = alphas[i] * v;
      }
    }
    return NDArray.make(x.shape, {values: resultValues}) as T;
  }

  preluDer<T extends NDArray>(x: T, alpha: T) {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    const alphas = alpha.dataSync();
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (v > 0) {
        resultValues[i] = 1;
      } else if (v < 0) {
        resultValues[i] = alphas[i];
      } else {
        resultValues[i] = v;
      }
    }
    return NDArray.make(x.shape, {values: resultValues}) as T;
  }

  clip<T extends NDArray>(x: T, min: number, max: number): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.min(max, Math.max(min, values[i]));
    }
    return NDArray.make(x.shape, {values: resultValues}) as T;
  }

  abs<T extends NDArray>(x: T): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.abs(values[i]);
    }
    return NDArray.make(x.shape, {values: resultValues}) as T;
  }

  int<R extends Rank>(x: NDArray<DataType, R>): NDArray<'int32', R> {
    const resultValues = new Int32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = values[i];
    }
    return NDArray.make(x.shape, {values: resultValues}, 'int32') as
        NDArray<'int32', R>;
  }

  sigmoid<T extends NDArray>(x: T): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = 1 / (1 + Math.exp(-values[i]));
    }
    return NDArray.make(x.shape, {values: resultValues}) as T;
  }

  sin<T extends NDArray>(x: T): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.sin(values[i]);
    }
    return NDArray.make(x.shape, {values: resultValues}) as T;
  }

  cos<T extends NDArray>(x: T): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.cos(values[i]);
    }
    return NDArray.make(x.shape, {values: resultValues}) as T;
  }

  tan<T extends NDArray>(x: T): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.tan(values[i]);
    }
    return NDArray.make(x.shape, {values: resultValues}) as T;
  }

  asin<T extends NDArray>(x: T): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.asin(values[i]);
    }
    return NDArray.make(x.shape, {values: resultValues}) as T;
  }

  acos<T extends NDArray>(x: T): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.acos(values[i]);
    }
    return NDArray.make(x.shape, {values: resultValues}) as T;
  }

  atan<T extends NDArray>(x: T): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.atan(values[i]);
    }
    return NDArray.make(x.shape, {values: resultValues}) as T;
  }

  sinh<T extends NDArray>(x: T): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.sinh(values[i]);
    }
    return NDArray.make(x.shape, {values: resultValues}) as T;
  }

  cosh<T extends NDArray>(x: T): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.cosh(values[i]);
    }
    return NDArray.make(x.shape, {values: resultValues}) as T;
  }

  tanh<T extends NDArray>(x: T): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = util.tanh(values[i]);
    }
    return NDArray.make(x.shape, {values: resultValues}) as T;
  }

  step<T extends NDArray>(x: T, alpha = 0): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      const value = values[i];
      if (util.isValNaN(value, x.dtype)) {
        resultValues[i] = util.getNaN(x.dtype);
      } else {
        resultValues[i] = value > 0 ? 1 : alpha;
      }
    }
    return NDArray.make(x.shape, {values: resultValues}) as T;
  }

  conv2d(x: Array4D, filter: Array4D, bias: Array1D|null, convInfo: Conv2DInfo):
      Array4D {
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const padLeft = convInfo.padInfo.left;
    const padTop = convInfo.padInfo.top;
    const y = Array4D.zeros(convInfo.outShape);

    for (let b = 0; b < convInfo.batchSize; ++b) {
      for (let d2 = 0; d2 < convInfo.outChannels; ++d2) {
        for (let yR = 0; yR < convInfo.outHeight; ++yR) {
          const xRCorner = yR * convInfo.strideHeight - padLeft;
          const xRMin = Math.max(0, xRCorner);
          const xRMax = Math.min(convInfo.inHeight, filterHeight + xRCorner);
          for (let yC = 0; yC < convInfo.outWidth; ++yC) {
            const xCCorner = yC * convInfo.strideWidth - padTop;
            const xCMin = Math.max(0, xCCorner);
            const xCMax = Math.min(convInfo.inWidth, filterWidth + xCCorner);
            let dotProd = 0;
            for (let xR = xRMin; xR < xRMax; ++xR) {
              const wR = xR - xRCorner;
              for (let xC = xCMin; xC < xCMax; ++xC) {
                const wC = xC - xCCorner;
                for (let d1 = 0; d1 < convInfo.inChannels; ++d1) {
                  const pixel = x.get(b, xR, xC, d1);
                  const weight = filter.get(wR, wC, d1, d2);
                  dotProd += pixel * weight;
                }
              }
            }
            const biasVal = (bias != null) ? bias.get(d2) : 0;
            y.set(dotProd + biasVal, b, yR, yC, d2);
          }
        }
      }
    }
    return y;
  }

  conv2dDerInput(dy: Array4D, filter: Array4D, convInfo: Conv2DInfo): Array4D {
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const topPad = filterHeight - 1 - convInfo.padInfo.top;
    const leftPad = filterWidth - 1 - convInfo.padInfo.left;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dx = Array4D.zeros(convInfo.inShape);
    for (let b = 0; b < convInfo.batchSize; ++b) {
      for (let d1 = 0; d1 < convInfo.inChannels; ++d1) {
        for (let xR = 0; xR < convInfo.inHeight; ++xR) {
          const xRCorner = xR - leftPad;
          const xRMin = Math.max(0, Math.ceil(xRCorner / strideHeight));
          const yRMax = Math.min(
              convInfo.outHeight, (filterHeight + xRCorner) / strideHeight);

          for (let xC = 0; xC < convInfo.inWidth; ++xC) {
            const xCCorner = xC - topPad;
            const xCMin = Math.max(0, Math.ceil(xCCorner / strideWidth));
            const yCMax = Math.min(
                convInfo.outWidth, (filterWidth + xCCorner) / strideWidth);

            let dotProd = 0;
            for (let yR = xRMin; yR < yRMax; ++yR) {
              const wR = yR * strideHeight - xRCorner;

              for (let yC = xCMin; yC < yCMax; ++yC) {
                const wC = yC * strideWidth - xCCorner;

                for (let d2 = 0; d2 < convInfo.outChannels; ++d2) {
                  const pixel = dy.get(b, yR, yC, d2);
                  const weight = filter.get(
                      filterHeight - 1 - wR, filterWidth - 1 - wC, d1, d2);
                  dotProd += pixel * weight;
                }
              }
            }
            dx.set(dotProd, b, xR, xC, d1);
          }
        }
      }
    }
    return dx;
  }

  conv2dDerFilter(x: Array4D, dy: Array4D, convInfo: Conv2DInfo): Array4D {
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const dW = Array4D.zeros(convInfo.filterShape);

    const leftPad = convInfo.padInfo.left;
    const topPad = convInfo.padInfo.top;

    for (let wR = 0; wR < filterHeight; ++wR) {
      const yRMin = Math.max(0, Math.ceil((topPad - wR) / strideHeight));
      const yRMax = Math.min(
          convInfo.outHeight, (convInfo.inHeight + topPad - wR) / strideHeight);

      for (let wC = 0; wC < filterWidth; ++wC) {
        const yCMin = Math.max(0, Math.ceil((leftPad - wC) / strideWidth));
        const yCMax = Math.min(
            convInfo.outWidth, (convInfo.inWidth + leftPad - wC) / strideWidth);

        for (let d1 = 0; d1 < convInfo.inChannels; ++d1) {
          for (let d2 = 0; d2 < convInfo.outChannels; ++d2) {
            // Need to convolve.
            let dotProd = 0;
            for (let b = 0; b < convInfo.batchSize; ++b) {
              for (let yR = yRMin; yR < yRMax; ++yR) {
                const xR = wR + yR * strideHeight - topPad;
                for (let yC = yCMin; yC < yCMax; ++yC) {
                  const xC = wC + yC * strideWidth - leftPad;
                  dotProd += x.get(b, xR, xC, d1) * dy.get(b, yR, yC, d2);
                }
              }
            }
            dW.set(dotProd, wR, wC, d1, d2);
          }
        }
      }
    }
    return dW;
  }

  conv2dDerBias(dy: Array4D): Array1D {
    const [batchSize, numRows, numCols, outDepth] = dy.shape;
    const values = new Float32Array(outDepth);
    for (let d2 = 0; d2 < outDepth; ++d2) {
      let sum = 0;
      for (let b = 0; b < batchSize; ++b) {
        for (let r = 0; r < numRows; ++r) {
          for (let c = 0; c < numCols; ++c) {
            sum += dy.get(b, r, c, d2);
          }
        }
      }
      values[d2] = sum;
    }
    return Array1D.new(values);
  }

  depthwiseConv2D(x: Array4D, filter: Array4D, convInfo: Conv2DInfo): Array4D {
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const padLeft = convInfo.padInfo.left;
    const padTop = convInfo.padInfo.top;
    const chMul = convInfo.outChannels / convInfo.inChannels;
    const y = Array4D.zeros(convInfo.outShape);

    for (let b = 0; b < convInfo.batchSize; ++b) {
      for (let d1 = 0; d1 < convInfo.inChannels; ++d1) {
        for (let yR = 0; yR < convInfo.outHeight; ++yR) {
          const xRCorner = yR * convInfo.strideHeight - padLeft;
          const xRMin = Math.max(0, xRCorner);
          const xRMax = Math.min(convInfo.inHeight, filterHeight + xRCorner);
          for (let yC = 0; yC < convInfo.outWidth; ++yC) {
            const xCCorner = yC * convInfo.strideWidth - padTop;
            const xCMin = Math.max(0, xCCorner);
            const xCMax = Math.min(convInfo.inWidth, filterWidth + xCCorner);
            for (let q = 0; q < chMul; ++q) {
              let dotProd = 0;
              for (let xR = xRMin; xR < xRMax; ++xR) {
                const wR = xR - xRCorner;
                for (let xC = xCMin; xC < xCMax; ++xC) {
                  const wC = xC - xCCorner;
                  const pixel = x.get(b, xR, xC, d1);
                  const weight = filter.get(wR, wC, d1, q);
                  dotProd += pixel * weight;
                }
              }
              y.set(dotProd, b, yR, yC, d1 * chMul + q);
            }
          }
        }
      }
    }
    return y;
  }

  tile<D extends DataType, T extends NDArray<D>>(x: T, reps: number[]): T {
    const newShape: number[] = new Array(x.rank);
    for (let i = 0; i < newShape.length; i++) {
      newShape[i] = x.shape[i] * reps[i];
    }
    const result = NDArray.zeros(newShape, x.dtype);
    const newValues = result.dataSync();
    const values = x.dataSync();
    for (let i = 0; i < result.size; ++i) {
      const newLoc = result.indexToLoc(i);

      const originalLoc: number[] = new Array(x.rank);
      for (let i = 0; i < originalLoc.length; i++) {
        originalLoc[i] = newLoc[i] % x.shape[i];
      }

      const originalIndex = x.locToIndex(originalLoc);

      newValues[i] = values[originalIndex];
    }
    return result as T;
  }

  pad1D(x: Array1D, paddings: [number, number], constantValue: number):
      Array1D {
    const leftPadding = paddings[0];
    const rightPadding = paddings[1];

    const values = x.dataSync();
    const result =
        Array1D.zeros([leftPadding + values.length + rightPadding], x.dtype);
    const newValues = result.dataSync();

    let z = 0;
    for (let i = 0; i < newValues.length; i++) {
      if (i >= leftPadding && i < leftPadding + values.length) {
        newValues[i] = values[z++];
      } else {
        newValues[i] = constantValue;
      }
    }
    return result;
  }

  pad2D(
      x: Array2D, paddings: [[number, number], [number, number]],
      constantValue: number): Array2D {
    const topPadding = paddings[0][0];
    const bottomPadding = paddings[0][1];
    const leftPadding = paddings[1][0];
    const rightPadding = paddings[1][1];

    const newShape: [number, number] = [
      topPadding + x.shape[0] + bottomPadding,
      leftPadding + x.shape[1] + rightPadding
    ];

    const result = Array2D.zeros(newShape, x.dtype);
    const newValues = result.dataSync();

    const values = x.dataSync();

    let z = 0;
    for (let i = 0; i < newShape[0]; i++) {
      let rangeStart = -1;
      let rangeEnd = -1;

      if (i >= topPadding && i < newShape[0] - bottomPadding) {
        rangeStart = i * newShape[1] + leftPadding;
        rangeEnd = rangeStart + x.shape[1] - 1;
      }

      for (let j = 0; j < newShape[1]; j++) {
        const v = i * newShape[1] + j;
        if (v >= rangeStart && v <= rangeEnd) {
          newValues[v] = values[z++];
        } else {
          newValues[v] = constantValue;
        }
      }
    }
    return result;
  }

  transpose<D extends DataType, T extends NDArray<D>>(x: T, perm: number[]): T {
    const newShape: number[] = new Array(x.rank);
    for (let i = 0; i < newShape.length; i++) {
      newShape[i] = x.shape[perm[i]];
    }
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    const result = NDArray.make(newShape, {values: resultValues}) as T;
    for (let i = 0; i < x.size; ++i) {
      const loc = x.indexToLoc(i);

      // Permute location.
      const newLoc: number[] = new Array(loc.length);
      for (let i = 0; i < newLoc.length; i++) {
        newLoc[i] = loc[perm[i]];
      }

      const newIndex = result.locToIndex(newLoc);
      resultValues[newIndex] = values[i];
    }
    return result;
  }

  gather<D extends DataType, T extends NDArray<D>>(
      x: T, indices: Array1D<'int32'>, axis: number): T {
    const newShape: number[] = x.shape.slice();
    const indicesValues = indices.dataSync();
    newShape[axis] = indicesValues.length;
    const result = NDArray.zeros(newShape, x.dtype) as T;
    const values = x.dataSync();
    const resultValues = result.dataSync();
    for (let i = 0; i < result.size; ++i) {
      const newLoc = result.indexToLoc(i);

      const originalLoc: number[] = newLoc.slice();
      originalLoc[axis] = indicesValues[newLoc[axis]];

      const originalIndex = x.locToIndex(originalLoc);
      resultValues[i] = values[originalIndex];
    }
    return result;
  }

  private pool(x: Array4D, convInfo: Conv2DInfo, poolType: 'max'|'min'|'avg') {
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const y = Array4D.zeros(convInfo.outShape);
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;
    for (let b = 0; b < convInfo.batchSize; ++b) {
      for (let d = 0; d < convInfo.inChannels; ++d) {
        for (let yR = 0; yR < convInfo.outHeight; ++yR) {
          const xRCorner = yR * strideHeight - padTop;
          const xRMin = Math.max(0, xRCorner);
          const xRMax = Math.min(convInfo.inHeight, filterHeight + xRCorner);
          for (let yC = 0; yC < convInfo.outWidth; ++yC) {
            const xCCorner = yC * strideWidth - padLeft;
            const xCMin = Math.max(0, xCCorner);
            const xCMax = Math.min(convInfo.inWidth, filterWidth + xCCorner);

            let minMaxValue =
                (poolType === 'max' ? Number.NEGATIVE_INFINITY :
                                      Number.POSITIVE_INFINITY);
            let avgValue = 0;
            for (let xR = xRMin; xR < xRMax; ++xR) {
              for (let xC = xCMin; xC < xCMax; ++xC) {
                const pixel = x.get(b, xR, xC, d);
                if (isNaN(pixel)) {
                  minMaxValue = NaN;
                  avgValue = NaN;
                  break;
                }
                if ((poolType === 'max' && pixel > minMaxValue) ||
                    (poolType === 'min' && pixel < minMaxValue)) {
                  minMaxValue = pixel;
                } else if (poolType === 'avg') {
                  avgValue += pixel / (filterHeight * filterWidth);
                }
              }
              if (isNaN(minMaxValue)) {
                break;
              }
            }
            y.set(poolType === 'avg' ? avgValue : minMaxValue, b, yR, yC, d);
          }
        }
      }
    }
    return y;
  }

  maxPool(x: Array4D, convInfo: Conv2DInfo): Array4D {
    return this.pool(x, convInfo, 'max');
  }

  maxPoolPositions(x: Array4D, convInfo: Conv2DInfo) {
    const maxPositions = Array4D.zeros(convInfo.outShape);
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;

    for (let b = 0; b < convInfo.batchSize; ++b) {
      for (let d = 0; d < convInfo.inChannels; ++d) {
        for (let yR = 0; yR < convInfo.outHeight; ++yR) {
          const xRCorner = yR * strideHeight - padTop;
          const xRMin = Math.max(0, xRCorner);
          const xRMax = Math.min(convInfo.inHeight, filterHeight + xRCorner);
          for (let yC = 0; yC < convInfo.outWidth; ++yC) {
            const xCCorner = yC * strideWidth - padLeft;
            const xCMin = Math.max(0, xCCorner);
            const xCMax = Math.min(convInfo.inWidth, filterWidth + xCCorner);
            let maxValue = Number.NEGATIVE_INFINITY;
            let maxPosition = -1;
            for (let xR = xRMin; xR < xRMax; ++xR) {
              const wR = xR - xRCorner;
              for (let xC = xCMin; xC < xCMax; ++xC) {
                const wC = xC - xCCorner;
                const pixel = x.get(b, xR, xC, d);
                if (pixel > maxValue) {
                  maxValue = pixel;
                  maxPosition = wR * filterWidth + wC;
                }
              }
            }
            maxPositions.set(maxPosition, b, yR, yC, d);
          }
        }
      }
    }
    return maxPositions;
  }

  maxPoolBackprop(dy: Array4D, x: Array4D, convInfo: Conv2DInfo): Array4D {
    const maxPositions = this.maxPoolPositions(x, convInfo);
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const padLeft = filterWidth - 1 - convInfo.padInfo.left;
    const padTop = filterHeight - 1 - convInfo.padInfo.top;
    const dx = Array4D.zeros(x.shape);

    for (let b = 0; b < convInfo.batchSize; ++b) {
      for (let d = 0; d < convInfo.inChannels; ++d) {
        for (let dxR = 0; dxR < convInfo.inHeight; ++dxR) {
          for (let dxC = 0; dxC < convInfo.inWidth; ++dxC) {
            // Shader code begins.
            const dyRCorner = dxR - padTop;
            const dyCCorner = dxC - padLeft;
            let dotProd = 0;
            for (let wR = 0; wR < filterHeight; ++wR) {
              const dyR = (dyRCorner + wR) / strideHeight;
              if (dyR < 0 || dyR >= convInfo.outHeight ||
                  Math.floor(dyR) !== dyR) {
                continue;
              }
              for (let wC = 0; wC < filterWidth; ++wC) {
                const dyC = (dyCCorner + wC) / strideWidth;
                if (dyC < 0 || dyC >= convInfo.outWidth ||
                    Math.floor(dyC) !== dyC) {
                  continue;
                }
                const maxPos = filterHeight * filterWidth - 1 -
                    maxPositions.get(b, dyR, dyC, d);
                const curPos = wR * filterWidth + wC;

                const mask = maxPos === curPos ? 1 : 0;
                if (mask === 0) {
                  continue;
                }

                const pixel = dy.get(b, dyR, dyC, d);
                dotProd += pixel * mask;
              }
            }
            dx.set(dotProd, b, dxR, dxC, d);
          }
        }
      }
    }
    return dx.asType(x.dtype);
  }

  avgPoolBackprop(dy: Array4D, x: Array4D, convInfo: Conv2DInfo): Array4D {
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const padLeft = filterWidth - 1 - convInfo.padInfo.left;
    const padTop = filterHeight - 1 - convInfo.padInfo.top;
    const dx = Array4D.zeros(x.shape);

    const avgMultiplier = 1 / (filterHeight * filterWidth);

    for (let b = 0; b < convInfo.batchSize; ++b) {
      for (let d = 0; d < convInfo.inChannels; ++d) {
        for (let dxR = 0; dxR < convInfo.inHeight; ++dxR) {
          for (let dxC = 0; dxC < convInfo.inWidth; ++dxC) {
            // Shader code begins.
            const dyRCorner = dxR - padTop;
            const dyCCorner = dxC - padLeft;
            let dotProd = 0;
            for (let wR = 0; wR < filterHeight; ++wR) {
              const dyR = (dyRCorner + wR) / strideHeight;
              if (dyR < 0 || dyR >= convInfo.outHeight ||
                  Math.floor(dyR) !== dyR) {
                continue;
              }
              for (let wC = 0; wC < filterWidth; ++wC) {
                const dyC = (dyCCorner + wC) / strideWidth;
                if (dyC < 0 || dyC >= convInfo.outWidth ||
                    Math.floor(dyC) !== dyC) {
                  continue;
                }

                const pixel = dy.get(b, dyR, dyC, d);
                dotProd += pixel;
              }
            }
            dx.set(dotProd * avgMultiplier, b, dxR, dxC, d);
          }
        }
      }
    }
    return dx.asType(x.dtype);
  }

  minPool(x: Array4D, convInfo: Conv2DInfo): Array4D {
    return this.pool(x, convInfo, 'min');
  }

  avgPool(x: Array4D, convInfo: Conv2DInfo): Array4D {
    return this.pool(x, convInfo, 'avg').asType('float32');
  }

  resizeBilinear3D(
      x: Array3D, newShape2D: [number, number],
      alignCorners: boolean): Array3D {
    const output = Array3D.zeros([newShape2D[0], newShape2D[1], x.shape[2]]);

    const effectiveInputSize =
        alignCorners ? [x.shape[0] - 1, x.shape[1] - 1, x.shape[2]] : x.shape;
    const effectiveOutputSize = alignCorners ?
        [output.shape[0] - 1, output.shape[1] - 1, output.shape[2]] :
        output.shape;
    for (let r = 0; r < output.shape[0]; r++) {
      for (let c = 0; c < output.shape[1]; c++) {
        for (let d = 0; d < output.shape[2]; d++) {
          // Begin shader.

          // Compute the fractional index of the source.
          const sourceFracRow =
              (effectiveInputSize[0]) * r / (effectiveOutputSize[0]);
          const sourceFracCol =
              (effectiveInputSize[1]) * c / (effectiveOutputSize[1]);

          const sourceRowFloor = Math.floor(sourceFracRow);
          const sourceRowCeil =
              Math.min(x.shape[0] - 1, Math.ceil(sourceFracRow));
          const sourceColFloor = Math.floor(sourceFracCol);
          const sourceColCeil =
              Math.min(x.shape[1] - 1, Math.ceil(sourceFracCol));

          const topLeft = x.get(sourceRowFloor, sourceColFloor, d);
          const bottomLeft = x.get(sourceRowCeil, sourceColFloor, d);
          const topRight = x.get(sourceRowFloor, sourceColCeil, d);
          const bottomRight = x.get(sourceRowCeil, sourceColCeil, d);

          const rowFrac = sourceFracRow - sourceRowFloor;
          const colFrac = sourceFracCol - sourceColFloor;

          const top = topLeft + (topRight - topLeft) * colFrac;
          const bottom = bottomLeft + (bottomRight - bottomLeft) * colFrac;
          const newValue = top + (bottom - top) * rowFrac;

          output.set(newValue, r, c, d);
        }
      }
    }

    return output;
  }

  batchNormalization2D(
      x: Array2D, mean: Array2D|Array1D, variance: Array2D|Array1D,
      varianceEpsilon: number, scale?: Array2D|Array1D,
      offset?: Array2D|Array1D): Array2D {
    const xValues = x.dataSync();
    const meanValues = mean.dataSync();
    const varianceValues = variance.dataSync();
    const scaleValues = scale ? scale.dataSync() : [1];
    const offsetValues = offset ? offset.dataSync() : [0];
    const outValues = new Float32Array(xValues.length);

    for (let i = 0; i < xValues.length; i++) {
      outValues[i] = offsetValues[i % offsetValues.length] +
          (xValues[i] - meanValues[i % meanValues.length]) *
              scaleValues[i % scaleValues.length] /
              Math.sqrt(
                  varianceValues[i % varianceValues.length] + varianceEpsilon);
    }
    return Array2D.new(x.shape, outValues);
  }

  batchNormalization3D(
      x: Array3D, mean: Array3D|Array1D, variance: Array3D|Array1D,
      varianceEpsilon: number, scale?: Array3D|Array1D,
      offset?: Array3D|Array1D): Array3D {
    const xValues = x.dataSync();
    const meanValues = mean.dataSync();
    const varianceValues = variance.dataSync();
    const scaleValues = scale ? scale.dataSync() : [1];
    const offsetValues = offset ? offset.dataSync() : [0];
    const outValues = new Float32Array(xValues.length);

    for (let i = 0; i < xValues.length; i++) {
      outValues[i] = offsetValues[i % offsetValues.length] +
          (xValues[i] - meanValues[i % meanValues.length]) *
              scaleValues[i % scaleValues.length] /
              Math.sqrt(
                  varianceValues[i % varianceValues.length] + varianceEpsilon);
    }
    return Array3D.new(x.shape, outValues);
  }

  batchNormalization4D(
      x: Array4D, mean: Array4D|Array1D, variance: Array4D|Array1D,
      varianceEpsilon: number, scale?: Array4D|Array1D,
      offset?: Array4D|Array1D): Array4D {
    const xValues = x.dataSync();
    const meanValues = mean.dataSync();
    const varianceValues = variance.dataSync();
    const scaleValues = scale ? scale.dataSync() : new Float32Array([1]);
    const offsetValues = offset ? offset.dataSync() : new Float32Array([0]);
    const outValues = new Float32Array(xValues.length);

    for (let i = 0; i < xValues.length; i++) {
      outValues[i] = offsetValues[i % offsetValues.length] +
          (xValues[i] - meanValues[i % meanValues.length]) *
              scaleValues[i % scaleValues.length] /
              Math.sqrt(
                  varianceValues[i % varianceValues.length] + varianceEpsilon);
    }
    return Array4D.new(x.shape, outValues);
  }

  localResponseNormalization4D(
      x: Array4D, radius: number, bias: number, alpha: number, beta: number,
      normRegion: 'acrossChannels'|'withinChannel'): Array4D {
    const output = Array4D.zeros(x.shape);
    const rad = radius;
    const maxW = output.shape[1] - 1;
    const maxH = output.shape[2] - 1;
    const maxD = output.shape[3] - 1;

    const sumAcrossChannels =
        (b: number, r: number, c: number, d: number): number => {
          let sum = 0.0;
          for (let j = Math.max(0, d - rad); j <= Math.min(d + rad, maxD);
               j++) {
            const z = x.get(b, r, c, j);
            sum += z * z;
          }
          return sum;
        };

    const sumWithinChannel =
        (b: number, r: number, c: number, d: number): number => {
          let sum = 0.0;
          for (let u = Math.max(0, r - rad); u <= Math.min(r + rad, maxW);
               u++) {
            for (let v = Math.max(0, c - rad); v <= Math.min(c + rad, maxH);
                 v++) {
              sum += Math.pow(x.get(b, u, v, d), 2);
            }
          }
          return sum;
        };

    for (let b = 0; b < output.shape[0]; b++) {
      for (let r = 0; r <= output.shape[1]; r++) {
        for (let c = 0; c < output.shape[2]; c++) {
          for (let d = 0; d < output.shape[3]; d++) {
            const sum = normRegion === 'withinChannel' ?
                sumWithinChannel(b, r, c, d) :
                sumAcrossChannels(b, r, c, d);
            const val = x.get(b, r, c, d) * Math.pow(bias + alpha * sum, -beta);
            output.set(val, b, r, c, d);
          }
        }
      }
    }

    return output;
  }

  multinomial(probabilities: Array2D, numSamples: number, seed: number):
      Array2D<'int32'> {
    const batchSize = probabilities.shape[0];
    const numEvents = probabilities.shape[1];
    const res = Array2D.zeros([batchSize, numSamples], 'int32');
    const resVals = res.dataSync();
    const probVals = probabilities.dataSync();

    for (let b = 0; b < batchSize; ++b) {
      const offset = b * numEvents;
      // The cdf won't include the last event. It will be implicit if no other
      // event happened.
      const cdf = new Float32Array(numEvents - 1);
      cdf[0] = probVals[offset];
      for (let event = 1; event < cdf.length; ++event) {
        cdf[event] = cdf[event - 1] + probVals[offset + event];
      }

      const random = seedrandom.alea(seed.toString());
      const outOffset = b * numSamples;
      for (let sampleId = 0; sampleId < numSamples; ++sampleId) {
        const r = random();

        // Assume last event happened by default.
        resVals[outOffset + sampleId] = cdf.length;

        for (let event = 0; event < cdf.length; event++) {
          if (r < cdf[event]) {
            resVals[outOffset + sampleId] = event;
            break;
          }
        }
      }
    }
    return res;
  }

  oneHot(indices: Array1D, depth: number, onValue: number, offValue: number):
      Array2D {
    const res = new Float32Array(indices.size * depth);
    res.fill(offValue);

    for (let event = 0; event < indices.size; ++event) {
      res[event * depth + indices.get(event)] = onValue;
    }
    return Array2D.new([indices.size, depth], res);
  }

  private broadcastedBinaryOp<D extends DataType>(
      a: NDArray, b: NDArray, dtype: D,
      op: (a: number, b: number) => number): NDArray<D> {
    const newShape =
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    const result = NDArray.zeros(newShape, dtype);
    const newValues = result.dataSync();
    const aValues = a.dataSync();
    const bValues = b.dataSync();

    const aBroadcastDims = broadcast_util.getBroadcastDims(a.shape, newShape);
    const bBroadcastDims = broadcast_util.getBroadcastDims(b.shape, newShape);

    for (let i = 0; i < newValues.length; ++i) {
      const loc = result.indexToLoc(i);

      const aLoc = loc.slice(-a.rank);
      aBroadcastDims.forEach(d => aLoc[d] = 0);
      const aIndex = a.locToIndex(aLoc);

      const bLoc = loc.slice(-b.rank);
      bBroadcastDims.forEach(d => bLoc[d] = 0);
      const bIndex = b.locToIndex(bLoc);

      newValues[i] = op(aValues[aIndex], bValues[bIndex]);
    }
    return result;
  }
  dispose() {}
}

ENV.registerBackend('cpu', () => new MathBackendCPU());

// TODO(nsthorat): Deprecate this once we export non-abstract NDArrayMath.
export class NDArrayMathCPU extends NDArrayMath {
  constructor(safeMode = false) {
    console.warn(
        'new NDArrayMathCPU() is deprecated. Please use the global ' +
        'dl.ENV.math. In rare cases, to construct your own NDArrayMath ' +
        'that runs on CPU, use math = new NDArrayMath(\'cpu\', safeMode); ' +
        'and make sure to set it as global: dl.ENV.setMath(math);');
    super('cpu', safeMode);
    ENV.setMath(this);
  }
}
