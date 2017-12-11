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

import * as util from '../../util';
import * as broadcast_util from '../broadcast_util';
import * as concat_util from '../concat_util';
import {Conv2DInfo} from '../conv_util';
import * as copy2D_util from '../copy2d_util';
import {NDArrayMath} from '../math';
// tslint:disable-next-line:max-line-length
import {Array1D, Array2D, Array3D, Array4D, DataTypes, NDArray, Scalar} from '../ndarray';
import {SumTypes, SumTypesMap} from '../types';

import * as axis_util from './../axis_util';
import {MathBackend, MatrixOrientation} from './backend';

export class MathBackendCPU implements MathBackend {
  clone<T extends NDArray>(ndarray: T): T {
    return NDArray.make(
               ndarray.shape,
               {values: new Float32Array(ndarray.getValues())}) as T;
  }

  slice1D(input: Array1D, begin: number, size: number): Array1D {
    const newVals = input.getValues().slice(begin, begin + size);
    return Array1D.new(newVals);
  }

  slice2D(input: Array2D, begin: [number, number], size: [number, number]):
      Array2D {
    const result = Array2D.zeros(size);
    const [startI, startJ] = begin;

    for (let i = 0; i < size[0]; ++i) {
      for (let j = 0; j < size[1]; ++j) {
        const val = input.get(i + startI, j + startJ);
        result.set(val, i, j);
      }
    }
    return result;
  }

  slice3D(input: Array3D, begin: [number, number, number], size: [
    number, number, number
  ]): Array3D {
    const result = Array3D.zeros(size);
    const [startI, startJ, startK] = begin;

    for (let i = 0; i < size[0]; ++i) {
      for (let j = 0; j < size[1]; ++j) {
        for (let k = 0; k < size[2]; ++k) {
          const val = input.get(i + startI, j + startJ, k + startK);
          result.set(val, i, j, k);
        }
      }
    }
    return result;
  }

  slice4D(input: Array4D, begin: [number, number, number, number], size: [
    number, number, number, number
  ]): Array4D {
    const result = Array4D.zeros(size);
    const [startI, startJ, startK, startL] = begin;

    for (let i = 0; i < size[0]; ++i) {
      for (let j = 0; j < size[1]; ++j) {
        for (let k = 0; k < size[2]; ++k) {
          for (let l = 0; l < size[3]; ++l) {
            const val =
                input.get(i + startI, j + startJ, k + startK, l + startL);
            result.set(val, i, j, k, l);
          }
        }
      }
    }
    return result;
  }

  copy2D(
      source: Array2D, sourceBeginRowCol: [number, number],
      sourceSizeRowCol: [number, number], dest: Array2D,
      destBeginRowCol: [number, number],
      destSizeRowCol: [number, number]): void {
    copy2D_util.validateShapes(sourceSizeRowCol, destSizeRowCol);
    const srcValues = source.getValues();
    const dstValues = dest.getValues();
    const n = sourceSizeRowCol[0] * sourceSizeRowCol[1];
    for (let i = 0; i < n; ++i) {
      const srcRow = sourceBeginRowCol[0] + Math.floor(i / sourceSizeRowCol[1]);
      const srcCol = sourceBeginRowCol[1] + (i % sourceSizeRowCol[1]);
      const srcOff = srcRow * source.shape[1] + srcCol;
      const dstRow = destBeginRowCol[0] + Math.floor(i / destSizeRowCol[1]);
      const dstCol = destBeginRowCol[1] + (i % destSizeRowCol[1]);
      const dstOff = dstRow * dest.shape[1] + dstCol;
      dstValues[dstOff] = srcValues[srcOff];
    }
  }

  concat1D(a: Array1D, b: Array1D): Array1D {
    const outShape = concat_util.computeOutShape(a.shape, b.shape, 0);
    const result = Array1D.zeros(outShape as [number]);

    // Use built-in TypedArray.set() method for speed.
    const aVals = a.getValues();
    const bVals = b.getValues();
    const vals = result.getValues();
    vals.set(aVals, 0);
    vals.set(bVals, a.size);

    return result;
  }

  concat2D(a: Array2D, b: Array2D, axis: number): Array2D {
    const outShape = concat_util.computeOutShape(a.shape, b.shape, axis);
    const result = Array2D.zeros(outShape as [number, number]);

    if (axis === 0) {
      // Use built-in TypedArray.set() method for speed.
      const aVals = a.getValues();
      const bVals = b.getValues();
      const vals = result.getValues();
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
      const aVals = a.getValues();
      const bVals = b.getValues();
      const vals = result.getValues();
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
      const aVals = a.getValues();
      const bVals = b.getValues();
      const vals = result.getValues();
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

  scaledArrayAdd<T extends NDArray>(c1: Scalar, a: T, c2: Scalar, b: T): T {
    const c1Val = c1.get();
    const c2Val = c2.get();
    return this.broadcastedBinaryOp(a, b, 'float32', (aVal, bVal) => {
      return c1Val * aVal + c2Val * bVal;
    }) as T;
  }

  neg<T extends NDArray>(a: T): T {
    return this.multiply(Scalar.NEG_ONE, a) as T;
  }

  add<T extends NDArray>(a: T, b: T): T {
    return this.scaledArrayAdd<T>(Scalar.ONE, a, Scalar.ONE, b);
  }

  subtract<T extends NDArray>(a: T, b: T): T {
    return this.scaledArrayAdd<T>(Scalar.ONE, a, Scalar.NEG_ONE, b);
  }

  pow<T extends NDArray>(a: T, b: NDArray<'int32'>): T {
    const newShape =
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    const newValues = new Float32Array(util.sizeFromShape(newShape));

    const aValues = a.getValues();
    const bValues = b.getValues();
    for (let i = 0; i < newValues.length; ++i) {
      newValues[i] = Math.pow(aValues[i % a.size], bValues[i % b.size]);
    }
    return NDArray.make(newShape, {values: newValues}) as T;
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

  multiply<T extends NDArray>(a: T, b: T): T {
    const newShape =
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    const newValues = new Float32Array(util.sizeFromShape(newShape));

    const aValues = a.getValues();
    const bValues = b.getValues();
    for (let i = 0; i < newValues.length; ++i) {
      newValues[i] = aValues[i % a.size] * bValues[i % b.size];
    }
    return NDArray.make(newShape, {values: newValues}) as T;
  }

  divide(a: NDArray, b: NDArray): NDArray<'float32'> {
    const newShape =
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    const newValues = new Float32Array(util.sizeFromShape(newShape));

    const aValues = a.getValues();
    const bValues = b.getValues();

    for (let i = 0; i < newValues.length; ++i) {
      newValues[i] = aValues[i % a.size] / bValues[i % b.size];
    }
    return NDArray.make(newShape, {values: newValues}, 'float32');
  }

  sum<T extends keyof DataTypes>(input: NDArray<T>, axes: number[]):
      NDArray<SumTypes[T]> {
    axis_util.assertAxesAreInnerMostDims('sum', axes, input.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(input.shape, axes);
    const resultDtype = SumTypesMap[input.dtype] as keyof SumTypes;
    const result = NDArray.zeros(outShape, resultDtype);
    const reduceSize = util.sizeFromShape(reduceShape);
    const vals = result.getValues();

    const aVals = input.getValues();
    for (let i = 0; i < vals.length; ++i) {
      const offset = i * reduceSize;
      let sum = 0;
      for (let j = 0; j < reduceSize; ++j) {
        sum += aVals[offset + j];
      }
      vals[i] = sum;
    }
    return result as NDArray<SumTypes[T]>;
  }

  argMin(input: NDArray, axes: number[]): NDArray<'int32'> {
    axis_util.assertAxesAreInnerMostDims('argMin', axes, input.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(input.shape, axes);
    const result = NDArray.zeros(outShape, 'int32');
    const reduceSize = util.sizeFromShape(reduceShape);
    const vals = result.getValues();

    const aVals = input.getValues();
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

  argMax(input: NDArray, axes: number[]): NDArray<'int32'> {
    axis_util.assertAxesAreInnerMostDims('argMax', axes, input.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(input.shape, axes);
    const result = NDArray.zeros(outShape, 'int32');
    const reduceSize = util.sizeFromShape(reduceShape);
    const vals = result.getValues();

    const aVals = input.getValues();
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

  topKValues<D extends keyof DataTypes, T extends NDArray<D>>(
      ndarray: T, k: number): Array1D<D> {
    return this.topK(ndarray, k).values as Array1D<D>;
  }

  topKIndices(ndarray: NDArray, k: number): Array1D<'int32'> {
    return this.topK(ndarray, k).indices;
  }

  private topK<D extends keyof DataTypes, T extends NDArray<D>>(
      ndarray: T, k: number): {values: Array1D<D>, indices: Array1D<'int32'>} {
    const values = ndarray.getValues();
    const valuesAndIndices: Array<{value: number, index: number}> = [];
    for (let i = 0; i < values.length; i++) {
      valuesAndIndices.push({value: values[i], index: i});
    }
    valuesAndIndices.sort((a, b) => {
      return b.value - a.value;
    });

    const topkValues = util.getTypedArrayFromDType(ndarray.dtype, k);
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

  min<G extends keyof DataTypes>(input: NDArray<G>, axes: number[]):
      NDArray<G> {
    axis_util.assertAxesAreInnerMostDims('min', axes, input.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(input.shape, axes);
    const result = NDArray.zeros(outShape, input.dtype);
    const reduceSize = util.sizeFromShape(reduceShape);
    const vals = result.getValues();

    const aVals = input.getValues();
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

  max<G extends keyof DataTypes>(input: NDArray<G>, axes: number[]):
      NDArray<G> {
    axis_util.assertAxesAreInnerMostDims('max', axes, input.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(input.shape, axes);
    const result = NDArray.zeros(outShape, input.dtype);
    const reduceSize = util.sizeFromShape(reduceShape);
    const vals = result.getValues();

    const aVals = input.getValues();
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

  ceil<T extends NDArray>(ndarray: T): T {
    const values = ndarray.getValues();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      newValues[i] = Math.ceil(values[i]);
    }
    return NDArray.make(ndarray.shape, {values: newValues}) as T;
  }

  floor<T extends NDArray>(ndarray: T): T {
    const values = ndarray.getValues();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      newValues[i] = Math.floor(values[i]);
    }
    return NDArray.make(ndarray.shape, {values: newValues}) as T;
  }

  exp<T extends NDArray>(ndarray: T): T {
    const values = ndarray.getValues();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      newValues[i] = Math.exp(values[i]);
    }
    return NDArray.make(ndarray.shape, {values: newValues}) as T;
  }

  log<T extends NDArray>(ndarray: T): T {
    const values = ndarray.getValues();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      const value = values[i];
      newValues[i] = Math.log(value);
    }
    return NDArray.make(ndarray.shape, {values: newValues}) as T;
  }

  sqrt<T extends NDArray>(ndarray: T): T {
    const values = ndarray.getValues();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      const value = values[i];
      newValues[i] = Math.sqrt(value);
    }
    return NDArray.make(ndarray.shape, {values: newValues}) as T;
  }

  square<T extends NDArray>(x: T): T {
    const values = x.getValues();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      const value = values[i];
      newValues[i] = value * value;
    }
    return NDArray.make(x.shape, {values: newValues}) as T;
  }

  relu<T extends NDArray>(input: T): T {
    const res = NDArray.zeros(input.shape, input.dtype);
    const resVals = res.getValues();
    const inVals = input.getValues();
    for (let i = 0; i < inVals.length; ++i) {
      const val = inVals[i];
      if (util.isValNaN(val, input.dtype)) {
        resVals[i] = util.getNaN(res.dtype);
      } else {
        resVals[i] = Math.max(0, inVals[i]);
      }
    }
    return res as T;
  }

  elu<T extends NDArray>(ndarray: T): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.dataSync();
    for (let i = 0; i < values.length; ++i) {
      const v = values[i];
      if (v >= 0) {
        resultValues[i] = v;
      } else {
        resultValues[i] = (Math.exp(v) - 1);
      }
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
  }

  eluDer<T extends NDArray>(ndarray: T): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.dataSync();
    for (let i = 0; i < values.length; ++i) {
      const v = values[i];
      if (v >= 0) {
        resultValues[i] = 1;
      } else {
        resultValues[i] = Math.exp(v);
      }
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
  }

  selu<T extends NDArray>(ndarray: T): T {
    // Stable and Attracting Fixed Point (0, 1) for Normalized Weights.
    // see: https://arxiv.org/abs/1706.02515
    const scaleAlpha = 1.7580993408473768599402175208123;
    const scale = 1.0507009873554804934193349852946;

    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.dataSync();
    for (let i = 0; i < values.length; ++i) {
      const v = values[i];
      if (v >= 0) {
        resultValues[i] = scale * v;
      } else {
        resultValues[i] = scaleAlpha * (Math.exp(v) - 1);
      }
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
  }

  leakyRelu<T extends NDArray>(ndarray: T, alpha: number) {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.dataSync();
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (v >= 0) {
        resultValues[i] = v;
      } else {
        resultValues[i] = alpha * v;
      }
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
  }

  clip<T extends NDArray>(ndarray: T, min: number, max: number): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.min(max, Math.max(min, values[i]));
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
  }

  abs<T extends NDArray>(ndarray: T): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.abs(values[i]);
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
  }

  sigmoid<T extends NDArray>(ndarray: T): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = 1 / (1 + Math.exp(-values[i]));
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
  }

  sin<T extends NDArray>(ndarray: T): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.sin(values[i]);
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
  }

  cos<T extends NDArray>(ndarray: T): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.cos(values[i]);
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
  }

  tan<T extends NDArray>(ndarray: T): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.tan(values[i]);
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
  }

  asin<T extends NDArray>(ndarray: T): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.asin(values[i]);
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
  }

  acos<T extends NDArray>(ndarray: T): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.acos(values[i]);
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
  }

  atan<T extends NDArray>(ndarray: T): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.atan(values[i]);
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
  }

  sinh<T extends NDArray>(ndarray: T): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.sinh(values[i]);
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
  }

  cosh<T extends NDArray>(ndarray: T): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.cosh(values[i]);
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
  }

  tanh<T extends NDArray>(ndarray: T): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = util.tanh(values[i]);
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
  }

  step<T extends NDArray>(ndarray: T, alpha = 0): T {
    const resultValues = new Float32Array(ndarray.size);
    const values = ndarray.getValues();
    for (let i = 0; i < values.length; ++i) {
      const value = values[i];
      resultValues[i] = value > 0 ? 1 : (value < 0 ? alpha : value);
    }
    return NDArray.make(ndarray.shape, {values: resultValues}) as T;
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

  conv2dDerFilter(x: Array4D, dY: Array4D, convInfo: Conv2DInfo): Array4D {
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
                  dotProd += x.get(b, xR, xC, d1) * dY.get(b, yR, yC, d2);
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

  conv2dDerBias(dY: Array4D): Array1D {
    const [batchSize, numRows, numCols, outDepth] = dY.shape;
    const values = new Float32Array(outDepth);
    for (let d2 = 0; d2 < outDepth; ++d2) {
      let sum = 0;
      for (let b = 0; b < batchSize; ++b) {
        for (let r = 0; r < numRows; ++r) {
          for (let c = 0; c < numCols; ++c) {
            sum += dY.get(b, r, c, d2);
          }
        }
      }
      values[d2] = sum;
    }
    return Array1D.new(values);
  }

  depthwiseConv2D(input: Array4D, filter: Array4D, convInfo: Conv2DInfo):
      Array4D {
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
                  const pixel = input.get(b, xR, xC, d1);
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

  tile<D extends keyof DataTypes, T extends NDArray<D>>(a: T, reps: number[]):
      T {
    const newShape: number[] = new Array(a.rank);
    for (let i = 0; i < newShape.length; i++) {
      newShape[i] = a.shape[i] * reps[i];
    }
    let dtype;
    if (a.dtype === 'float32') {
      dtype = Float32Array;
    } else if (a.dtype === 'int32') {
      dtype = Int32Array;
    } else if (a.dtype === 'bool') {
      dtype = Uint8Array;
    } else {
      throw new Error(`Dtype ${a.dtype} not supported for tile`);
    }
    const resultValues = new dtype(util.sizeFromShape(newShape));
    const result = NDArray.make(newShape, {values: resultValues}, a.dtype) as T;
    const values = a.getValues();
    for (let i = 0; i < result.size; ++i) {
      const newLoc = result.indexToLoc(i);

      const originalLoc: number[] = new Array(a.rank);
      for (let i = 0; i < originalLoc.length; i++) {
        originalLoc[i] = newLoc[i] % a.shape[i];
      }

      const originalIndex = a.locToIndex(originalLoc);

      resultValues[i] = values[originalIndex];
    }
    return result;
  }

  transpose<D extends keyof DataTypes, T extends NDArray<D>>(
      a: T, perm: number[]): T {
    const newShape: number[] = new Array(a.rank);
    for (let i = 0; i < newShape.length; i++) {
      newShape[i] = a.shape[perm[i]];
    }
    const resultValues = new Float32Array(a.size);
    const values = a.getValues();
    const result = NDArray.make(newShape, {values: resultValues}) as T;
    for (let i = 0; i < a.size; ++i) {
      const loc = a.indexToLoc(i);

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
    return dx;
  }

  minPool(x: Array4D, convInfo: Conv2DInfo): Array4D {
    return this.pool(x, convInfo, 'min');
  }

  avgPool(x: Array4D, convInfo: Conv2DInfo): Array4D {
    return this.pool(x, convInfo, 'avg');
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
    const xValues = x.getValues();
    const meanValues = mean.getValues();
    const varianceValues = variance.getValues();
    const scaleValues = scale ? scale.getValues() : new Float32Array([1]);
    const offsetValues = offset ? offset.getValues() : new Float32Array([0]);
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
    const xValues = x.getValues();
    const meanValues = mean.getValues();
    const varianceValues = variance.getValues();
    const scaleValues = scale ? scale.getValues() : new Float32Array([1]);
    const offsetValues = offset ? offset.getValues() : new Float32Array([0]);
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

  multinomial(probabilities: Array2D, numSamples: number, seed: number):
      Array2D<'int32'> {
    const batchSize = probabilities.shape[0];
    const numEvents = probabilities.shape[1];
    const res = Array2D.zeros([batchSize, numSamples], 'int32');
    const resVals = res.getValues();
    const probVals = probabilities.getValues();

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

  private broadcastedBinaryOp<D extends keyof DataTypes>(
      a: NDArray, b: NDArray, dtype: D,
      op: (a: number, b: number) => number): NDArray<D> {
    const newShape =
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    const result = NDArray.zeros(newShape, dtype);
    const newValues = result.getValues();
    const aValues = a.getValues();
    const bValues = b.getValues();

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
}

// TODO(nsthorat): Deprecate this once we export non-abstract NDArrayMath.
export class NDArrayMathCPU extends NDArrayMath {
  constructor(safeMode = false) {
    super(new MathBackendCPU(), safeMode);
  }
}
