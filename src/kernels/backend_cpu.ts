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

import {ENV} from '../environment';
import * as axis_util from '../ops/axis_util';
import * as broadcast_util from '../ops/broadcast_util';
import * as concat_util from '../ops/concat_util';
import {Conv2DInfo} from '../ops/conv_util';
import * as ops from '../ops/ops';
import {buffer, tensor3d, tensor4d} from '../ops/ops';
import * as selu_util from '../ops/selu_util';
// tslint:disable-next-line:max-line-length
import {DataId, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D} from '../tensor';
import * as types from '../types';
import {DataType, DataTypeMap, Rank, TypedArray} from '../types';
import * as util from '../util';

import {BackendTimingInfo, KernelBackend} from './backend';
import * as backend_util from './backend_util';

export class MathBackendCPU implements KernelBackend {
  private data = new WeakMap<DataId, DataTypeMap[DataType]>();
  private canvas: HTMLCanvasElement;

  constructor() {
    if (typeof document !== 'undefined') {
      this.canvas = document.createElement('canvas');
    }
  }

  register(dataId: DataId, shape: number[], dtype: DataType): void {
    if (this.data.has(dataId)) {
      throw new Error(`Data buffer is already registered`);
    }
    this.data.set(dataId, null);
  }
  write(dataId: DataId, values: TypedArray): void {
    if (values == null) {
      throw new Error('MathBackendCPU.write(): values can not be null');
    }
    this.throwIfNoData(dataId);
    this.data.set(dataId, values);
  }
  fromPixels(
      pixels: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      numChannels: number): Tensor3D {
    if (pixels == null) {
      throw new Error('MathBackendCPU.writePixels(): pixels can not be null');
    }
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
    const outShape: [number, number, number] =
        [pixels.height, pixels.width, numChannels];
    return tensor3d(values, outShape, 'int32');
  }
  async read(dataId: DataId): Promise<TypedArray> {
    return this.readSync(dataId);
  }
  readSync(dataId: DataId): TypedArray {
    this.throwIfNoData(dataId);
    return this.data.get(dataId);
  }

  disposeData(dataId: DataId): void {
    if (this.data.has(dataId)) {
      this.data.delete(dataId);
    }
  }

  async time(f: () => void): Promise<BackendTimingInfo> {
    const start = performance.now();
    f();
    const kernelMs = performance.now() - start;
    return {kernelMs};
  }
  memory() {
    return {
      // Unreliable due to automatic gc. The numbers above are cumulative.
      unreliable: true
    };
  }

  private throwIfNoData(dataId: DataId) {
    if (!this.data.has(dataId)) {
      throw new Error(
          `CPU backend: No data found for this tensor. ` +
          `Did you change your backend in the middle of the program? ` +
          `New backends can't use Tensors created with previous backends`);
    }
  }

  slice<T extends Tensor>(x: T, begin: number[], size: number[]): T {
    const buffer = ops.buffer(size, x.dtype);

    for (let i = 0; i < buffer.size; ++i) {
      const loc = buffer.indexToLoc(i);
      const xLoc = loc.map((idx, j) => idx + begin[j]);
      buffer.set(x.get(...xLoc), ...loc);
    }
    return buffer.toTensor() as T;
  }

  reverse<T extends Tensor>(x: T, axis: number[]): T {
    const buffer = ops.buffer(x.shape, x.dtype);
    const xBuffer = x.buffer();

    for (let i = 0; i < buffer.size; i++) {
      const outLoc = buffer.indexToLoc(i);
      const inLoc = outLoc.slice();
      axis.forEach(ax => inLoc[ax] = x.shape[ax] - 1 - inLoc[ax]);
      buffer.set(xBuffer.get(...inLoc), ...outLoc);
    }

    return buffer.toTensor() as T;
  }

  // Concats 2d tensors along axis=1. See comments in MathBackend.concat().
  concat(a: Tensor2D, b: Tensor2D): Tensor2D {
    const outShape = concat_util.computeOutShape(
                         a.shape, b.shape, 1 /* axis */) as [number, number];
    const buffer = ops.buffer<Rank.R2>(outShape, a.dtype);

    if (a.shape[0] === 1 && b.shape[0] === 1) {
      // Use built-in TypedArray.set() method for speed.
      const aVals = a.dataSync();
      const bVals = b.dataSync();
      const vals = buffer.values;
      vals.set(aVals, 0);
      vals.set(bVals, a.size);
      return buffer.toTensor();
    }

    for (let i = 0; i < outShape[0]; ++i) {
      for (let j = 0; j < a.shape[1]; ++j) {
        buffer.set(a.get(i, j), i, j);
      }
      for (let j = 0; j < b.shape[1]; ++j) {
        buffer.set(b.get(i, j), i, j + a.shape[1]);
      }
    }
    return buffer.toTensor();
  }

  neg<T extends Tensor>(x: T): T {
    return this.multiply(ops.scalar(-1), x) as T;
  }

  add(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp(
               a, b, types.upcastType(a.dtype, b.dtype),
               (aValue, bValue) => aValue + bValue) as Tensor;
  }

  subtract(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp(
               a, b, types.upcastType(a.dtype, b.dtype),
               (aValue, bValue) => aValue - bValue) as Tensor;
  }

  pow<T extends Tensor>(a: T, b: Tensor): T {
    return this.broadcastedBinaryOp(
               a, b, a.dtype, (aValue, bValue) => Math.pow(aValue, bValue)) as
        T;
  }

  matMul(a: Tensor2D, b: Tensor2D, transposeA: boolean, transposeB: boolean):
      Tensor2D {
    const sharedDim = transposeA ? a.shape[0] : a.shape[1];
    const leftDim = transposeA ? a.shape[1] : a.shape[0];
    const rightDim = transposeB ? b.shape[0] : b.shape[1];

    const normalGetter = (matrix: Tensor2D, i: number, j: number) =>
        matrix.get(i, j);
    const transposedGetter = (matrix: Tensor2D, i: number, j: number) =>
        matrix.get(j, i);

    const aGetter = transposeA ? transposedGetter : normalGetter;
    const bGetter = transposeB ? transposedGetter : normalGetter;

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
    return ops.tensor2d(values, [leftDim, rightDim]);
  }

  multiply(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp(
               a, b, types.upcastType(a.dtype, b.dtype),
               (aValue, bValue) => aValue * bValue) as Tensor;
  }

  divide(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp(
               a, b, 'float32', (aValue, bValue) => aValue / bValue) as Tensor;
  }

  sum(x: Tensor, axes: number[]): Tensor {
    axis_util.assertAxesAreInnerMostDims('sum', axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const resultDtype = types.upcastType(x.dtype, 'int32');
    const result = ops.zeros(outShape, resultDtype);
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
    return result;
  }

  argMin(x: Tensor, axes: number[]): Tensor {
    axis_util.assertAxesAreInnerMostDims('argMin', axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const result = ops.zeros(outShape, 'int32');
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

  argMax(x: Tensor, axes: number[]): Tensor {
    axis_util.assertAxesAreInnerMostDims('argMax', axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const result = ops.zeros(outShape, 'int32');
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

  equal(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      if (util.isValNaN(aVal, a.dtype) || util.isValNaN(bVal, b.dtype)) {
        return util.getNaN('bool');
      } else {
        return (aVal === bVal) ? 1 : 0;
      }
    });
  }

  notEqual(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      if (util.isValNaN(aVal, a.dtype) || util.isValNaN(bVal, b.dtype)) {
        return util.getNaN('bool');
      } else {
        return (aVal !== bVal) ? 1 : 0;
      }
    });
  }

  less(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      if (util.isValNaN(aVal, a.dtype) || util.isValNaN(bVal, b.dtype)) {
        return util.getNaN('bool');
      } else {
        return (aVal < bVal) ? 1 : 0;
      }
    });
  }

  lessEqual(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      if (util.isValNaN(aVal, a.dtype) || util.isValNaN(bVal, b.dtype)) {
        return util.getNaN('bool');
      } else {
        return (aVal <= bVal) ? 1 : 0;
      }
    });
  }

  greater(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      if (util.isValNaN(aVal, a.dtype) || util.isValNaN(bVal, b.dtype)) {
        return util.getNaN('bool');
      } else {
        return (aVal > bVal) ? 1 : 0;
      }
    });
  }

  greaterEqual(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      if (util.isValNaN(aVal, a.dtype) || util.isValNaN(bVal, b.dtype)) {
        return util.getNaN('bool');
      } else {
        return (aVal >= bVal) ? 1 : 0;
      }
    });
  }

  logicalNot<T extends Tensor>(x: T): T {
    const values = x.dataSync();
    const newValues = new Int32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      if (util.isValNaN(values[i], x.dtype)) {
        newValues[i] = util.getNaN('bool');
      } else {
        newValues[i] = values[i] ? 0 : 1;
      }
    }
    return Tensor.make(x.shape, {values: newValues}, 'bool') as T;
  }

  logicalAnd(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      if (util.isValNaN(aVal, a.dtype) || util.isValNaN(bVal, b.dtype)) {
        return util.getNaN('bool');
      } else {
        return aVal && bVal;
      }
    });
  }

  logicalOr(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      if (util.isValNaN(aVal, a.dtype) || util.isValNaN(bVal, b.dtype)) {
        return util.getNaN('bool');
      } else {
        return aVal || bVal;
      }
    });
  }

  logicalXor(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      if (util.isValNaN(aVal, a.dtype) || util.isValNaN(bVal, b.dtype)) {
        return util.getNaN('bool');
      } else {
        return aVal ^ bVal;
      }
    });
  }

  where(condition: Tensor, a: Tensor, b: Tensor, dtype: DataType): Tensor {
    const values = condition.dataSync();
    const aValues = a.dataSync();
    const bValues = b.dataSync();
    const result = ops.zeros(a.shape, dtype);
    const newValues = result.dataSync();
    let index = 0;
    const offset = condition.rank === 0 || condition.rank > 1 || a.rank === 1 ?
        1 :
        a.shape[1];

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

  topKValues<T extends Tensor>(x: T, k: number): Tensor1D {
    return this.topK(x, k).values as Tensor1D;
  }

  topKIndices(x: Tensor, k: number): Tensor1D {
    return this.topK(x, k).indices;
  }

  private topK<T extends Tensor>(x: T, k: number):
      {values: Tensor1D, indices: Tensor1D} {
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
      values: ops.tensor1d(topkValues, x.dtype),
      indices: ops.tensor1d(topkIndices, 'int32')
    };
  }

  min(x: Tensor, axes: number[]): Tensor {
    axis_util.assertAxesAreInnerMostDims('min', axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const result = ops.zeros(outShape, x.dtype);
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

  minimum(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp(
        a, b, a.dtype, (aVal, bVal) => Math.min(aVal, bVal));
  }

  max(x: Tensor, axes: number[]): Tensor {
    axis_util.assertAxesAreInnerMostDims('max', axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const result = ops.zeros(outShape, x.dtype);
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

  maximum(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp(
        a, b, a.dtype, (aVal, bVal) => Math.max(aVal, bVal));
  }

  squaredDifference(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp(a, b, a.dtype, (aVal, bVal) => {
      const diff = aVal - bVal;
      return diff * diff;
    });
  }

  ceil<T extends Tensor>(x: T): T {
    const values = x.dataSync();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      newValues[i] = Math.ceil(values[i]);
    }
    return Tensor.make(x.shape, {values: newValues}) as T;
  }

  floor<T extends Tensor>(x: T): T {
    const values = x.dataSync();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      newValues[i] = Math.floor(values[i]);
    }
    return Tensor.make(x.shape, {values: newValues}) as T;
  }

  exp<T extends Tensor>(x: T): T {
    const values = x.dataSync();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      newValues[i] = Math.exp(values[i]);
    }
    return Tensor.make(x.shape, {values: newValues}) as T;
  }

  expm1<T extends Tensor>(x: T): T {
    const values = x.dataSync();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      newValues[i] = Math.expm1(values[i]);
    }
    return Tensor.make(x.shape, {values: newValues}) as T;
  }

  log<T extends Tensor>(x: T): T {
    const values = x.dataSync();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      const value = values[i];
      newValues[i] = Math.log(value);
    }
    return Tensor.make(x.shape, {values: newValues}) as T;
  }

  log1p<T extends Tensor>(x: T): T {
    const values = x.dataSync();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      const value = values[i];
      newValues[i] = Math.log1p(value);
    }
    return Tensor.make(x.shape, {values: newValues}) as T;
  }

  sqrt<T extends Tensor>(x: T): T {
    const values = x.dataSync();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      const value = values[i];
      newValues[i] = Math.sqrt(value);
    }
    return Tensor.make(x.shape, {values: newValues}) as T;
  }

  square<T extends Tensor>(x: T): T {
    const values = x.dataSync();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      const value = values[i];
      newValues[i] = value * value;
    }
    return Tensor.make(x.shape, {values: newValues}) as T;
  }

  relu<T extends Tensor>(x: T): T {
    const res = ops.zeros(x.shape, x.dtype);
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

  elu<T extends Tensor>(x: T): T {
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
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  eluDer<T extends Tensor>(x: T): T {
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
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  selu<T extends Tensor>(x: T): T {
    // Stable and Attracting Fixed Point (0, 1) for Normalized Weights.
    // see: https://arxiv.org/abs/1706.02515
    const scaleAlpha = selu_util.SELU_SCALEALPHA;
    const scale = selu_util.SELU_SCALE;

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
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  leakyRelu<T extends Tensor>(x: T, alpha: number) {
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
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  prelu<T extends Tensor>(x: T, alpha: T) {
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
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  preluDer<T extends Tensor>(x: T, alpha: T) {
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
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  clip<T extends Tensor>(x: T, min: number, max: number): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.min(max, Math.max(min, values[i]));
    }
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  abs<T extends Tensor>(x: T): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.abs(values[i]);
    }
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  int<T extends Tensor>(x: T): T {
    const resultValues = new Int32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = values[i];
    }
    return Tensor.make(x.shape, {values: resultValues}, 'int32');
  }

  sigmoid<T extends Tensor>(x: T): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = 1 / (1 + Math.exp(-values[i]));
    }
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  sin<T extends Tensor>(x: T): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.sin(values[i]);
    }
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  cos<T extends Tensor>(x: T): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.cos(values[i]);
    }
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  tan<T extends Tensor>(x: T): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.tan(values[i]);
    }
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  asin<T extends Tensor>(x: T): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.asin(values[i]);
    }
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  acos<T extends Tensor>(x: T): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.acos(values[i]);
    }
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  atan<T extends Tensor>(x: T): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.atan(values[i]);
    }
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  atan2<T extends Tensor>(a: T, b: T): T {
    return this.broadcastedBinaryOp(
               a, b, a.dtype, (aValue, bValue) => Math.atan2(aValue, bValue)) as
        T;
  }

  sinh<T extends Tensor>(x: T): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.sinh(values[i]);
    }
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  cosh<T extends Tensor>(x: T): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.cosh(values[i]);
    }
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  tanh<T extends Tensor>(x: T): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = util.tanh(values[i]);
    }
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  step<T extends Tensor>(x: T, alpha = 0): T {
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
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  conv2d(x: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const padLeft = convInfo.padInfo.left;
    const padTop = convInfo.padInfo.top;
    const y = ops.buffer<Rank.R4>(convInfo.outShape, x.dtype);

    for (let b = 0; b < convInfo.batchSize; ++b) {
      for (let d2 = 0; d2 < convInfo.outChannels; ++d2) {
        for (let yR = 0; yR < convInfo.outHeight; ++yR) {
          const xRCorner = yR * convInfo.strideHeight - padLeft;
          for (let yC = 0; yC < convInfo.outWidth; ++yC) {
            const xCCorner = yC * convInfo.strideWidth - padTop;

            let dotProd = 0;
            for (let wR = 0; wR < filterHeight; wR++) {
              const xR = xRCorner + wR * dilationHeight;

              if (xR < 0 || xR >= convInfo.inHeight) {
                continue;
              }

              for (let wC = 0; wC < filterWidth; wC++) {
                const xC = xCCorner + wC * dilationWidth;

                if (xC < 0 || xC >= convInfo.inWidth) {
                  continue;
                }

                for (let d1 = 0; d1 < convInfo.inChannels; ++d1) {
                  const pixel = x.get(b, xR, xC, d1);
                  const weight = filter.get(wR, wC, d1, d2);
                  dotProd += pixel * weight;
                }
              }
            }
            y.set(dotProd, b, yR, yC, d2);
          }
        }
      }
    }
    return y.toTensor();
  }

  conv2dDerInput(dy: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const topPad = filterHeight - 1 - convInfo.padInfo.top;
    const leftPad = filterWidth - 1 - convInfo.padInfo.left;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dx = ops.buffer<Rank.R4>(convInfo.inShape, 'float32');

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
    return dx.toTensor();
  }

  conv2dDerFilter(x: Tensor4D, dy: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const dW = ops.buffer<Rank.R4>(convInfo.filterShape, 'float32');

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
    return dW.toTensor();
  }

  depthwiseConv2D(x: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const padLeft = convInfo.padInfo.left;
    const padTop = convInfo.padInfo.top;
    const chMul = convInfo.outChannels / convInfo.inChannels;
    const y = ops.buffer<Rank.R4>(convInfo.outShape, x.dtype);

    for (let b = 0; b < convInfo.batchSize; ++b) {
      for (let d1 = 0; d1 < convInfo.inChannels; ++d1) {
        for (let yR = 0; yR < convInfo.outHeight; ++yR) {
          const xRCorner = yR * convInfo.strideHeight - padLeft;
          for (let yC = 0; yC < convInfo.outWidth; ++yC) {
            const xCCorner = yC * convInfo.strideWidth - padTop;
            for (let q = 0; q < chMul; ++q) {
              let dotProd = 0;
              for (let wR = 0; wR < filterHeight; ++wR) {
                const xR = xRCorner + wR * dilationHeight;

                if (xR < 0 || xR >= convInfo.inHeight) {
                  continue;
                }

                for (let wC = 0; wC < filterWidth; ++wC) {
                  const xC = xCCorner + wC * dilationWidth;

                  if (xC < 0 || xC >= convInfo.inWidth) {
                    continue;
                  }

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

    return y.toTensor();
  }

  tile<T extends Tensor>(x: T, reps: number[]): T {
    const newShape: number[] = new Array(x.rank);
    for (let i = 0; i < newShape.length; i++) {
      newShape[i] = x.shape[i] * reps[i];
    }
    const result = ops.buffer(newShape, x.dtype);
    const xBuf = x.buffer();
    for (let i = 0; i < result.values.length; ++i) {
      const newLoc = result.indexToLoc(i);

      const originalLoc: number[] = new Array(x.rank);
      for (let i = 0; i < originalLoc.length; i++) {
        originalLoc[i] = newLoc[i] % x.shape[i];
      }

      const originalIndex = xBuf.locToIndex(originalLoc);

      result.values[i] = xBuf.values[originalIndex];
    }
    return result.toTensor() as T;
  }

  pad<T extends Tensor>(
      x: T, paddings: Array<[number, number]>, constantValue: number): T {
    const outShape = paddings.map(
        (p, i) => p[0] /* beforePad */ + x.shape[i] + p[1] /* afterPad */);
    const start = paddings.map(p => p[0]);
    const xBuffer = x.buffer();
    const buffer = ops.buffer(outShape, x.dtype);
    if (constantValue !== 0) {
      buffer.values.fill(constantValue);
    }

    for (let i = 0; i < x.size; i++) {
      const coords = xBuffer.indexToLoc(i);
      const outCoords = coords.map((c, i) => c + start[i]);
      buffer.set(x.get(...coords), ...outCoords);
    }
    return buffer.toTensor() as T;
  }

  transpose<T extends Tensor>(x: T, perm: number[]): T {
    const newShape: number[] = new Array(x.rank);
    for (let i = 0; i < newShape.length; i++) {
      newShape[i] = x.shape[perm[i]];
    }
    const values = x.dataSync();
    const result = buffer(newShape, x.dtype);

    const xBuf = x.buffer();
    for (let i = 0; i < x.size; ++i) {
      const loc = xBuf.indexToLoc(i);

      // Permute location.
      const newLoc: number[] = new Array(loc.length);
      for (let i = 0; i < newLoc.length; i++) {
        newLoc[i] = loc[perm[i]];
      }

      const newIndex = result.locToIndex(newLoc);
      result.values[newIndex] = values[i];
    }
    return result.toTensor() as T;
  }

  gather<T extends Tensor>(x: T, indices: Tensor1D, axis: number): T {
    const newShape: number[] = x.shape.slice();
    const indicesValues = indices.dataSync();
    newShape[axis] = indicesValues.length;
    const result = buffer(newShape, x.dtype);
    const xBuf = x.buffer();

    for (let i = 0; i < result.size; ++i) {
      const newLoc = result.indexToLoc(i);

      const originalLoc: number[] = newLoc.slice();
      originalLoc[axis] = indicesValues[newLoc[axis]];

      const originalIndex = xBuf.locToIndex(originalLoc);
      result.values[i] = xBuf.values[originalIndex];
    }
    return result.toTensor() as T;
  }

  private pool(x: Tensor4D, convInfo: Conv2DInfo, poolType: 'max'|'min'|'avg'):
      Tensor4D {
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const y = ops.buffer<Rank.R4>(convInfo.outShape, 'float32');
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
    return y.toTensor();
  }

  maxPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    return this.pool(x, convInfo, 'max');
  }

  private maxPoolPositions(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    const maxPositions = ops.buffer<Rank.R4>(convInfo.outShape, 'int32');
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
    return maxPositions.toTensor();
  }

  maxPoolBackprop(dy: Tensor4D, x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    const maxPositions = this.maxPoolPositions(x, convInfo);
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const padLeft = filterWidth - 1 - convInfo.padInfo.left;
    const padTop = filterHeight - 1 - convInfo.padInfo.top;
    const dx = ops.buffer<Rank.R4>(x.shape, 'float32');

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
    return dx.toTensor();
  }

  avgPoolBackprop(dy: Tensor4D, x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const padLeft = filterWidth - 1 - convInfo.padInfo.left;
    const padTop = filterHeight - 1 - convInfo.padInfo.top;
    const dx = ops.buffer<Rank.R4>(x.shape, 'float32');

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
    return dx.toTensor();
  }

  cast<T extends Tensor<types.Rank>>(x: T, dtype: DataType): T {
    return backend_util.castTensor(x, dtype, this);
  }

  reshape<T extends Tensor<types.Rank>, R extends types.Rank>(
      x: T, shape: types.ShapeMap[R]): Tensor<R> {
    return backend_util.reshapeTensor(x, shape);
  }

  minPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    return this.pool(x, convInfo, 'min');
  }

  avgPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    return this.pool(x, convInfo, 'avg').toFloat();
  }

  resizeBilinear(
      x: Tensor4D, newHeight: number, newWidth: number,
      alignCorners: boolean): Tensor4D {
    const [batch, oldHeight, oldWidth, numChannels] = x.shape;
    const output =
        ops.buffer<Rank.R4>([batch, newHeight, newWidth, numChannels], x.dtype);

    const effectiveInputSize: [number, number] =
        alignCorners ? [oldHeight - 1, oldWidth - 1] : [oldHeight, oldWidth];
    const effectiveOutputSize: [number, number] =
        alignCorners ? [newHeight - 1, newWidth - 1] : [newHeight, newWidth];
    for (let b = 0; b < batch; b++) {
      for (let r = 0; r < newHeight; r++) {
        for (let c = 0; c < newWidth; c++) {
          for (let d = 0; d < numChannels; d++) {
            // Begin shader.

            // Compute the fractional index of the source.
            const sourceFracRow =
                (effectiveInputSize[0]) * r / (effectiveOutputSize[0]);
            const sourceFracCol =
                (effectiveInputSize[1]) * c / (effectiveOutputSize[1]);

            const sourceRowFloor = Math.floor(sourceFracRow);
            const sourceRowCeil =
                Math.min(oldHeight - 1, Math.ceil(sourceFracRow));
            const sourceColFloor = Math.floor(sourceFracCol);
            const sourceColCeil =
                Math.min(oldWidth - 1, Math.ceil(sourceFracCol));

            const topLeft = x.get(b, sourceRowFloor, sourceColFloor, d);
            const bottomLeft = x.get(b, sourceRowCeil, sourceColFloor, d);
            const topRight = x.get(b, sourceRowFloor, sourceColCeil, d);
            const bottomRight = x.get(b, sourceRowCeil, sourceColCeil, d);

            const rowFrac = sourceFracRow - sourceRowFloor;
            const colFrac = sourceFracCol - sourceColFloor;

            const top = topLeft + (topRight - topLeft) * colFrac;
            const bottom = bottomLeft + (bottomRight - bottomLeft) * colFrac;
            const newValue = top + (bottom - top) * rowFrac;

            output.set(newValue, b, r, c, d);
          }
        }
      }
    }

    return output.toTensor();
  }

  batchNormalization4D(
      x: Tensor4D, mean: Tensor4D|Tensor1D, variance: Tensor4D|Tensor1D,
      varianceEpsilon: number, scale?: Tensor4D|Tensor1D,
      offset?: Tensor4D|Tensor1D): Tensor4D {
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
    return tensor4d(outValues, x.shape);
  }

  localResponseNormalization4D(
      x: Tensor4D, radius: number, bias: number, alpha: number, beta: number,
      normRegion: 'acrossChannels'|'withinChannel'): Tensor4D {
    const output = ops.buffer<Rank.R4>(x.shape, 'float32');
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

    return output.toTensor();
  }

  multinomial(probabilities: Tensor2D, numSamples: number, seed: number):
      Tensor2D {
    const batchSize = probabilities.shape[0];
    const numEvents = probabilities.shape[1];
    const res = ops.zeros<Rank.R2>([batchSize, numSamples], 'int32');
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

  oneHot(indices: Tensor1D, depth: number, onValue: number, offValue: number):
      Tensor2D {
    const res = new Float32Array(indices.size * depth);
    res.fill(offValue);

    for (let event = 0; event < indices.size; ++event) {
      res[event * depth + indices.get(event)] = onValue;
    }
    return ops.tensor2d(res, [indices.size, depth]);
  }

  private broadcastedBinaryOp(
      a: Tensor, b: Tensor, dtype: DataType,
      op: (a: number, b: number) => number): Tensor {
    const newShape =
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    const result = ops.buffer(newShape, dtype);
    const aValues = a.dataSync();
    const bValues = b.dataSync();

    const aBroadcastDims = broadcast_util.getBroadcastDims(a.shape, newShape);
    const bBroadcastDims = broadcast_util.getBroadcastDims(b.shape, newShape);

    const aBuf = a.buffer();
    const bBuf = b.buffer();
    for (let i = 0; i < result.values.length; ++i) {
      const loc = result.indexToLoc(i);

      const aLoc = loc.slice(-a.rank);
      aBroadcastDims.forEach(d => aLoc[d] = 0);
      const aIndex = aBuf.locToIndex(aLoc);

      const bLoc = loc.slice(-b.rank);
      bBroadcastDims.forEach(d => bLoc[d] = 0);
      const bIndex = bBuf.locToIndex(bLoc);

      result.values[i] = op(aValues[aIndex], bValues[bIndex]);
    }
    return result.toTensor();
  }
  dispose() {}
}

ENV.registerBackend('cpu', () => new MathBackendCPU());
