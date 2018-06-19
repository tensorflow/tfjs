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
import * as erf_util from '../ops/erf_util';
import * as ops from '../ops/ops';
import {buffer, tensor3d, tensor4d} from '../ops/ops';
import * as selu_util from '../ops/selu_util';
import {getStridedSlicedInfo} from '../ops/slice_util';
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
  private firstUse = true;

  constructor() {
    if (ENV.get('IS_BROWSER')) {
      this.canvas = document.createElement('canvas');
    }
  }

  register(dataId: DataId, shape: number[], dtype: DataType): void {
    if (this.firstUse) {
      this.firstUse = false;
      if (ENV.get('IS_NODE')) {
        console.warn(
            '\n============================\n' +
            'Hi there 👋. Looks like you are running TensorFlow.js in ' +
            'Node.js. To speed things up dramatically, install our node ' +
            'backend, which binds to TensorFlow C++, by running ' +
            'npm i @tensorflow/tfjs-node, ' +
            'or npm i @tensorflow/tfjs-node-gpu if you have CUDA. ' +
            'Then call require(\'tensorflow/tfjs-node\'); (-gpu ' +
            'suffix for CUDA) at the start of your program. ' +
            'Visit https://github.com/tensorflow/tfjs-node for more details.' +
            '\n============================\n');
      }
    }
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

  stridedSlice<T extends Tensor>(
      x: T, begin: number[], end: number[], strides: number[],
      beginMask: number, endMask: number): T {
    const [beginIndex, size] =
        getStridedSlicedInfo(x.shape, begin, end, strides, beginMask, endMask);

    if (size.some(axis => axis === 0)) {
      return ops.tensor([], size) as T;
    }

    const buffer = ops.buffer(size, x.dtype);

    for (let i = 0; i < buffer.size; i++) {
      const loc = buffer.indexToLoc(i);

      const newLoc: number[] = new Array(loc.length);
      for (let j = 0; j < newLoc.length; j++) {
        newLoc[j] = loc[j] * strides[j] + beginIndex[j];
      }
      buffer.set(x.get(...newLoc), ...loc);
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

    const aValues = a.dataSync();
    const bValues = b.dataSync();

    const [aOuterStep, aInnerStep] =
        transposeA ? [1, a.strides[0]] : [a.strides[0], 1];
    const [bOuterStep, bInnerStep] =
        transposeB ? [b.strides[0], 1] : [1, b.strides[0]];

    const aOuterEnd = leftDim * aOuterStep;
    const bOuterEnd = rightDim * bOuterStep;

    const result = new Float32Array(leftDim * rightDim);
    let resultIndex = 0;

    for (let aOuter = 0; aOuter < aOuterEnd; aOuter += aOuterStep) {
      for (let bOuter = 0; bOuter < bOuterEnd; bOuter += bOuterStep) {
        let aInner = aOuter;
        let bInner = bOuter;
        let sum = 0;
        for (let k = 0; k < sharedDim; ++k) {
          sum += aValues[aInner] * bValues[bInner];
          aInner += aInnerStep;
          bInner += bInnerStep;
        }
        result[resultIndex++] = sum;
      }
    }
    return ops.tensor2d(result, [leftDim, rightDim]);
  }

  multiply(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp(
               a, b, types.upcastType(a.dtype, b.dtype),
               (aValue, bValue) => aValue * bValue) as Tensor;
  }

  realDivide(a: Tensor, b: Tensor): Tensor {
    const op = (a: number, b: number) => a / b;
    const outputDtype = 'float32';
    return this.broadcastedBinaryOp(a, b, outputDtype, op) as Tensor;
  }

  floorDiv(a: Tensor, b: Tensor): Tensor {
    const op = (a: number, b: number) => Math.floor(a / b);
    const outputDtype = 'int32';
    return this.broadcastedBinaryOp(a, b, outputDtype, op) as Tensor;
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

  unsortedSegmentSum<T extends Tensor>(
      x: T, segmentIds: Tensor1D, numSegments: number): Tensor {
    const res = [];

    // Reshape the segment id's so that they can be broadcast with
    // x. The new shape should be [segmentIds.shape, 1, ..., 1]
    const numIters = x.rank - segmentIds.rank;
    for (let i = 0; i < numIters; ++i) {
      segmentIds = segmentIds.expandDims(i + 1);
    }

    for (let i = 0; i < numSegments; ++i) {
      const segmentId = ops.scalar(i, 'int32');
      const mask = ops.equal(segmentId, segmentIds).asType('float32');
      const sum = mask.mul(x).sum(0);
      res.push(sum);
    }

    return ops.stack(res);
  }

  argMin(x: Tensor, axis: number): Tensor {
    const axes = [axis];
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
        if (value < min) {
          min = value;
          minIndex = j;
        }
      }
      vals[i] = minIndex;
    }
    return result;
  }

  argMax(x: Tensor, axis: number): Tensor {
    const axes = [axis];
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
        if (value > max) {
          max = value;
          maxIndex = j;
        }
      }
      vals[i] = maxIndex;
    }
    return result;
  }

  cumsum(x: Tensor, axis: number, exclusive: boolean, reverse: boolean):
      Tensor {
    if (axis !== x.rank - 1) {
      throw new Error(
          `backend.cumsum in CPU expects an inner-most axis=${x.rank - 1} ` +
          `but got axis=${axis}`);
    }
    const resultDtype = types.upcastType(x.dtype, 'int32');
    const result = ops.zeros(x.shape, resultDtype);
    const vals = result.dataSync();

    const aVals = x.dataSync();
    const finalDim = x.shape[x.rank - 1];
    const indexAdjuster = reverse ?
        (i: number, j: number) => i + finalDim - j - 1 :
        (i: number, j: number) => i + j;
    for (let i = 0; i < aVals.length; i += finalDim) {
      for (let j = 0; j < finalDim; j++) {
        const idx = indexAdjuster(i, j);
        if (j === 0) {
          vals[idx] = exclusive ? 0 : aVals[idx];
        } else {
          const prevIdx = indexAdjuster(i, j - 1);
          vals[idx] = exclusive ? aVals[prevIdx] + vals[prevIdx] :
                                  aVals[idx] + vals[prevIdx];
        }
      }
    }
    return result;
  }

  equal(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      return (aVal === bVal) ? 1 : 0;
    });
  }

  notEqual(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      return (aVal !== bVal) ? 1 : 0;
    });
  }

  less(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      return (aVal < bVal) ? 1 : 0;
    });
  }

  lessEqual(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      return (aVal <= bVal) ? 1 : 0;
    });
  }

  greater(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      return (aVal > bVal) ? 1 : 0;
    });
  }

  greaterEqual(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      return (aVal >= bVal) ? 1 : 0;
    });
  }

  logicalNot<T extends Tensor>(x: T): T {
    const values = x.dataSync();
    const newValues = new Int32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      newValues[i] = values[i] ? 0 : 1;
    }
    return Tensor.make(x.shape, {values: newValues}, 'bool') as T;
  }

  logicalAnd(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      return aVal && bVal;
    });
  }

  logicalOr(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      return aVal || bVal;
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
      let min = aVals[offset];
      for (let j = 0; j < reduceSize; ++j) {
        const value = aVals[offset + j];
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

  mod(a: Tensor, b: Tensor): Tensor {
    return this.broadcastedBinaryOp(a, b, a.dtype, (aVal, bVal) => {
      const rem = aVal % bVal;
      if ((aVal < 0 && bVal < 0) || (aVal >= 0 && bVal >= 0)) {
        return rem;
      } else {
        return (rem + bVal) % bVal;
      }
    });
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

  all(x: Tensor, axes: number[]): Tensor {
    axis_util.assertAxesAreInnerMostDims('all', axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const result = ops.zeros(outShape, x.dtype);
    const reduceSize = util.sizeFromShape(reduceShape);
    const vals = result.dataSync();

    const aVals = x.dataSync();
    for (let i = 0; i < vals.length; ++i) {
      const offset = i * reduceSize;
      let all = aVals[offset];
      for (let j = 0; j < reduceSize; ++j) {
        const value = aVals[offset + j];
        all = all && value;
      }
      vals[i] = all;
    }
    return result;
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

  sign<T extends Tensor>(x: T): T {
    const values = x.dataSync();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      if (values[i] < 0) {
        newValues[i] = -1;
      } else if (values[i] > 0) {
        newValues[i] = 1;
      } else {
        newValues[i] = 0;
      }
    }
    return Tensor.make(x.shape, {values: newValues}) as T;
  }

  round<T extends Tensor>(x: T): T {
    const values = x.dataSync();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      // The algorithm is based on banker's rounding.
      const base = Math.floor(values[i]);
      if (values[i] - base < 0.5) {
        newValues[i] = Math.floor(values[i]);
      } else if (values[i] - base > 0.5) {
        newValues[i] = Math.ceil(values[i]);
      } else {
        if (base % 2.0 === 0.0) {
          newValues[i] = base;
        } else {
          newValues[i] = base + 1.0;
        }
      }
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

  rsqrt<T extends Tensor>(x: T): T {
    const values = x.dataSync();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      const value = values[i];
      newValues[i] = 1 / Math.sqrt(value);
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

  reciprocal<T extends Tensor>(x: T): T {
    const values = x.dataSync();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      newValues[i] = 1 / values[i];
    }
    return Tensor.make(x.shape, {values: newValues}) as T;
  }

  relu<T extends Tensor>(x: T): T {
    const res = ops.zeros(x.shape, x.dtype);
    const resVals = res.dataSync();
    const inVals = x.dataSync();
    for (let i = 0; i < inVals.length; ++i) {
      resVals[i] = Math.max(0, inVals[i]);
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

  eluDer<T extends Tensor>(dy: T, y: T): T {
    const resultValues = new Float32Array(y.size);
    const values = y.dataSync();
    const dyValues = dy.dataSync();
    for (let i = 0; i < values.length; ++i) {
      const v = values[i];
      if (v >= 1) {
        resultValues[i] = dyValues[i];
      } else {
        resultValues[i] = dyValues[i] * (v + 1);
      }
    }
    return Tensor.make(y.shape, {values: resultValues}) as T;
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

  softplus<T extends Tensor>(x: T): T {
    // mirrors the implementation of tf.nn.softplus: https://goo.gl/vkcvwX

    // epsilon is the difference between 1.0 and the next representable float.
    // For a single precision 32 bit float this should be 2^-23, see:
    // https://math.byu.edu/~schow/work/IEEEFloatingPoint.htm
    const epsilon = 1.1920928955078125e-7;
    const threshold = Math.log(epsilon) + 2.0;

    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();

    for (let i = 0; i < values.length; ++i) {
      // Value above which exp(x) may overflow, but softplus(x) == x
      // is within machine epsilon.
      const tooLarge = values[i] > -threshold;

      // Value below which exp(x) may underflow, but softplus(x) == exp(x)
      // is within machine epsilon.
      const tooSmall = values[i] < threshold;

      const expX = Math.exp(values[i]);
      let result;

      if (tooSmall) {
        result = expX;
      } else if (tooLarge) {
        result = values[i];
      } else {
        result = Math.log(1.0 + expX);
      }
      resultValues[i] = result;
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

  asinh<T extends Tensor>(x: T): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.asinh(values[i]);
    }
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  acosh<T extends Tensor>(x: T): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.acosh(values[i]);
    }
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  atanh<T extends Tensor>(x: T): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.atanh(values[i]);
    }
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  erf<T extends Tensor>(x: T): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    const p = erf_util.ERF_P;
    const a1 = erf_util.ERF_A1;
    const a2 = erf_util.ERF_A2;
    const a3 = erf_util.ERF_A3;
    const a4 = erf_util.ERF_A4;
    const a5 = erf_util.ERF_A5;
    for (let i = 0; i < values.length; ++i) {
      const v = values[i];
      const t = 1.0 / (1.0 + p * v);
      resultValues[i] = 1.0 -
          (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t *
              Math.exp(-v * v);
    }
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  step<T extends Tensor>(x: T, alpha = 0): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      const value = values[i];
      if (isNaN(value)) {
        resultValues[i] = NaN;
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
    const dx = ops.buffer<Rank.R4>(convInfo.inShape, 'float32');
    const dxValues = dx.values;
    const [dxS0, dxS1, dxS2] = dx.strides;
    const dyValues = dy.dataSync();
    const [dyS0, dyS1, dyS2] = dy.strides;
    const fltValues = filter.dataSync();
    const [fltS0, fltS1, fltS2] = filter.strides;
    const {
      batchSize,
      filterHeight,
      filterWidth,
      inChannels,
      inHeight,
      inWidth,
      outChannels,
      outHeight,
      outWidth,
      strideHeight,
      strideWidth
    } = convInfo;
    const topPad = filterHeight - 1 - convInfo.padInfo.top;
    const leftPad = filterWidth - 1 - convInfo.padInfo.left;

    for (let b = 0; b < batchSize; ++b) {
      for (let d1 = 0; d1 < inChannels; ++d1) {
        for (let xR = 0; xR < inHeight; ++xR) {
          const xRCorner = xR - topPad;
          const xRMin = Math.max(0, Math.ceil(xRCorner / strideHeight));
          const yRMax =
              Math.min(outHeight, (filterHeight + xRCorner) / strideHeight);

          for (let xC = 0; xC < inWidth; ++xC) {
            const xCCorner = xC - leftPad;
            const xCMin = Math.max(0, Math.ceil(xCCorner / strideWidth));
            const yCMax =
                Math.min(outWidth, (filterWidth + xCCorner) / strideWidth);

            let dotProd = 0;
            for (let yR = xRMin; yR < yRMax; ++yR) {
              const wR = yR * strideHeight - xRCorner;

              for (let yC = xCMin; yC < yCMax; ++yC) {
                const wC = yC * strideWidth - xCCorner;
                const dyOffset = dyS0 * b + dyS1 * yR + dyS2 * yC;
                const fltOffset = fltS0 * (filterHeight - 1 - wR) +
                    fltS1 * (filterWidth - 1 - wC) + fltS2 * d1;

                for (let d2 = 0; d2 < outChannels; ++d2) {
                  const pixel = dyValues[dyOffset + d2];
                  const weight = fltValues[fltOffset + d2];
                  dotProd += pixel * weight;
                }
              }
            }
            dxValues[dxS0 * b + dxS1 * xR + dxS2 * xC + d1] = dotProd;
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

  depthwiseConv2DDerInput(dy: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    const dx = ops.buffer<Rank.R4>(convInfo.inShape, 'float32');
    const dxValues = dx.values;
    const [dxS0, dxS1, dxS2] = dx.strides;
    const dyValues = dy.dataSync();
    const [dyS0, dyS1, dyS2] = dy.strides;
    const fltValues = filter.dataSync();
    const [fltS0, fltS1, fltS2] = filter.strides;
    const {
      batchSize,
      filterHeight,
      filterWidth,
      inChannels,
      inHeight,
      inWidth,
      outChannels,
      outHeight,
      outWidth,
      strideHeight,
      strideWidth
    } = convInfo;
    const topPad = filterHeight - 1 - convInfo.padInfo.top;
    const leftPad = filterWidth - 1 - convInfo.padInfo.left;
    const chMul = outChannels / inChannels;

    for (let b = 0; b < batchSize; ++b) {
      for (let d1 = 0; d1 < inChannels; ++d1) {
        for (let xR = 0; xR < inHeight; ++xR) {
          const xRCorner = xR - topPad;
          const xRMin = Math.max(0, Math.ceil(xRCorner / strideHeight));
          const yRMax =
              Math.min(outHeight, (filterHeight + xRCorner) / strideHeight);

          for (let xC = 0; xC < inWidth; ++xC) {
            const xCCorner = xC - leftPad;
            const xCMin = Math.max(0, Math.ceil(xCCorner / strideWidth));
            const yCMax =
                Math.min(outWidth, (filterWidth + xCCorner) / strideWidth);

            let dotProd = 0;
            for (let yR = xRMin; yR < yRMax; ++yR) {
              const wR = yR * strideHeight - xRCorner;

              for (let yC = xCMin; yC < yCMax; ++yC) {
                const wC = yC * strideWidth - xCCorner;
                const dyOffset = dyS0 * b + dyS1 * yR + dyS2 * yC;
                const fltOffset = fltS0 * (filterHeight - 1 - wR) +
                    fltS1 * (filterWidth - 1 - wC) + fltS2 * d1;

                for (let dm = 0; dm < chMul; ++dm) {
                  const d2 = d1 * chMul + dm;
                  const pixel = dyValues[dyOffset + d2];
                  const weight = fltValues[fltOffset + dm];
                  dotProd += pixel * weight;
                }
              }
            }
            dxValues[dxS0 * b + dxS1 * xR + dxS2 * xC + d1] = dotProd;
          }
        }
      }
    }
    return dx.toTensor();
  }

  depthwiseConv2DDerFilter(x: Tensor4D, dy: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const dW = ops.buffer<Rank.R4>(convInfo.filterShape, 'float32');

    const leftPad = convInfo.padInfo.left;
    const topPad = convInfo.padInfo.top;
    const chMul = convInfo.outChannels / convInfo.inChannels;

    for (let wR = 0; wR < filterHeight; ++wR) {
      const yRMin = Math.max(0, Math.ceil((topPad - wR) / strideHeight));
      const yRMax = Math.min(
          convInfo.outHeight, (convInfo.inHeight + topPad - wR) / strideHeight);

      for (let wC = 0; wC < filterWidth; ++wC) {
        const yCMin = Math.max(0, Math.ceil((leftPad - wC) / strideWidth));
        const yCMax = Math.min(
            convInfo.outWidth, (convInfo.inWidth + leftPad - wC) / strideWidth);

        for (let d2 = 0; d2 < convInfo.outChannels; ++d2) {
          const d1 = Math.trunc(d2 / chMul);
          const dm = d2 % chMul;

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
          dW.set(dotProd, wR, wC, d1, dm);
        }
      }
    }
    return dW.toTensor();
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

  private pool(x: Tensor4D, convInfo: Conv2DInfo, poolType: 'max'|'avg'):
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
            let count = 0;
            for (let xR = xRMin; xR < xRMax; ++xR) {
              for (let xC = xCMin; xC < xCMax; ++xC) {
                const pixel = x.get(b, xR, xC, d);
                if ((poolType === 'max' && pixel > minMaxValue)) {
                  minMaxValue = pixel;
                } else if (poolType === 'avg') {
                  avgValue += pixel;
                  count++;
                }
              }
              if (isNaN(minMaxValue)) {
                break;
              }
            }
            y.set(
                poolType === 'avg' ? avgValue / count : minMaxValue, b, yR, yC,
                d);
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

  maxPoolBackprop(dy: Tensor4D, x: Tensor4D, y: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
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

  avgPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    return this.pool(x, convInfo, 'avg').toFloat();
  }

  resizeBilinear(
      x: Tensor4D, newHeight: number, newWidth: number,
      alignCorners: boolean): Tensor4D {
    const [batch, oldHeight, oldWidth, numChannels] = x.shape;
    const output =
        ops.buffer<Rank.R4>([batch, newHeight, newWidth, numChannels], x.dtype);

    const effectiveInputSize: [number, number] = [
      (alignCorners && newHeight > 1) ? oldHeight - 1 : oldHeight,
      (alignCorners && newWidth > 1) ? oldWidth - 1 : oldWidth
    ];

    const effectiveOutputSize: [number, number] = [
      (alignCorners && newHeight > 1) ? newHeight - 1 : newHeight,
      (alignCorners && newWidth > 1) ? newWidth - 1 : newWidth
    ];

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

  resizeBilinearBackprop(dy: Tensor4D, x: Tensor4D, alignCorners: boolean) {
    const [batch, xHeight, xWidth, depth] = x.shape;
    const [, yHeight, yWidth] = dy.shape;

    const output =
        ops.buffer<Rank.R4>([batch, xHeight, xWidth, depth], x.dtype);

    // In the backwards pass, we want to find the pixels that were generated for
    // each pixel in the input image the forward pass and add the corresponding
    // coefficient from dy to the gradient (with some interpolation).

    const effectiveXSize: [number, number] = [
      (alignCorners && yHeight > 1) ? xHeight - 1 : xHeight,
      (alignCorners && yWidth > 1) ? xWidth - 1 : xWidth
    ];

    const effectiveYSize: [number, number] = [
      (alignCorners && yHeight > 1) ? yHeight - 1 : yHeight,
      (alignCorners && yWidth > 1) ? yWidth - 1 : yWidth
    ];

    const heightScale = effectiveXSize[0] / effectiveYSize[0];
    const widthScale = effectiveXSize[1] / effectiveYSize[1];

    // Reference implementation
    // tslint:disable-next-line:max-line-length
    // https://github.com/tensorflow/tensorflow/blob/3039375c86a5bbc9610c7725dcaa95d635f87ba2/tensorflow/core/kernels/resize_bilinear_op.cc#L275

    for (let b = 0; b < batch; b++) {
      for (let r = 0; r < yHeight; r++) {
        const dxR = r * heightScale;
        const topDxRIndex = Math.floor(dxR);
        const bottomDxRIndex = Math.min(Math.ceil(dxR), xHeight - 1);
        const dxRLerp = dxR - topDxRIndex;
        const inverseDxRLerp = 1.0 - dxRLerp;

        for (let c = 0; c < yWidth; c++) {
          const dxC = c * widthScale;
          const leftDxCIndex = Math.floor(dxC);
          const rightDxCIndex = Math.min(Math.ceil(dxC), xWidth - 1);
          const dxCLerp = dxC - leftDxCIndex;
          const inverseDxCLerp = 1.0 - dxCLerp;

          for (let d = 0; d < depth; d++) {
            const dyVal = dy.get(b, r, c, d);

            let topLeft = output.get(b, topDxRIndex, leftDxCIndex, d);
            topLeft += dyVal * inverseDxRLerp * inverseDxCLerp;
            output.set(topLeft, b, topDxRIndex, leftDxCIndex, d);

            let topRight = output.get(b, topDxRIndex, rightDxCIndex, d);
            topRight += dyVal * inverseDxRLerp * dxCLerp;
            output.set(topRight, b, topDxRIndex, rightDxCIndex, d);

            let bottomLeft = output.get(b, bottomDxRIndex, leftDxCIndex, d);
            bottomLeft += dyVal * dxRLerp * inverseDxCLerp;
            output.set(bottomLeft, b, bottomDxRIndex, leftDxCIndex, d);

            let bottomRight = output.get(b, bottomDxRIndex, rightDxCIndex, d);
            bottomRight += dyVal * dxRLerp * dxCLerp;
            output.set(bottomRight, b, bottomDxRIndex, rightDxCIndex, d);
          }
        }
      }
    }

    return output.toTensor();
  }

  resizeNearestNeighbor(
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
            const sourceNearestRow = Math.min(
                oldHeight - 1,
                alignCorners ? Math.round(sourceFracRow) :
                               Math.floor(sourceFracRow));
            const sourceNearestCol = Math.min(
                oldWidth - 1,
                alignCorners ? Math.round(sourceFracCol) :
                               Math.floor(sourceFracCol));
            const newValue = x.get(b, sourceNearestRow, sourceNearestCol, d);
            output.set(newValue, b, r, c, d);
          }
        }
      }
    }

    return output.toTensor();
  }

  batchNormalization(
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
      x: Tensor4D, radius: number, bias: number, alpha: number,
      beta: number): Tensor4D {
    const output = ops.buffer<Rank.R4>(x.shape, 'float32');
    const rad = radius;
    const maxD = output.shape[3] - 1;

    function sumAcrossChannels(
        b: number, r: number, c: number, d: number): number {
      let sum = 0.0;
      for (let j = Math.max(0, d - rad); j <= Math.min(d + rad, maxD); j++) {
        const z = x.get(b, r, c, j);
        sum += z * z;
      }
      return sum;
    }

    for (let b = 0; b < output.shape[0]; b++) {
      for (let r = 0; r <= output.shape[1]; r++) {
        for (let c = 0; c < output.shape[2]; c++) {
          for (let d = 0; d < output.shape[3]; d++) {
            const sum = sumAcrossChannels(b, r, c, d);
            const val = x.get(b, r, c, d) * Math.pow(bias + alpha * sum, -beta);
            output.set(val, b, r, c, d);
          }
        }
      }
    }

    return output.toTensor();
  }

  multinomial(
      logits: Tensor2D, normalized: boolean, numSamples: number,
      seed: number): Tensor2D {
    const probabilities = normalized ? logits : ops.softmax(logits);
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
      if (indices.get(event) >= 0 && indices.get(event) < depth) {
        res[event * depth + indices.get(event)] = onValue;
      }
    }
    return ops.tensor2d(res, [indices.size, depth], 'int32');
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

ENV.registerBackend('cpu', () => new MathBackendCPU(), 1 /* priority */);
