/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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

import * as tf from '@tensorflow/tfjs-core';
import {backend_util, BackendTimingInfo, buffer, DataStorage, DataType, DataValues, engine, env, kernel_impls, KernelBackend, NumericDataType, Rank, Scalar, ShapeMap, Tensor, Tensor1D, Tensor2D, Tensor4D, Tensor5D, TensorBuffer, TensorInfo, TypedArray, upcastType, util} from '@tensorflow/tfjs-core';

const nonMaxSuppressionV3Impl = kernel_impls.nonMaxSuppressionV3Impl;
const split = kernel_impls.split;
const tile = kernel_impls.tile;
const topkImpl = kernel_impls.topkImpl;
const whereImpl = kernel_impls.whereImpl;
import * as seedrandom from 'seedrandom';
import {assertNotComplex} from './cpu_util';

interface DataId {}

export interface TensorData<D extends DataType> {
  values?: backend_util.BackendValues;
  dtype: D;
  // For complex numbers, the real and imaginary parts are stored as their own
  // individual tensors, with a parent joining the two with the
  // complexTensorInfos field.
  complexTensorInfos?: {real: TensorInfo, imag: TensorInfo};
  // refCount keeps track of how many tensors reference it. Used for memory
  // management.
  refCount: number;
}

export class MathBackendCPU extends KernelBackend {
  public blockSize = 48;

  data: DataStorage<TensorData<DataType>>;
  private firstUse = true;

  constructor() {
    super();
    this.data = new DataStorage(this, engine());
  }

  write(values: backend_util.BackendValues, shape: number[], dtype: DataType):
      DataId {
    if (this.firstUse) {
      this.firstUse = false;
      if (env().get('IS_NODE')) {
        backend_util.warn(
            '\n============================\n' +
            'Hi there 👋. Looks like you are running TensorFlow.js in ' +
            'Node.js. To speed things up dramatically, install our node ' +
            'backend, which binds to TensorFlow C++, by running ' +
            'npm i @tensorflow/tfjs-node, ' +
            'or npm i @tensorflow/tfjs-node-gpu if you have CUDA. ' +
            'Then call require(\'@tensorflow/tfjs-node\'); (-gpu ' +
            'suffix for CUDA) at the start of your program. ' +
            'Visit https://github.com/tensorflow/tfjs-node for more details.' +
            '\n============================');
      }
    }
    const dataId = {};

    this.data.set(dataId, {values, dtype, refCount: 1});

    return dataId;
  }

  /**
   * Create a data bucket in cpu backend.
   * @param shape Shape of the `TensorInfo`.
   * @param dtype DType of the `TensorInfo`.
   * @param values The value of the `TensorInfo` stored as a flattened array.
   */
  makeTensorInfo(
      shape: number[], dtype: DataType,
      values?: backend_util.BackendValues|string[]): TensorInfo {
    let outId;
    if (dtype === 'string' && values != null && values.length > 0 &&
        util.isString(values[0])) {
      const encodedValues =
          (values as {} as string[]).map(d => util.encodeString(d));

      outId = this.write(encodedValues, shape, dtype);
    } else {
      outId = this.write(values as TypedArray, shape, dtype);
    }

    return {dataId: outId, shape, dtype};
  }

  /** Increase refCount of a `TensorData`. */
  incRef(dataId: DataId): void {
    const tensorData = this.data.get(dataId);
    tensorData.refCount++;
  }

  /** Decrease refCount of a `TensorData`. */
  decRef(dataId: DataId): void {
    if (this.data.has(dataId)) {
      const tensorData = this.data.get(dataId);
      tensorData.refCount--;
    }
  }

  move(
      dataId: DataId, values: backend_util.BackendValues, shape: number[],
      dtype: DataType): void {
    this.data.set(dataId, {values, dtype, refCount: 1});
  }

  numDataIds(): number {
    return this.data.numDataIds();
  }

  async read(dataId: DataId): Promise<backend_util.BackendValues> {
    return this.readSync(dataId);
  }
  readSync(dataId: DataId): backend_util.BackendValues {
    const {dtype, complexTensorInfos} = this.data.get(dataId);

    if (dtype === 'complex64') {
      const realValues =
          this.readSync(complexTensorInfos.real.dataId) as Float32Array;
      const imagValues =
          this.readSync(complexTensorInfos.imag.dataId) as Float32Array;
      return backend_util.mergeRealAndImagArrays(realValues, imagValues);
    }

    return this.data.get(dataId).values;
  }

  bufferSync<R extends Rank>(t: TensorInfo): TensorBuffer<R> {
    const data = this.readSync(t.dataId);
    let decodedData = data as DataValues;
    if (t.dtype === 'string') {
      try {
        // Decode the bytes into string.
        decodedData = (data as Uint8Array[]).map(d => util.decodeString(d));
      } catch {
        throw new Error('Failed to decode encoded string bytes into utf-8');
      }
    }
    return buffer(t.shape as ShapeMap[R], t.dtype, decodedData) as
        TensorBuffer<R>;
  }

  makeOutput<T extends Tensor>(
      values: backend_util.BackendValues, shape: number[], dtype: DataType): T {
    const dataId = this.write(values, shape, dtype);
    return engine().makeTensorFromDataId(dataId, shape, dtype, this) as T;
  }

  disposeData(dataId: DataId): void {
    if (this.data.has(dataId)) {
      const {complexTensorInfos} = this.data.get(dataId);

      if (complexTensorInfos != null) {
        this.disposeData(complexTensorInfos.real.dataId);
        this.disposeData(complexTensorInfos.imag.dataId);
      }

      this.data.delete(dataId);
    }
  }

  disposeIntermediateTensorInfo(tensorInfo: TensorInfo): void {
    const dataId = tensorInfo.dataId;

    if (this.data.has(dataId)) {
      const tensorData = this.data.get(dataId);

      tensorData.refCount--;

      if (tensorData.refCount < 1) {
        this.disposeData(dataId);
      }
    }
  }

  async time(f: () => void): Promise<BackendTimingInfo> {
    const start = util.now();
    f();
    const kernelMs = util.now() - start;
    return {kernelMs};
  }

  memory() {
    return {
      // Unreliable due to automatic gc. The numbers above are cumulative.
      unreliable: true,
      reasons:
          ['The reported memory is an upper bound. Due to automatic garbage ' +
           'collection, the true allocated memory may be less.']
    };
  }

  diag(x: Tensor): Tensor {
    const xVals = this.readSync(x.dataId) as TypedArray;
    const buffer = tf.buffer([x.size, x.size], x.dtype);
    const vals = buffer.values;
    for (let i = 0; i < xVals.length; i++) {
      vals[i * x.size + i] = xVals[i];
    }
    return buffer.toTensor();
  }

  unsortedSegmentSum<T extends Tensor>(
      x: T, segmentIds: Tensor1D, numSegments: number): Tensor {
    assertNotComplex(x, 'unsortedSegmentSum');

    const res = [];

    // Reshape the segment id's so that they can be broadcast with
    // x. The new shape should be [segmentIds.shape, 1, ..., 1]
    const numIters = x.rank - segmentIds.rank;
    for (let i = 0; i < numIters; ++i) {
      segmentIds = segmentIds.expandDims(i + 1);
    }

    for (let i = 0; i < numSegments; ++i) {
      const segmentId = tf.scalar(i, 'int32');
      const mask = tf.equal(segmentId, segmentIds).asType('float32');
      const sum = mask.mul(x).sum(0);
      res.push(sum);
    }

    return tf.stack(res);
  }

  argMin(x: Tensor, axis: number): Tensor {
    assertNotComplex(x, 'argMin');

    const axes = [axis];
    backend_util.assertAxesAreInnerMostDims('argMin', axes, x.rank);
    const [outShape, reduceShape] =
        backend_util.computeOutAndReduceShapes(x.shape, axes);
    const result = tf.zeros(outShape, 'int32');
    const reduceSize = util.sizeFromShape(reduceShape);
    const vals = this.readSync(result.dataId) as TypedArray;

    const aVals = this.readSync(x.dataId) as TypedArray;
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
    assertNotComplex(x, 'argMax');

    const axes = [axis];
    backend_util.assertAxesAreInnerMostDims('argMax', axes, x.rank);
    const [outShape, reduceShape] =
        backend_util.computeOutAndReduceShapes(x.shape, axes);
    const result = tf.zeros(outShape, 'int32');
    const reduceSize = util.sizeFromShape(reduceShape);
    const vals = this.readSync(result.dataId) as TypedArray;

    const aVals = this.readSync(x.dataId) as TypedArray;
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
    assertNotComplex(x, 'cumsum');

    if (axis !== x.rank - 1) {
      throw new Error(
          `backend.cumsum in CPU expects an inner-most axis=${x.rank - 1} ` +
          `but got axis=${axis}`);
    }
    const resultDtype = upcastType(x.dtype, 'int32');
    const result = tf.zeros(x.shape, resultDtype);
    const vals = this.readSync(result.dataId) as TypedArray;

    const aVals = this.readSync(x.dataId) as TypedArray;
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

  select(condition: Tensor, a: Tensor, b: Tensor): Tensor {
    assertNotComplex([condition, a, b], 'select');

    const values = this.readSync(condition.dataId) as TypedArray;
    const aValues = this.readSync(a.dataId) as TypedArray;
    const bValues = this.readSync(b.dataId) as TypedArray;
    const result = tf.zeros(a.shape, upcastType(a.dtype, b.dtype));
    const newValues = this.readSync(result.dataId) as TypedArray;
    let index = 0;
    const offset = condition.rank === 0 || condition.rank > 1 || a.rank === 1 ?
        1 :
        util.sizeFromShape(a.shape.slice(1));

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

  where(condition: Tensor): Tensor2D {
    assertNotComplex([condition], 'where');

    const condVals = this.readSync(condition.dataId) as TypedArray;
    return whereImpl(condition.shape, condVals);
  }

  topk<T extends Tensor>(x: T, k: number, sorted: boolean): [T, T] {
    assertNotComplex(x, 'topk');

    const xVals = this.readSync(x.dataId) as TypedArray;
    return topkImpl(xVals, x.shape, x.dtype as NumericDataType, k, sorted);
  }

  min(x: Tensor, axes: number[]): Tensor {
    assertNotComplex(x, 'min');

    backend_util.assertAxesAreInnerMostDims('min', axes, x.rank);
    const [outShape, reduceShape] =
        backend_util.computeOutAndReduceShapes(x.shape, axes);
    const result = tf.zeros(outShape, x.dtype);
    const reduceSize = util.sizeFromShape(reduceShape);
    const vals = this.readSync(result.dataId) as TypedArray;

    const aVals = this.readSync(x.dataId) as TypedArray;
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
    assertNotComplex([a, b], 'minimum');

    return this.broadcastedBinaryOp(
        a, b, a.dtype, (aVal, bVal) => Math.min(aVal, bVal));
  }

  mod(a: Tensor, b: Tensor): Tensor {
    assertNotComplex([a, b], 'mod');

    return this.broadcastedBinaryOp(a, b, a.dtype, (aVal, bVal) => {
      const rem = aVal % bVal;
      if ((aVal < 0 && bVal < 0) || (aVal >= 0 && bVal >= 0)) {
        return rem;
      } else {
        return (rem + bVal) % bVal;
      }
    });
  }

  maximum(a: Tensor, b: Tensor): Tensor {
    assertNotComplex([a, b], 'maximum');

    return this.broadcastedBinaryOp(
        a, b, a.dtype, (aVal, bVal) => Math.max(aVal, bVal));
  }

  all(x: Tensor, axes: number[]): Tensor {
    assertNotComplex(x, 'all');

    backend_util.assertAxesAreInnerMostDims('all', axes, x.rank);
    const [outShape, reduceShape] =
        backend_util.computeOutAndReduceShapes(x.shape, axes);
    const result = tf.zeros(outShape, x.dtype);
    const reduceSize = util.sizeFromShape(reduceShape);
    const vals = this.readSync(result.dataId) as TypedArray;

    const aVals = this.readSync(x.dataId) as TypedArray;
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

  any(x: Tensor, axes: number[]): Tensor {
    assertNotComplex(x, 'any');

    backend_util.assertAxesAreInnerMostDims('any', axes, x.rank);
    const [outShape, reduceShape] =
        backend_util.computeOutAndReduceShapes(x.shape, axes);
    const result = tf.zeros(outShape, x.dtype);
    const reduceSize = util.sizeFromShape(reduceShape);
    const vals = this.readSync(result.dataId) as TypedArray;

    const aVals = this.readSync(x.dataId) as TypedArray;
    for (let i = 0; i < vals.length; ++i) {
      const offset = i * reduceSize;
      let anyVal = aVals[offset];
      for (let j = 0; j < reduceSize; ++j) {
        const value = aVals[offset + j];
        anyVal = anyVal || value;
      }
      vals[i] = anyVal;
    }
    return result;
  }

  squaredDifference(a: Tensor, b: Tensor): Tensor {
    assertNotComplex([a, b], 'squaredDifference');

    return this.broadcastedBinaryOp(a, b, a.dtype, (aVal, bVal) => {
      const diff = aVal - bVal;
      return diff * diff;
    });
  }

  eluDer<T extends Tensor>(dy: T, y: T): T {
    assertNotComplex([dy, y], 'eluDer');

    const resultValues = new Float32Array(y.size);
    const values = this.readSync(y.dataId) as TypedArray;
    const dyValues = this.readSync(dy.dataId) as TypedArray;
    for (let i = 0; i < values.length; ++i) {
      const v = values[i];
      if (v >= 1) {
        resultValues[i] = dyValues[i];
      } else {
        resultValues[i] = dyValues[i] * (v + 1);
      }
    }
    return this.makeOutput(resultValues, y.shape, 'float32');
  }

  atan2<T extends Tensor>(a: T, b: T): T {
    assertNotComplex([a, b], 'atan2');

    return this.broadcastedBinaryOp(
               a, b, a.dtype, (aValue, bValue) => Math.atan2(aValue, bValue)) as
        T;
  }

  tile<T extends Tensor>(x: T, reps: number[]): T {
    assertNotComplex(x, 'tile');
    return tile(this.bufferSync(x), reps) as T;
  }

  gather<T extends Tensor>(x: T, indices: Tensor1D, axis: number): T {
    assertNotComplex([x, indices], 'gather');

    const newShape: number[] = x.shape.slice();
    const indicesValues = this.readSync(indices.dataId) as TypedArray;
    newShape[axis] = indicesValues.length;
    const result = tf.buffer(newShape, x.dtype);
    const xBuf = this.bufferSync(x);

    for (let i = 0; i < result.size; ++i) {
      const newLoc = result.indexToLoc(i);

      const originalLoc: number[] = newLoc.slice();
      originalLoc[axis] = indicesValues[newLoc[axis]];

      const originalIndex = xBuf.locToIndex(originalLoc);
      result.values[i] = xBuf.values[originalIndex];
    }
    return result.toTensor() as T;
  }

  batchToSpaceND<T extends Tensor>(
      x: T, blockShape: number[], crops: number[][]): T {
    assertNotComplex([x], 'batchToSpaceND');

    const prod = blockShape.reduce((a, b) => a * b);

    const reshaped = backend_util.getReshaped(x.shape, blockShape, prod);
    const permuted =
        backend_util.getPermuted(reshaped.length, blockShape.length);
    const reshapedPermuted =
        backend_util.getReshapedPermuted(x.shape, blockShape, prod);
    const sliceBeginCoords =
        backend_util.getSliceBeginCoords(crops, blockShape.length);
    const sliceSize =
        backend_util.getSliceSize(reshapedPermuted, crops, blockShape.length);

    return tf.transpose(x.reshape(reshaped), permuted)
               .reshape(reshapedPermuted)
               .slice(sliceBeginCoords, sliceSize) as T;
  }

  private pool3d(
      x: Tensor5D, convInfo: backend_util.Conv3DInfo,
      poolType: 'max'|'avg'): Tensor5D {
    assertNotComplex(x, 'pool3d');

    const strideDepth = convInfo.strideDepth;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationDepth = convInfo.dilationDepth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const effectiveFilterDepth = convInfo.effectiveFilterDepth;
    const effectiveFilterHeight = convInfo.effectiveFilterHeight;
    const effectiveFilterWidth = convInfo.effectiveFilterWidth;
    const padFront = convInfo.padInfo.front;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;

    const initialValue =
        (poolType === 'max' ? Number.NEGATIVE_INFINITY :
                              Number.POSITIVE_INFINITY);

    const xValues = this.readSync(x.dataId) as TypedArray;
    const output = tf.buffer(convInfo.outShape, x.dtype);
    const outputVals = output.values;

    const outputBatchStrides = convInfo.outShape[1] * convInfo.outShape[2] *
        convInfo.outShape[3] * convInfo.outShape[4];
    const outputDepthStrides =
        convInfo.outShape[2] * convInfo.outShape[3] * convInfo.outShape[4];
    const outputRowStrides = convInfo.outShape[3] * convInfo.outShape[4];
    const outputColStrides = convInfo.outShape[4];

    for (let batch = 0; batch < convInfo.batchSize; ++batch) {
      const outputBatchOffset = batch * outputBatchStrides;
      const inputBatchOffset = batch * x.strides[0];
      for (let channel = 0; channel < convInfo.inChannels; ++channel) {
        for (let yDepth = 0; yDepth < convInfo.outDepth; ++yDepth) {
          const xDepthCorner = yDepth * strideDepth - padFront;
          let xDepthMin = xDepthCorner;
          while (xDepthMin < 0) {
            xDepthMin += dilationDepth;
          }
          const xDepthMax =
              Math.min(convInfo.inDepth, effectiveFilterDepth + xDepthCorner);
          const outputDepthOffset =
              outputBatchOffset + yDepth * outputDepthStrides;
          for (let yRow = 0; yRow < convInfo.outHeight; ++yRow) {
            const xRowCorner = yRow * strideHeight - padTop;
            let xRowMin = xRowCorner;
            while (xRowMin < 0) {
              xRowMin += dilationHeight;
            }
            const xRowMax =
                Math.min(convInfo.inHeight, effectiveFilterHeight + xRowCorner);
            const outputRowOffset = outputDepthOffset + yRow * outputRowStrides;
            for (let yCol = 0; yCol < convInfo.outWidth; ++yCol) {
              const xColCorner = yCol * strideWidth - padLeft;
              let xColMin = xColCorner;
              while (xColMin < 0) {
                xColMin += dilationWidth;
              }
              const xColMax =
                  Math.min(convInfo.inWidth, effectiveFilterWidth + xColCorner);
              // Shader code begins
              const outputColOffset = outputRowOffset + yCol * outputColStrides;
              let minMaxValue = initialValue;
              let avgValue = 0;
              let count = 0;
              for (let xDepth = xDepthMin; xDepth < xDepthMax;
                   xDepth += dilationDepth) {
                const xDepthOffset = inputBatchOffset + xDepth * x.strides[1];
                for (let xRow = xRowMin; xRow < xRowMax;
                     xRow += dilationHeight) {
                  const xRowOffset = xDepthOffset + xRow * x.strides[2];
                  for (let xCol = xColMin; xCol < xColMax;
                       xCol += dilationWidth) {
                    const xColOffset = xRowOffset + xCol * x.strides[3];
                    const pixel = xValues[xColOffset + channel];
                    if ((poolType === 'max' && pixel > minMaxValue)) {
                      minMaxValue = pixel;
                    } else if (poolType === 'avg') {
                      avgValue += pixel;
                      count++;
                    }
                    if (isNaN(minMaxValue)) {
                      break;
                    }
                  }
                  if (isNaN(minMaxValue)) {
                    break;
                  }
                }
                if (isNaN(minMaxValue)) {
                  break;
                }
              }
              const outputOffset = outputColOffset + channel;
              outputVals[outputOffset] =
                  poolType === 'avg' ? avgValue / count : minMaxValue;
            }
          }
        }
      }
    }
    return output.toTensor() as Tensor5D;
  }

  avgPool3d(x: Tensor5D, convInfo: backend_util.Conv3DInfo): Tensor5D {
    assertNotComplex(x, 'avgPool3d');

    return this.pool3d(x, convInfo, 'avg').toFloat();
  }

  avgPool3dBackprop(
      dy: Tensor5D, x: Tensor5D, convInfo: backend_util.Conv3DInfo): Tensor5D {
    assertNotComplex([dy, x], 'avgPool3dBackprop');

    const strideDepth = convInfo.strideDepth;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const filterDepth = convInfo.filterDepth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const dilationDepth = convInfo.dilationDepth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const effectiveFilterDepth = convInfo.effectiveFilterDepth;
    const effectiveFilterHeight = convInfo.effectiveFilterHeight;
    const effectiveFilterWidth = convInfo.effectiveFilterWidth;
    const padFront = effectiveFilterDepth - 1 - convInfo.padInfo.front;
    const padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;
    const padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
    const dx = tf.buffer<Rank.R5>(x.shape, 'float32');

    const avgMultiplier = 1 / (filterDepth * filterHeight * filterWidth);

    const dyBuf = this.bufferSync(dy);

    for (let batch = 0; batch < convInfo.batchSize; ++batch) {
      for (let channel = 0; channel < convInfo.inChannels; ++channel) {
        for (let dxDepth = 0; dxDepth < convInfo.inDepth; ++dxDepth) {
          for (let dxRow = 0; dxRow < convInfo.inHeight; ++dxRow) {
            for (let dxCol = 0; dxCol < convInfo.inWidth; ++dxCol) {
              // Shader code begins.
              const dyDepthCorner = dxDepth - padFront;
              const dyRowCorner = dxRow - padTop;
              const dyColCorner = dxCol - padLeft;
              let dotProd = 0;
              for (let wDepth = 0; wDepth < effectiveFilterDepth;
                   wDepth += dilationDepth) {
                const dyDepth = (dyDepthCorner + wDepth) / strideDepth;
                if (dyDepth < 0 || dyDepth >= convInfo.outDepth ||
                    Math.floor(dyDepth) !== dyDepth) {
                  continue;
                }
                for (let wRow = 0; wRow < effectiveFilterHeight;
                     wRow += dilationHeight) {
                  const dyRow = (dyRowCorner + wRow) / strideHeight;
                  if (dyRow < 0 || dyRow >= convInfo.outHeight ||
                      Math.floor(dyRow) !== dyRow) {
                    continue;
                  }
                  for (let wCol = 0; wCol < effectiveFilterWidth;
                       wCol += dilationWidth) {
                    const dyCol = (dyColCorner + wCol) / strideWidth;
                    if (dyCol < 0 || dyCol >= convInfo.outWidth ||
                        Math.floor(dyCol) !== dyCol) {
                      continue;
                    }

                    const pixel =
                        dyBuf.get(batch, dyDepth, dyRow, dyCol, channel);
                    dotProd += pixel;
                  }
                }
              }
              dx.set(
                  dotProd * avgMultiplier, batch, dxDepth, dxRow, dxCol,
                  channel);
            }
          }
        }
      }
    }
    return dx.toTensor();
  }

  maxPool3d(x: Tensor5D, convInfo: backend_util.Conv3DInfo): Tensor5D {
    assertNotComplex(x, 'maxPool3d');

    return this.pool3d(x, convInfo, 'max').toFloat();
  }

  private maxPool3dPositions(x: Tensor5D, convInfo: backend_util.Conv3DInfo):
      Tensor5D {
    const maxPositions = tf.buffer(convInfo.outShape, 'int32');
    const strideDepth = convInfo.strideDepth;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationDepth = convInfo.dilationDepth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const effectiveFilterDepth = convInfo.effectiveFilterDepth;
    const effectiveFilterHeight = convInfo.effectiveFilterHeight;
    const effectiveFilterWidth = convInfo.effectiveFilterWidth;
    const padFront = convInfo.padInfo.front;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;

    const xBuf = this.bufferSync(x);
    for (let batch = 0; batch < convInfo.batchSize; ++batch) {
      for (let channel = 0; channel < convInfo.inChannels; ++channel) {
        for (let yDepth = 0; yDepth < convInfo.outDepth; ++yDepth) {
          const xDepthCorner = yDepth * strideDepth - padFront;
          let xDepthMin = xDepthCorner;
          while (xDepthMin < 0) {
            xDepthMin += dilationDepth;
          }
          const xDepthMax =
              Math.min(convInfo.inDepth, effectiveFilterDepth + xDepthCorner);
          for (let yRow = 0; yRow < convInfo.outHeight; ++yRow) {
            const xRowCorner = yRow * strideHeight - padTop;
            let xRowMin = xRowCorner;
            while (xRowMin < 0) {
              xRowMin += dilationHeight;
            }
            const xRowMax =
                Math.min(convInfo.inHeight, effectiveFilterHeight + xRowCorner);
            for (let yCol = 0; yCol < convInfo.outWidth; ++yCol) {
              const xColCorner = yCol * strideWidth - padLeft;
              let xColMin = xColCorner;
              while (xColMin < 0) {
                xColMin += dilationWidth;
              }
              const xColMax =
                  Math.min(convInfo.inWidth, effectiveFilterWidth + xColCorner);

              // Shader code begins
              let maxValue = Number.NEGATIVE_INFINITY;
              let maxPosition = -1;

              for (let xDepth = xDepthMin; xDepth < xDepthMax;
                   xDepth += dilationDepth) {
                const wDepth = xDepth - xDepthCorner;
                for (let xRow = xRowMin; xRow < xRowMax;
                     xRow += dilationHeight) {
                  const wRow = xRow - xRowCorner;
                  for (let xCol = xColMin; xCol < xColMax;
                       xCol += dilationWidth) {
                    const wCol = xCol - xColCorner;
                    const pixel = xBuf.get(batch, xDepth, xRow, xCol, channel);
                    if (pixel >= maxValue) {
                      maxValue = pixel;
                      maxPosition = wDepth * effectiveFilterHeight *
                              effectiveFilterWidth +
                          wRow * effectiveFilterHeight + wCol;
                    }
                  }
                }
              }

              maxPositions.set(maxPosition, batch, yDepth, yRow, yCol, channel);
            }
          }
        }
      }
    }
    return maxPositions.toTensor() as Tensor5D;
  }

  maxPool3dBackprop(
      dy: Tensor5D, x: Tensor5D, y: Tensor5D,
      convInfo: backend_util.Conv3DInfo): Tensor5D {
    assertNotComplex([x, y], 'maxPool3dBackprop');

    const maxPositions = this.maxPool3dPositions(x, convInfo);
    const strideDepth = convInfo.strideDepth;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationDepth = convInfo.dilationDepth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const effectiveFilterDepth = convInfo.effectiveFilterDepth;
    const effectiveFilterHeight = convInfo.effectiveFilterHeight;
    const effectiveFilterWidth = convInfo.effectiveFilterWidth;
    const padFront = effectiveFilterDepth - 1 - convInfo.padInfo.front;
    const padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;
    const padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
    const dx = tf.buffer<Rank.R5>(x.shape, 'float32');

    const maxPosBuf = this.bufferSync(maxPositions);
    const dyBuf = this.bufferSync(dy);

    for (let batch = 0; batch < convInfo.batchSize; ++batch) {
      for (let channel = 0; channel < convInfo.inChannels; ++channel) {
        for (let dxDepth = 0; dxDepth < convInfo.inDepth; ++dxDepth) {
          for (let dxRow = 0; dxRow < convInfo.inHeight; ++dxRow) {
            for (let dxCol = 0; dxCol < convInfo.inWidth; ++dxCol) {
              // Shader code begins
              const dyDepthCorner = dxDepth - padFront;
              const dyRowCorner = dxRow - padTop;
              const dyColCorner = dxCol - padLeft;
              let dotProd = 0;
              for (let wDepth = 0; wDepth < effectiveFilterDepth;
                   wDepth += dilationDepth) {
                const dyDepth = (dyDepthCorner + wDepth) / strideDepth;
                if (dyDepth < 0 || dyDepth >= convInfo.outDepth ||
                    Math.floor(dyDepth) !== dyDepth) {
                  continue;
                }
                for (let wRow = 0; wRow < effectiveFilterHeight;
                     wRow += dilationHeight) {
                  const dyRow = (dyRowCorner + wRow) / strideHeight;
                  if (dyRow < 0 || dyRow >= convInfo.outHeight ||
                      Math.floor(dyRow) !== dyRow) {
                    continue;
                  }
                  for (let wCol = 0; wCol < effectiveFilterWidth;
                       wCol += dilationWidth) {
                    const dyCol = (dyColCorner + wCol) / strideWidth;
                    if (dyCol < 0 || dyCol >= convInfo.outWidth ||
                        Math.floor(dyCol) !== dyCol) {
                      continue;
                    }

                    const maxPos = effectiveFilterDepth *
                            effectiveFilterHeight * effectiveFilterWidth -
                        1 -
                        maxPosBuf.get(batch, dyDepth, dyRow, dyCol, channel);
                    const curPos =
                        wDepth * effectiveFilterHeight * effectiveFilterWidth +
                        wRow * effectiveFilterWidth + wCol;

                    const mask = maxPos === curPos ? 1 : 0;
                    if (mask === 0) {
                      continue;
                    }

                    const pixel =
                        dyBuf.get(batch, dyDepth, dyRow, dyCol, channel);
                    dotProd += pixel * mask;
                  }
                }
              }
              dx.set(dotProd, batch, dxDepth, dxRow, dxCol, channel);
            }
          }
        }
      }
    }
    return dx.toTensor();
  }

  resizeBilinear(
      x: Tensor4D, newHeight: number, newWidth: number, alignCorners: boolean,
      halfPixelCenters: boolean): Tensor4D {
    assertNotComplex(x, 'resizeBilinear');

    const [batch, oldHeight, oldWidth, numChannels] = x.shape;
    const xValues = this.readSync(x.dataId) as TypedArray;
    const result = new Float32Array(
        util.sizeFromShape([batch, newHeight, newWidth, numChannels]));

    const effectiveInputSize: [number, number] = [
      (alignCorners && newHeight > 1) ? oldHeight - 1 : oldHeight,
      (alignCorners && newWidth > 1) ? oldWidth - 1 : oldWidth
    ];

    const effectiveOutputSize: [number, number] = [
      (alignCorners && newHeight > 1) ? newHeight - 1 : newHeight,
      (alignCorners && newWidth > 1) ? newWidth - 1 : newWidth
    ];
    let outputIdx = 0;
    const effectiveRowSizeRatio =
        effectiveInputSize[0] / effectiveOutputSize[0];
    const effectiveColSizeRatio =
        effectiveInputSize[1] / effectiveOutputSize[1];
    for (let b = 0; b < batch; b++) {
      for (let r = 0; r < newHeight; r++) {
        let sourceFracRow: number;
        if (halfPixelCenters) {
          sourceFracRow = effectiveRowSizeRatio * (r + 0.5) - 0.5;
        } else {
          sourceFracRow = effectiveRowSizeRatio * r;
        }

        const sourceRowFloor = Math.max(0, Math.floor(sourceFracRow));
        const rowFrac = sourceFracRow - sourceRowFloor;
        const sourceRowCeil = Math.min(oldHeight - 1, Math.ceil(sourceFracRow));
        const topRowOffset = b * x.strides[0] + sourceRowFloor * x.strides[1];
        const botRowOffset = b * x.strides[0] + sourceRowCeil * x.strides[1];
        for (let c = 0; c < newWidth; c++) {
          let sourceFracCol: number;
          if (halfPixelCenters) {
            sourceFracCol = effectiveColSizeRatio * (c + 0.5) - 0.5;
          } else {
            sourceFracCol = effectiveColSizeRatio * c;
          }
          const sourceColFloor = Math.max(0, Math.floor(sourceFracCol));
          const colFrac = sourceFracCol - sourceColFloor;
          const sourceColCeil =
              Math.min(oldWidth - 1, Math.ceil(sourceFracCol));
          const topLeftOffest = topRowOffset + sourceColFloor * x.strides[2];
          const botLeftOffset = botRowOffset + sourceColFloor * x.strides[2];
          const topRightOffset = topRowOffset + sourceColCeil * x.strides[2];
          const botRightOffest = botRowOffset + sourceColCeil * x.strides[2];
          for (let d = 0; d < numChannels; d++) {
            // Begin shader.

            // Compute the fractional index of the source.
            const topLeft = xValues[topLeftOffest + d];
            const bottomLeft = xValues[botLeftOffset + d];
            const topRight = xValues[topRightOffset + d];
            const bottomRight = xValues[botRightOffest + d];

            const top = topLeft + (topRight - topLeft) * colFrac;
            const bottom = bottomLeft + (bottomRight - bottomLeft) * colFrac;
            const newValue = top + (bottom - top) * rowFrac;

            result[outputIdx++] = newValue;
          }
        }
      }
    }
    return tf.tensor(result, [batch, newHeight, newWidth, numChannels]);
  }

  resizeBilinearBackprop(dy: Tensor4D, x: Tensor4D, alignCorners: boolean) {
    assertNotComplex([dy, x], 'resizeBilinearBackprop');

    const [batch, xHeight, xWidth, depth] = x.shape;
    const [, yHeight, yWidth] = dy.shape;

    const output = new Float32Array(batch * xHeight * xWidth * depth);

    // In the backwards pass, we want to find the pixels that were generated
    // for each pixel in the input image the forward pass and add the
    // corresponding coefficient from dy to the gradient (with some
    // interpolation).

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

    const dyValues = this.readSync(dy.dataId) as TypedArray;
    let offset = 0;
    for (let b = 0; b < batch; b++) {
      const bOffset = b * x.strides[0];
      for (let r = 0; r < yHeight; r++) {
        const dxR = r * heightScale;
        const topDxRIndex = Math.floor(dxR);
        const bottomDxRIndex = Math.min(Math.ceil(dxR), xHeight - 1);

        const topDxROffset = bOffset + topDxRIndex * x.strides[1];
        const bottomDxROffset = bOffset + bottomDxRIndex * x.strides[1];

        const dxRLerp = dxR - topDxRIndex;
        const inverseDxRLerp = 1.0 - dxRLerp;
        for (let c = 0; c < yWidth; c++) {
          const dxC = c * widthScale;
          const leftDxCIndex = Math.floor(dxC);
          const rightDxCIndex = Math.min(Math.ceil(dxC), xWidth - 1);
          const dxCLerp = dxC - leftDxCIndex;
          const inverseDxCLerp = 1.0 - dxCLerp;

          const topLeftRCOffset = topDxROffset + leftDxCIndex * x.strides[2];
          const topRightRCOffset = topDxROffset + rightDxCIndex * x.strides[2];
          const bottomLeftRCOffset =
              bottomDxROffset + leftDxCIndex * x.strides[2];
          const bottomRightRCOffset =
              bottomDxROffset + rightDxCIndex * x.strides[2];

          const inverseDxRLerpTimesInverseDxCLerp =
              inverseDxRLerp * inverseDxCLerp;
          const inverseDxRLerpTimesDxCLerp = inverseDxRLerp * dxCLerp;
          const dxRLerpTimesInverseDxCLerp = dxRLerp * inverseDxCLerp;
          const dxRLerpTimesDxCLerp = dxRLerp * dxCLerp;
          for (let d = 0; d < depth; d++) {
            const dyVal = dyValues[offset++];
            output[topLeftRCOffset + d] +=
                dyVal * inverseDxRLerpTimesInverseDxCLerp;
            output[topRightRCOffset + d] += dyVal * inverseDxRLerpTimesDxCLerp;
            output[bottomLeftRCOffset + d] +=
                dyVal * dxRLerpTimesInverseDxCLerp;
            output[bottomRightRCOffset + d] += dyVal * dxRLerpTimesDxCLerp;
          }
        }
      }
    }
    return tf.tensor4d(output, [batch, xWidth, xHeight, depth], x.dtype);
  }

  resizeNearestNeighbor(
      x: Tensor4D, newHeight: number, newWidth: number,
      alignCorners: boolean): Tensor4D {
    assertNotComplex(x, 'resizeNearestNeighbor');

    const [batch, oldHeight, oldWidth, numChannels] = x.shape;
    const xValues = this.readSync(x.dataId) as TypedArray;
    const output = new Float32Array(batch * newHeight * newWidth * numChannels);

    const effectiveInputSize: [number, number] = [
      (alignCorners && newHeight > 1) ? oldHeight - 1 : oldHeight,
      (alignCorners && newWidth > 1) ? oldWidth - 1 : oldWidth
    ];

    const effectiveOutputSize: [number, number] = [
      (alignCorners && newHeight > 1) ? newHeight - 1 : newHeight,
      (alignCorners && newWidth > 1) ? newWidth - 1 : newWidth
    ];

    const effectiveRowSizeRatio =
        effectiveInputSize[0] / effectiveOutputSize[0];
    const effectiveColSizeRatio =
        effectiveInputSize[1] / effectiveOutputSize[1];

    let outputOffset = 0;
    for (let b = 0; b < batch; b++) {
      const batchOffset = b * x.strides[0];
      for (let r = 0; r < newHeight; r++) {
        const sourceFracRow = effectiveRowSizeRatio * r;
        const sourceNearestRow = Math.min(
            oldHeight - 1,
            alignCorners ? Math.round(sourceFracRow) :
                           Math.floor(sourceFracRow));
        const rowOffset = batchOffset + sourceNearestRow * x.strides[1];
        for (let c = 0; c < newWidth; c++) {
          const sourceFracCol = effectiveColSizeRatio * c;
          const sourceNearestCol = Math.min(
              oldWidth - 1,
              alignCorners ? Math.round(sourceFracCol) :
                             Math.floor(sourceFracCol));
          const colOffset = rowOffset + sourceNearestCol * x.strides[2];
          for (let d = 0; d < numChannels; d++) {
            // Begin shader.
            // Compute the fractional index of the source.
            const newVal = xValues[colOffset + d];
            output[outputOffset++] = newVal;
          }
        }
      }
    }
    return tf.tensor(
        output, [batch, newHeight, newWidth, numChannels], x.dtype);
  }

  resizeNearestNeighborBackprop(
      dy: Tensor4D, x: Tensor4D, alignCorners: boolean) {
    assertNotComplex([dy, x], 'resizeNearestNeighborBackprop');

    const [batch, xHeight, xWidth, depth] = x.shape;
    const [, yHeight, yWidth] = dy.shape;

    const output = new Float32Array(batch * xHeight * xWidth * depth);
    const dyValues = this.readSync(dy.dataId) as TypedArray;

    // In the backwards pass, we want to find the pixels that were generated
    // for each pixel in the input image the forward pass

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

    const invHeightScale = 1 / heightScale;
    const invWidthScale = 1 / widthScale;

    // This defines the size of the window of values around a particular
    // index in dy that we want to search for contributions to dx.
    const winHeight = (Math.ceil(invHeightScale) * 2) + 2;
    const winWidth = (Math.ceil(invWidthScale) * 2) + 2;

    // Loop over the output space.
    for (let b = 0; b < batch; b++) {
      const batchOffset = b * x.strides[0];
      for (let r = 0; r < xHeight; r++) {
        const rowOffset = batchOffset + r * x.strides[1];

        // Compute bounds for where in dy we will look
        const startRLerp = Math.floor(r * invHeightScale);
        const startDyR = Math.floor(startRLerp - (winHeight / 2));
        for (let c = 0; c < xWidth; c++) {
          const colOffset = rowOffset + c * x.strides[2];

          // Compute bounds for where in dy we will look
          const startCLerp = Math.floor(c * invWidthScale);
          const startDyC = Math.floor(startCLerp - (winWidth / 2));

          for (let d = 0; d < depth; d++) {
            let accum = 0;
            // loop over dy

            for (let dyRIndex = 0; dyRIndex < winHeight; dyRIndex++) {
              const dyR = dyRIndex + startDyR;
              // Guard against the window exceeding the bounds of dy
              if (dyR < 0 || dyR >= yHeight) {
                continue;
              }

              const dyROffset = batchOffset + dyR * dy.strides[1];
              const sourceFracRow = dyR * heightScale;
              const sourceNearestRow = Math.min(
                  xHeight - 1,
                  alignCorners ? Math.round(sourceFracRow) :
                                 Math.floor(sourceFracRow));
              if (r !== sourceNearestRow) {
                continue;
              }
              for (let dyCIndex = 0; dyCIndex < winWidth; dyCIndex++) {
                const dyC = dyCIndex + startDyC;
                // Guard against the window exceeding the bounds of dy
                if (dyC < 0 || dyC >= yWidth) {
                  continue;
                }

                const dyCOffset = dyROffset + dyC * dy.strides[2];
                const sourceFracCol = dyC * widthScale;
                const sourceNearestCol = Math.min(
                    xWidth - 1,
                    alignCorners ? Math.round(sourceFracCol) :
                                   Math.floor(sourceFracCol));

                if (c === sourceNearestCol) {
                  accum += dyValues[dyCOffset + d];
                }
              }
            }
            output[colOffset + d] = accum;
          }
        }
      }
    }
    return tf.tensor4d(output, x.shape, x.dtype);
  }

  localResponseNormalization4D(
      x: Tensor4D, depthRadius: number, bias: number, alpha: number,
      beta: number): Tensor4D {
    assertNotComplex(x, 'localResponseNormalization4D');

    const channels = x.shape[3];
    const maxD = channels - 1;
    const xValues = this.readSync(x.dataId) as TypedArray;
    const size = x.size;
    const result = new Float32Array(size);

    function sumAcrossChannels(offset: number) {
      const currentChannel = offset % channels;
      let beginSumOffset =
          offset - currentChannel + Math.max(0, currentChannel - depthRadius);
      const endSumOffset = offset - currentChannel +
          Math.min(currentChannel + depthRadius, maxD);

      let sum = 0.0;
      for (; beginSumOffset <= endSumOffset; beginSumOffset++) {
        const z = xValues[beginSumOffset];
        sum += z * z;
      }
      return sum;
    }

    for (let offset = 0; offset < size; offset++) {
      const sum = sumAcrossChannels(offset);
      const val = xValues[offset] * Math.pow(bias + alpha * sum, -beta);
      result[offset] = val;
    }

    return tf.tensor4d(result, x.shape);
  }

  LRNGrad(
      dy: Tensor4D, inputImage: Tensor4D, outputImage: Tensor4D,
      depthRadius: number, bias: number, alpha: number,
      beta: number): Tensor4D {
    assertNotComplex(dy, 'LRNGrad');
    const channels = dy.shape[3];
    const dyValues = this.readSync(dy.dataId) as TypedArray;
    const inputImageValues = this.readSync(inputImage.dataId) as TypedArray;
    const outputImageValues = this.readSync(outputImage.dataId) as TypedArray;
    const result = new Float32Array(dy.size);
    const size = dy.size;

    for (let offset = 0; offset < size; offset++) {
      const currentChannel = offset % channels;
      const depthBegin =
          (offset - currentChannel) + Math.max(0, currentChannel - depthRadius);
      const depthEnd = (offset - currentChannel) +
          Math.min(channels, currentChannel + depthRadius + 1);

      let norm = 0;
      for (let k = depthBegin; k < depthEnd; k++) {
        norm += Math.pow(inputImageValues[k], 2);
      }
      norm = alpha * norm + bias;

      for (let k = depthBegin; k < depthEnd; k++) {
        let dyi = -2 * alpha * beta * inputImageValues[k] *
            outputImageValues[offset] / norm;
        if (offset === k) {
          dyi += Math.pow(norm, -beta);
        }
        dyi *= dyValues[offset];
        result[k] += dyi;
      }
    }
    return tf.tensor4d(result, dy.shape);
  }

  multinomial(
      logits: Tensor2D, normalized: boolean, numSamples: number,
      seed: number): Tensor2D {
    assertNotComplex(logits, 'multinomial');

    const probabilities = normalized ? logits : tf.softmax(logits);
    const batchSize = probabilities.shape[0];
    const numEvents = probabilities.shape[1];
    const res = tf.zeros<Rank.R2>([batchSize, numSamples], 'int32');
    const resVals = this.readSync(res.dataId) as TypedArray;
    const probVals = this.readSync(probabilities.dataId) as TypedArray;

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

  nonMaxSuppression(
      boxes: Tensor2D, scores: Tensor1D, maxOutputSize: number,
      iouThreshold: number, scoreThreshold: number): Tensor1D {
    assertNotComplex(boxes, 'nonMaxSuppression');

    const boxesVals = this.readSync(boxes.dataId) as TypedArray;
    const scoresVals = this.readSync(scores.dataId) as TypedArray;
    return nonMaxSuppressionV3Impl(
        boxesVals, scoresVals, maxOutputSize, iouThreshold, scoreThreshold);
  }

  depthToSpace(x: Tensor4D, blockSize: number, dataFormat: 'NHWC'|'NCHW'):
      Tensor4D {
    util.assert(
        dataFormat === 'NHWC',
        () => `Only NHWC dataFormat supported on CPU for depthToSpace. Got ${
            dataFormat}`);
    util.assert(
        blockSize > 1,
        () =>
            `blockSize should be > 1 for depthToSpace, but was: ${blockSize}`);

    const batchSize = x.shape[0];
    const inputHeight = x.shape[1];
    const inputWidth = x.shape[2];
    const inputDepth = x.shape[3];

    const outputHeight = inputHeight * blockSize;
    const outputWidth = inputWidth * blockSize;
    const outputDepth = inputDepth / (blockSize * blockSize);

    const xValues = this.readSync(x.dataId) as TypedArray;
    const result =
        new Float32Array(batchSize * outputHeight * outputWidth * outputDepth);

    let outputIdx = 0;
    for (let b = 0; b < batchSize; ++b) {
      for (let h = 0; h < outputHeight; ++h) {
        const inH = Math.floor(h / blockSize);
        const offsetH = (h % blockSize);
        for (let w = 0; w < outputWidth; ++w) {
          const inW = Math.floor(w / blockSize);
          const offsetW = (w % blockSize);
          const offsetD = (offsetH * blockSize + offsetW) * outputDepth;
          for (let d = 0; d < outputDepth; ++d) {
            const inD = d + offsetD;
            const inputIdx =
                inD + inputDepth * (inW + inputWidth * (inH + inputHeight * b));
            result[outputIdx++] = xValues[inputIdx];
          }
        }
      }
    }
    return tf.tensor4d(
        result, [batchSize, outputHeight, outputWidth, outputDepth]);
  }

  private broadcastedBinaryOp(
      a: Tensor, b: Tensor, dtype: DataType,
      op: (a: number, b: number) => number): Tensor {
    const newShape = backend_util.assertAndGetBroadcastShape(a.shape, b.shape);
    const result = tf.buffer(newShape, dtype);
    const aVals = this.readSync(a.dataId) as TypedArray;
    const bVals = this.readSync(b.dataId) as TypedArray;
    const aBroadcastDims = backend_util.getBroadcastDims(a.shape, newShape);
    const bBroadcastDims = backend_util.getBroadcastDims(b.shape, newShape);

    const resVals = result.values;
    if (aBroadcastDims.length + bBroadcastDims.length === 0) {
      for (let i = 0; i < resVals.length; ++i) {
        resVals[i] = op(aVals[i % aVals.length], bVals[i % bVals.length]);
      }
    } else {
      const aBuf = this.bufferSync(a);
      const bBuf = this.bufferSync(b);
      for (let i = 0; i < resVals.length; ++i) {
        const loc = result.indexToLoc(i);

        const aLoc = loc.slice(-a.rank);
        aBroadcastDims.forEach(d => aLoc[d] = 0);
        const aIndex = aBuf.locToIndex(aLoc);

        const bLoc = loc.slice(-b.rank);
        bBroadcastDims.forEach(d => bLoc[d] = 0);
        const bIndex = bBuf.locToIndex(bLoc);

        resVals[i] = op(aVals[aIndex], bVals[bIndex]);
      }
    }
    return result.toTensor();
  }

  split<T extends Tensor>(x: T, sizeSplits: number[], axis: number): T[] {
    return split(x, sizeSplits, axis);
  }

  dispose() {}

  floatPrecision(): 16|32 {
    return 32;
  }

  /** Returns the smallest representable number.  */
  epsilon(): number {
    return super.epsilon();
  }

  cropAndResize(
      images: Tensor4D,
      boxes: Tensor2D,
      boxIndex: Tensor1D,
      cropSize: [number, number],
      method: string,
      extrapolationValue: number,
  ) {
    const [batch, imageHeight, imageWidth, numChannels] = images.shape;
    const numBoxes = boxes.shape[0];

    const [cropHeight, cropWidth] = cropSize;
    const output =
        tf.buffer([numBoxes, cropHeight, cropWidth, numChannels], 'float32');

    const boxVals = this.readSync(boxes.dataId) as TypedArray;
    const boxIndVals = this.readSync(boxIndex.dataId) as TypedArray;
    const imageVals = this.readSync(images.dataId) as TypedArray;

    const inStride = images.strides;   // to calculate flat indexes into image
    const outStride = output.strides;  // to calculate flat indexes into output

    // Reference implementation
    // tslint:disable-next-line:max-line-length
    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/crop_and_resize_op.cc
    for (let b = 0; b < numBoxes; b++) {
      const startInd = b * 4;
      const y1 = boxVals[startInd];
      const x1 = boxVals[startInd + 1];
      const y2 = boxVals[startInd + 2];
      const x2 = boxVals[startInd + 3];

      const bInd: number = boxIndVals[b];
      if (bInd >= batch) {
        continue;
      }

      const heightScale = (cropHeight > 1) ?
          (y2 - y1) * (imageHeight - 1) / (cropHeight - 1) :
          0;
      const widthScale =
          (cropWidth > 1) ? (x2 - x1) * (imageWidth - 1) / (cropWidth - 1) : 0;

      for (let y = 0; y < cropHeight; y++) {
        const yInd: number = (cropHeight > 1) ?
            y1 * (imageHeight - 1) + y * (heightScale) :
            0.5 * (y1 + y2) * (imageHeight - 1);

        if (yInd < 0 || yInd > imageHeight - 1) {
          for (let x = 0; x < cropWidth; x++) {
            for (let c = 0; c < numChannels; c++) {
              const ind =
                  c + x * outStride[2] + y * outStride[1] + b * outStride[0];
              output.values[ind] = extrapolationValue;
            }
          }
          continue;
        }

        if (method === 'bilinear') {
          const topInd = Math.floor(yInd);
          const bottomInd = Math.ceil(yInd);
          const yLerp = yInd - topInd;

          for (let x = 0; x < cropWidth; x++) {
            const xInd = (cropWidth > 1) ?
                x1 * (imageWidth - 1) + x * widthScale :
                0.5 * (x1 + x2) * (imageWidth - 1);

            if (xInd < 0 || xInd > imageWidth - 1) {
              for (let c = 0; c < numChannels; c++) {
                const ind =
                    c + x * outStride[2] + y * outStride[1] + b * outStride[0];
                output.values[ind] = extrapolationValue;
              }
              continue;
            }

            const leftInd = Math.floor(xInd);
            const rightInd = Math.ceil(xInd);
            const xLerp = xInd - leftInd;

            for (let c = 0; c < numChannels; c++) {
              let ind = c + leftInd * inStride[2] + topInd * inStride[1] +
                  bInd * inStride[0];
              const topLeft = imageVals[ind];

              ind = c + rightInd * inStride[2] + topInd * inStride[1] +
                  bInd * inStride[0];
              const topRight = imageVals[ind];

              ind = c + leftInd * inStride[2] + bottomInd * inStride[1] +
                  bInd * inStride[0];
              const bottomLeft = imageVals[ind];

              ind = c + rightInd * inStride[2] + bottomInd * inStride[1] +
                  bInd * inStride[0];
              const bottomRight = imageVals[ind];

              const top = topLeft + (topRight - topLeft) * xLerp;
              const bottom = bottomLeft + (bottomRight - bottomLeft) * xLerp;

              ind = c + x * outStride[2] + y * outStride[1] + b * outStride[0];
              output.values[ind] = top + ((bottom - top) * yLerp);
            }
          }
        } else {  // method == "nearest"
          for (let x = 0; x < cropWidth; ++x) {
            const xInd = (cropWidth > 1) ?
                x1 * (imageWidth - 1) + x * widthScale :
                0.5 * (x1 + x2) * (imageWidth - 1);

            if (xInd < 0 || xInd > imageWidth - 1) {
              for (let c = 0; c < numChannels; c++) {
                const ind =
                    c + x * outStride[2] + y * outStride[1] + b * outStride[0];
                output.values[ind] = extrapolationValue;
              }
              continue;
            }

            const closestX = Math.round(xInd);
            const closestY = Math.round(yInd);
            for (let c = 0; c < numChannels; c++) {
              const inInd = c + closestX * inStride[2] +
                  closestY * inStride[1] + bInd * inStride[0];
              const outInd =
                  c + x * outStride[2] + y * outStride[1] + b * outStride[0];
              output.values[outInd] = imageVals[inInd];
            }
          }
        }
      }
    }
    return output.toTensor() as Tensor4D;
  }

  sparseToDense<R extends Rank>(
      sparseIndices: Tensor, sparseValues: Tensor, outputShape: ShapeMap[R],
      defaultValue: Scalar): Tensor<R> {
    const {sliceRank, numUpdates, sliceSize, strides, outputSize} =
        backend_util.calculateShapes(sparseValues, sparseIndices, outputShape);
    const sumDupeIndices = false;
    return this.scatter(
        sparseIndices, sparseValues, outputShape, outputSize, sliceSize,
        numUpdates, sliceRank, strides, defaultValue, sumDupeIndices);
  }

  gatherND(x: Tensor, indices: Tensor): Tensor {
    const indicesShape = indices.shape;
    const sliceRank = indicesShape[indicesShape.length - 1];

    const [resultShape, numSlices, sliceSize, strides] =
        backend_util.prepareAndValidate(x, indices);
    if (numSlices === 0) {
      return tf.tensor([], resultShape, x.dtype);
    }

    const buffer = new TensorBuffer([numSlices, sliceSize], x.dtype);
    const indicesData = this.readSync(indices.dataId) as TypedArray;
    const xData = this.readSync(x.dataId) as TypedArray;

    for (let i = 0; i < numSlices; i++) {
      const index = [];
      let flattenIndex = 0;
      for (let j = 0; j < sliceRank; j++) {
        const dim = indicesData[i * sliceRank + j];
        flattenIndex += dim * strides[j];
        index.push(dim);
      }
      if (flattenIndex < 0 || flattenIndex >= x.size / sliceSize) {
        throw new Error(
            `Invalid indices: ${index} does not index into ${x.shape}`);
      }

      for (let k = 0; k < sliceSize; k++) {
        buffer.values[i * sliceSize + k] = xData[flattenIndex * sliceSize + k];
      }
    }
    return buffer.toTensor().reshape(resultShape);
  }

  scatterND<R extends Rank>(
      indices: Tensor, updates: Tensor, shape: ShapeMap[R]): Tensor<R> {
    const {sliceRank, numUpdates, sliceSize, strides, outputSize} =
        backend_util.calculateShapes(updates, indices, shape);
    const defaultValue = tf.scalar(0);
    const sumDupeIndices = true;
    return this.scatter(
        indices, updates, shape, outputSize, sliceSize, numUpdates, sliceRank,
        strides, defaultValue, sumDupeIndices);
  }

  onesLike<R extends Rank>(x: Tensor<R>): Tensor<R> {
    if (x.dtype === 'string') {
      throw new Error('onesLike is not supported for string tensors');
    } else {
      // TODO(lina128): Use fill kernel directly once this kernel is
      // modularized.
      return tf.fill(x.shape, 1, x.dtype);
    }
  }

  zerosLike<R extends Rank>(x: Tensor<R>): Tensor<R> {
    const values = util.getArrayFromDType(
                       x.dtype, util.sizeFromShape(x.shape)) as TypedArray;
    return this.makeOutput(values, x.shape, x.dtype);
  }

  linspace(start: number, stop: number, num: number): Tensor1D {
    return backend_util.linspaceImpl(start, stop, num);
  }

  private scatter<R extends Rank>(
      indices: Tensor, updates: Tensor, shape: ShapeMap[R], outputSize: number,
      sliceSize: number, numUpdates: number, sliceRank: number,
      strides: number[], defaultValue: Scalar,
      sumDupeIndices: boolean): Tensor<R> {
    const flattenShape = [outputSize / sliceSize, sliceSize];

    const indicesData = this.readSync(indices.dataId) as TypedArray;
    const updatesData = this.readSync(updates.dataId) as TypedArray;

    if (outputSize === 0) {
      return tf.tensor([], shape, updates.dtype);
    }

    const buffer = new TensorBuffer(flattenShape, updates.dtype as 'float32');
    buffer.values.fill((this.readSync(defaultValue.dataId) as TypedArray)[0]);

    for (let i = 0; i < numUpdates; i++) {
      const index = [];
      let flattenIndex = 0;
      for (let j = 0; j < sliceRank; j++) {
        const dim = indicesData[i * sliceRank + j];
        index.push(dim);
        flattenIndex += dim * strides[j];
      }

      if (flattenIndex < 0 || flattenIndex >= outputSize / sliceSize) {
        throw new Error(
            `Invalid indices: ${index} does not index into ${shape}`);
      }

      for (let k = 0; k < sliceSize; k++) {
        if (sumDupeIndices) {
          buffer.values[flattenIndex * sliceSize + k] +=
              updatesData[i * sliceSize + k];
        } else {
          buffer.values[flattenIndex * sliceSize + k] = updates.rank === 0 ?
              updatesData[0] :
              updatesData[i * sliceSize + k];
        }
      }
    }
    return buffer.toTensor().reshape(shape);
  }
}
