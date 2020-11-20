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
import {backend_util, BackendTimingInfo, buffer, DataStorage, DataType, DataValues, engine, env, kernel_impls, KernelBackend, Rank, Scalar, ShapeMap, Tensor, Tensor1D, Tensor2D, Tensor4D, TensorBuffer, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

const nonMaxSuppressionV3Impl = kernel_impls.nonMaxSuppressionV3Impl;
const split = kernel_impls.split;
const whereImpl = kernel_impls.whereImpl;
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

  where(condition: Tensor): Tensor2D {
    assertNotComplex([condition], 'where');

    const condVals = this.readSync(condition.dataId) as TypedArray;
    return whereImpl(condition.shape, condVals);
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

  gather<T extends Tensor>(
      x: T, indices: Tensor1D, axis: number, batchDims = 0): T {
    assertNotComplex([x, indices], 'gather');
    const parsedAxis = util.parseAxisParam(axis, x.shape)[0];
    const shapeInfo = backend_util.segment_util.collectGatherOpShapeInfo(
        x, indices, parsedAxis, batchDims);

    const flattenX = x.reshape([
      shapeInfo.batchSize, shapeInfo.outerSize, shapeInfo.dimSize,
      shapeInfo.sliceSize
    ]);
    const flattenIndex = indices.reshape(
        [shapeInfo.batchSize, indices.size / shapeInfo.batchSize]);
    const flattenOutputShape = [
      shapeInfo.batchSize, shapeInfo.outerSize,
      indices.size / shapeInfo.batchSize, shapeInfo.sliceSize
    ];
    const indicesBuf = this.bufferSync(flattenIndex);
    const result = tf.buffer(flattenOutputShape, x.dtype);
    const xBuf = this.bufferSync(flattenX);

    for (let i = 0; i < result.size; ++i) {
      const newLoc = result.indexToLoc(i);

      const originalLoc: number[] = newLoc.slice();
      const batchIdx = originalLoc[0];
      const indicesIdx = originalLoc[2];
      const indicesIndex = indicesBuf.locToIndex([batchIdx, indicesIdx]);
      originalLoc[2] = indicesBuf.values[indicesIndex];

      const originalIndex = xBuf.locToIndex(originalLoc);
      result.values[i] = xBuf.values[originalIndex];
    }
    return result.toTensor().reshape(shapeInfo.outputShape);
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

  nonMaxSuppression(
      boxes: Tensor2D, scores: Tensor1D, maxOutputSize: number,
      iouThreshold: number, scoreThreshold: number): Tensor1D {
    assertNotComplex(boxes, 'nonMaxSuppression');

    const boxesVals = this.readSync(boxes.dataId) as TypedArray;
    const scoresVals = this.readSync(scores.dataId) as TypedArray;
    return nonMaxSuppressionV3Impl(
        boxesVals, scoresVals, maxOutputSize, iouThreshold, scoreThreshold);
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
