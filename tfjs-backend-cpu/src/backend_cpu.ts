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
      x: Tensor4D, newHeight: number, newWidth: number, alignCorners: boolean,
      halfPixelCenters: boolean): Tensor4D {
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
        const sourceFracRow = halfPixelCenters ?
            effectiveRowSizeRatio * (r + 0.5) :
            effectiveRowSizeRatio * r;
        let sourceNearestRow = Math.min(
            oldHeight - 1,
            alignCorners ? Math.round(sourceFracRow) :
                           Math.floor(sourceFracRow));
        if (halfPixelCenters) {
          sourceNearestRow = Math.max(0, sourceNearestRow);
        }
        const rowOffset = batchOffset + sourceNearestRow * x.strides[1];
        for (let c = 0; c < newWidth; c++) {
          const sourceFracCol = halfPixelCenters ?
              effectiveColSizeRatio * (c + 0.5) :
              effectiveColSizeRatio * c;
          let sourceNearestCol = Math.min(
              oldWidth - 1,
              alignCorners ? Math.round(sourceFracCol) :
                             Math.floor(sourceFracCol));
          if (halfPixelCenters) {
            sourceNearestCol = Math.max(0, sourceNearestCol);
          }
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
