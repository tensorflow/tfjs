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

import {ENGINE} from '../../engine';
import {ENV} from '../../environment';
import {warn} from '../../log';
import * as array_ops_util from '../../ops/array_ops_util';
import * as axis_util from '../../ops/axis_util';
import * as broadcast_util from '../../ops/broadcast_util';
import * as concat_util from '../../ops/concat_util';
import {Conv2DInfo, Conv3DInfo} from '../../ops/conv_util';
import * as erf_util from '../../ops/erf_util';
import {Activation} from '../../ops/fused_util';
import * as gather_nd_util from '../../ops/gather_nd_util';
import * as ops from '../../ops/ops';
import {buffer, scalar, tensor, tensor3d, tensor4d} from '../../ops/ops';
import * as scatter_nd_util from '../../ops/scatter_nd_util';
import * as selu_util from '../../ops/selu_util';
import {computeFlatOffset, getStridedSlicedInfo, isSliceContinous} from '../../ops/slice_util';
import {DataId, Scalar, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D, Tensor5D, TensorBuffer} from '../../tensor';
import {DataType, DataTypeMap, DataValues, NumericDataType, Rank, ShapeMap, TypedArray, upcastType} from '../../types';
import * as util from '../../util';
import {getArrayFromDType, inferDtype, now, sizeFromShape} from '../../util';
import {BackendTimingInfo, DataStorage, EPSILON_FLOAT32, KernelBackend} from '../backend';
import * as backend_util from '../backend_util';
import * as complex_util from '../complex_util';
import {nonMaxSuppressionImpl} from '../non_max_suppression_impl';
import {split} from '../split_shared';
import {topkImpl} from '../topk_impl';
import {whereImpl} from '../where_impl';

function mapActivation(
    backend: MathBackendCPU, activation: Activation, x: Tensor): Tensor {
  if (activation === 'linear') {
    return backend.linear(x);
  } else if (activation === 'relu') {
    return backend.relu(x);
  }
  throw new Error(
      `Activation ${activation} has not been implemented for the CPU backend.`);
}

interface TensorData<D extends DataType> {
  values?: DataTypeMap[D];
  dtype: D;
  // For complex numbers, the real and imaginary parts are stored as their own
  // individual tensors, with a parent joining the two with the
  // complexTensors field. When this is defined, texture will be null.
  complexTensors?: {real: Tensor, imag: Tensor};
}

export class MathBackendCPU implements KernelBackend {
  public blockSize = 48;

  private data: DataStorage<TensorData<DataType>>;
  private fromPixels2DContext: CanvasRenderingContext2D;
  private firstUse = true;

  constructor() {
    if (ENV.get('IS_BROWSER')) {
      this.fromPixels2DContext =
          document.createElement('canvas').getContext('2d');
    }
    this.data = new DataStorage(ENGINE);
  }

  register(dataId: DataId, shape: number[], dtype: DataType): void {
    if (this.firstUse) {
      this.firstUse = false;
      if (ENV.get('IS_NODE')) {
        warn(
            '\n============================\n' +
            'Hi there 👋. Looks like you are running TensorFlow.js in ' +
            'Node.js. To speed things up dramatically, install our node ' +
            'backend, which binds to TensorFlow C++, by running ' +
            'npm i @tensorflow/tfjs-node, ' +
            'or npm i @tensorflow/tfjs-node-gpu if you have CUDA. ' +
            'Then call require(\'@tensorflow/tfjs-node\'); (-gpu ' +
            'suffix for CUDA) at the start of your program. ' +
            'Visit https://github.com/tensorflow/tfjs-node for more details.' +
            '\n============================\n');
      }
    }
    if (this.data.has(dataId)) {
      throw new Error(`Data buffer is already registered`);
    }
    this.data.set(dataId, {dtype});
  }
  write(dataId: DataId, values: DataValues): void {
    if (values == null) {
      throw new Error('MathBackendCPU.write(): values can not be null');
    }
    this.data.get(dataId).values = values;
  }
  fromPixels(
      pixels: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      numChannels: number): Tensor3D {
    if (pixels == null) {
      throw new Error(
          'pixels passed to tf.browser.fromPixels() can not be null');
    }
    let vals: Uint8ClampedArray;
    // tslint:disable-next-line:no-any
    if (ENV.get('IS_NODE') && (pixels as any).getContext == null) {
      throw new Error(
          'When running in node, pixels must be an HTMLCanvasElement ' +
          'like the one returned by the `canvas` npm package');
    }
    // tslint:disable-next-line:no-any
    if ((pixels as any).getContext != null) {
      // tslint:disable-next-line:no-any
      vals = (pixels as any)
                 .getContext('2d')
                 .getImageData(0, 0, pixels.width, pixels.height)
                 .data;
    } else if (pixels instanceof ImageData) {
      vals = pixels.data;
    } else if (
        pixels instanceof HTMLImageElement ||
        pixels instanceof HTMLVideoElement) {
      if (this.fromPixels2DContext == null) {
        throw new Error(
            'Can\'t read pixels from HTMLImageElement outside ' +
            'the browser.');
      }
      this.fromPixels2DContext.canvas.width = pixels.width;
      this.fromPixels2DContext.canvas.height = pixels.height;
      this.fromPixels2DContext.drawImage(
          pixels, 0, 0, pixels.width, pixels.height);
      vals = this.fromPixels2DContext
                 .getImageData(0, 0, pixels.width, pixels.height)
                 .data;
    } else {
      throw new Error(
          'pixels passed to tf.browser.fromPixels() must be either an ' +
          `HTMLVideoElement, HTMLImageElement, HTMLCanvasElement or ` +
          `ImageData, but was ${(pixels as {}).constructor.name}`);
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
  async read(dataId: DataId): Promise<DataValues> {
    return this.readSync(dataId);
  }
  readSync(dataId: DataId): DataValues {
    const {dtype, complexTensors} = this.data.get(dataId);
    if (dtype === 'complex64') {
      const realValues = complexTensors.real.dataSync() as Float32Array;
      const imagValues = complexTensors.imag.dataSync() as Float32Array;
      return complex_util.mergeRealAndImagArrays(realValues, imagValues);
    }
    return this.data.get(dataId).values;
  }

  disposeData(dataId: DataId): void {
    if (this.data.has(dataId)) {
      const {complexTensors} = this.data.get(dataId);
      if (complexTensors != null) {
        complexTensors.real.dispose();
        complexTensors.imag.dispose();
      }
      this.data.delete(dataId);
    }
  }

  async time(f: () => void): Promise<BackendTimingInfo> {
    const start = now();
    f();
    const kernelMs = now() - start;
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

  complex<T extends Tensor>(real: T, imag: T): T {
    const result = Tensor.make(real.shape, {}, 'complex64') as T;

    const resultData = this.data.get(result.dataId);
    // The backend owns the reference to the underlying real and imaginary
    // clones. These will explicitly get disposed when the complex tensor is
    // disposed.
    resultData.complexTensors = {
      real: ENGINE.keep(real.clone()),
      imag: ENGINE.keep(imag.clone())
    };

    return result;
  }
  real<T extends Tensor>(input: T): T {
    const resultData = this.data.get(input.dataId);
    return resultData.complexTensors.real.clone() as T;
  }
  imag<T extends Tensor>(input: T): T {
    const resultData = this.data.get(input.dataId);
    return resultData.complexTensors.imag.clone() as T;
  }

  private assertNotComplex(tensor: Tensor|Tensor[], opName: string) {
    if (!Array.isArray(tensor)) {
      tensor = [tensor];
    }
    tensor.forEach(t => {
      if (t != null) {
        util.assert(
            t.dtype !== 'complex64',
            () => `${opName} does not support complex64 tensors.`);
      }
    });
  }

  slice<T extends Tensor>(x: T, begin: number[], size: number[]): T {
    this.assertNotComplex(x, 'slice');

    const isContinous = isSliceContinous(x.shape, begin, size);
    if (isContinous) {
      const flatOffset = computeFlatOffset(begin, x.strides);
      const length = util.sizeFromShape(size);
      const vals = x.dataSync();
      return tensor(
                 vals.subarray(flatOffset, flatOffset + length), size,
                 x.dtype) as T;
    }

    const buffer = ops.buffer(size, x.dtype);
    const xBuf = x.bufferSync();
    for (let i = 0; i < buffer.size; ++i) {
      const loc = buffer.indexToLoc(i);
      const xLoc = loc.map((idx, j) => idx + begin[j]);
      buffer.values[i] = xBuf.get(...xLoc);
    }
    return buffer.toTensor() as T;
  }

  stridedSlice<T extends Tensor>(
      x: T, begin: number[], end: number[], strides: number[],
      beginMask: number, endMask: number, ellipsisMask: number,
      newAxisMask: number, shrinkAxisMask: number): T {
    this.assertNotComplex(x, 'stridedSlice');

    const [beginIndex, size, shrinkAxis] = getStridedSlicedInfo(
        x.shape, begin, end, strides, beginMask, endMask, ellipsisMask,
        newAxisMask, shrinkAxisMask);

    const shape = size.filter((v, index) => shrinkAxis.indexOf(index) === -1);

    if (shape.some(axis => axis === 0)) {
      return ops.tensor([], shape) as T;
    }

    const buffer = ops.buffer(size, x.dtype);
    const xBuf = x.bufferSync();
    for (let i = 0; i < buffer.size; i++) {
      const loc = buffer.indexToLoc(i);

      const newLoc: number[] = new Array(loc.length);
      for (let j = 0; j < newLoc.length; j++) {
        newLoc[j] = loc[j] * strides[j] + beginIndex[j];
      }
      buffer.set(xBuf.get(...newLoc), ...loc);
    }

    return buffer.toTensor().reshape(shape) as T;
  }

  unstack(x: Tensor, axis: number): Tensor[] {
    const num = x.shape[axis];
    const outShape: number[] = new Array(x.rank - 1);
    let outIndex = 0;
    for (let i = 0; i < x.rank; i++) {
      if (i !== axis) {
        outShape[outIndex++] = x.shape[i];
      }
    }

    const begin = new Array(x.rank).fill(0);
    const size = x.shape.slice();
    size[axis] = 1;
    const res = new Array(num);
    for (let i = 0; i < res.length; i++) {
      begin[axis] = i;
      res[i] = this.slice(x, begin, size).reshape(outShape);
    }
    return res;
  }

  reverse<T extends Tensor>(x: T, axis: number[]): T {
    this.assertNotComplex(x, 'reverse');

    const buffer = ops.buffer(x.shape, x.dtype);
    const xBuf = x.bufferSync();

    for (let i = 0; i < buffer.size; i++) {
      const outLoc = buffer.indexToLoc(i);
      const inLoc = outLoc.slice();
      axis.forEach(ax => inLoc[ax] = x.shape[ax] - 1 - inLoc[ax]);
      buffer.set(xBuf.get(...inLoc), ...outLoc);
    }

    return buffer.toTensor() as T;
  }

  concat(tensors: Tensor[], axis: number): Tensor {
    this.assertNotComplex(tensors, 'concat');
    const tensors2D = tensors.map(t => {
      const innerSize = util.sizeFromShape(t.shape.slice(axis));
      return t.as2D(-1, innerSize);
    });
    const outShape =
        concat_util.computeOutShape(tensors2D.map(t => t.shape), 1 /* axis */);
    const values =
        ops.buffer(outShape as [number, number], tensors[0].dtype as 'float32')
            .values;
    if (tensors2D[0].shape[0] === 1) {
      // Use built-in TypedArray.set() method for speed.
      let offset = 0;
      tensors2D.forEach(t => {
        values.set(t.dataSync(), offset);
        offset += t.size;
      });
    } else {
      let colOffset = 0;
      tensors2D.forEach(t => {
        const tVals = t.dataSync();
        let tIdx = 0;
        for (let row = 0; row < t.shape[0]; ++row) {
          const resIdx = row * outShape[1] + colOffset;
          for (let col = 0; col < t.shape[1]; ++col) {
            values[resIdx + col] = tVals[tIdx++];
          }
        }
        colOffset += t.shape[1];
      });
    }
    const finalOutShape =
        concat_util.computeOutShape(tensors.map(t => t.shape), axis);
    return tensor(values, finalOutShape, tensors[0].dtype);
  }

  neg<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'neg');

    return this.multiply(ops.scalar(-1), x) as T;
  }

  add(a: Tensor, b: Tensor): Tensor {
    if (a.dtype === 'complex64' || b.dtype === 'complex64') {
      return this.broadcastedBinaryComplexOp(
                 a.cast('complex64'), b.cast('complex64'),
                 (aReal, aImag, bReal, bImag) => {
                   return {real: aReal + bReal, imag: aImag + bImag};
                 }) as Tensor;
    }

    return this.broadcastedBinaryOp(
               a, b, upcastType(a.dtype, b.dtype),
               (aValue, bValue) => aValue + bValue) as Tensor;
  }

  addN<T extends Tensor>(tensors: T[]): T {
    this.assertNotComplex(tensors, 'addN');

    const vals = tensors.map(t => t.dataSync());
    const result = ops.buffer(tensors[0].shape, tensors[0].dtype as 'float32');
    const resultVals = result.values;
    for (let i = 0; i < tensors.length; i++) {
      const currVals = vals[i];
      for (let j = 0; j < resultVals.length; j++) {
        resultVals[j] += currVals[j];
      }
    }
    return result.toTensor() as T;
  }

  subtract(a: Tensor, b: Tensor): Tensor {
    if (a.dtype === 'complex64' || b.dtype === 'complex64') {
      return this.broadcastedBinaryComplexOp(
                 a.cast('complex64'), b.cast('complex64'),
                 (aReal, aImag, bReal, bImag) => {
                   return {real: aReal - bReal, imag: aImag - bImag};
                 }) as Tensor;
    }

    return this.broadcastedBinaryOp(
               a, b, upcastType(a.dtype, b.dtype),
               (aValue, bValue) => aValue - bValue) as Tensor;
  }

  pow<T extends Tensor>(a: T, b: Tensor): T {
    this.assertNotComplex([a, b], 'pow');

    return this.broadcastedBinaryOp(
               a, b, a.dtype, (aValue, bValue) => Math.pow(aValue, bValue)) as
        T;
  }

  batchMatMul(
      a: Tensor3D, b: Tensor3D, transposeA: boolean,
      transposeB: boolean): Tensor3D {
    this.assertNotComplex([a, b], 'matMul');

    const sharedDim = transposeA ? a.shape[1] : a.shape[2];
    const leftDim = transposeA ? a.shape[2] : a.shape[1];
    const rightDim = transposeB ? b.shape[1] : b.shape[2];
    const batchDim = a.shape[0];

    const aValues = a.dataSync();
    const bValues = b.dataSync();
    const [aBatch, aOuterStep, aInnerStep] = transposeA ?
        [a.strides[0], 1, a.strides[1]] :
        [a.strides[0], a.strides[1], 1];
    const [bInnerStep, bOuterStep, bBatch] = transposeB ?
        [1, b.strides[1], b.strides[0]] :
        [b.strides[1], 1, b.strides[0]];

    const size = leftDim * rightDim;
    const result = buffer([batchDim, leftDim, rightDim], a.dtype);
    const resVals = result.values as TypedArray;
    const blockSize = this.blockSize;

    for (let b = 0; b < batchDim; b++) {
      for (let i0 = 0; i0 < leftDim; i0 += blockSize) {
        for (let j0 = 0; j0 < rightDim; j0 += blockSize) {
          for (let k0 = 0; k0 < sharedDim; k0 += blockSize) {
            // for when blockSize doesn't evenly divide the input
            const iBlock = Math.min(i0 + blockSize, leftDim);
            const jBlock = Math.min(j0 + blockSize, rightDim);
            const kBlock = Math.min(k0 + blockSize, sharedDim);

            for (let i = i0; i < iBlock; i++) {
              for (let j = j0; j < jBlock; j++) {
                let sum = 0.0;

                for (let k = k0; k < kBlock; k++) {
                  sum += aValues[b * aBatch + i * aOuterStep + k * aInnerStep] *
                      bValues[k * bInnerStep + j * bOuterStep + b * bBatch];
                }
                resVals[b * size + (i * rightDim + j)] += sum;
              }
            }
          }
        }
      }
    }
    return result.toTensor() as Tensor3D;
  }

  fusedBatchMatMul(
      a: Tensor3D, b: Tensor3D, transposeA: boolean, transposeB: boolean,
      bias?: Tensor, activation?: Activation): Tensor3D {
    let result = this.batchMatMul(a, b, transposeA, transposeB);
    if (bias) {
      result = this.add(result, bias) as Tensor3D;
    }
    if (activation) {
      result = mapActivation(this, activation, result) as Tensor3D;
    }
    return result;
  }

  multiply(a: Tensor, b: Tensor): Tensor {
    if (a.dtype === 'complex64' || b.dtype === 'complex64') {
      return this.broadcastedBinaryComplexOp(
                 a.cast('complex64'), b.cast('complex64'),
                 (aReal, aImag, bReal, bImag) => {
                   return {
                     real: aReal * bReal - aImag * bImag,
                     imag: aReal * bImag + aImag * bReal
                   };
                 }) as Tensor;
    }

    return this.broadcastedBinaryOp(
               a, b, upcastType(a.dtype, b.dtype),
               (aValue, bValue) => aValue * bValue) as Tensor;
  }

  realDivide(a: Tensor, b: Tensor): Tensor {
    this.assertNotComplex([a, b], 'realDivide');

    const op = (a: number, b: number) => a / b;
    const outputDtype = 'float32';
    return this.broadcastedBinaryOp(a, b, outputDtype, op) as Tensor;
  }

  floorDiv(a: Tensor, b: Tensor): Tensor {
    this.assertNotComplex([a, b], 'floorDiv');

    const op = (a: number, b: number) => Math.floor(a / b);
    const outputDtype = 'int32';
    return this.broadcastedBinaryOp(a, b, outputDtype, op) as Tensor;
  }

  sum(x: Tensor, axes: number[]): Tensor {
    this.assertNotComplex(x, 'sum');

    axis_util.assertAxesAreInnerMostDims('sum', axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const resultDtype = upcastType(x.dtype, 'int32');
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

  prod(x: Tensor, axes: number[]): Tensor {
    this.assertNotComplex(x, 'sum');

    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const resultDtype = upcastType(x.dtype, 'int32');
    const result = ops.zeros(outShape, resultDtype);
    const reduceSize = util.sizeFromShape(reduceShape);
    const vals = result.dataSync();

    const aVals = x.dataSync();
    for (let i = 0; i < vals.length; ++i) {
      const offset = i * reduceSize;
      let prod = 1;
      for (let j = 0; j < reduceSize; ++j) {
        prod *= aVals[offset + j];
      }
      vals[i] = prod;
    }
    return result;
  }

  unsortedSegmentSum<T extends Tensor>(
      x: T, segmentIds: Tensor1D, numSegments: number): Tensor {
    this.assertNotComplex(x, 'unsortedSegmentSum');

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
    this.assertNotComplex(x, 'argMin');

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
    this.assertNotComplex(x, 'argMax');

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
    this.assertNotComplex(x, 'cumsum');

    if (axis !== x.rank - 1) {
      throw new Error(
          `backend.cumsum in CPU expects an inner-most axis=${x.rank - 1} ` +
          `but got axis=${axis}`);
    }
    const resultDtype = upcastType(x.dtype, 'int32');
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
    this.assertNotComplex([a, b], 'equal');

    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      return (aVal === bVal) ? 1 : 0;
    });
  }

  notEqual(a: Tensor, b: Tensor): Tensor {
    this.assertNotComplex([a, b], 'notEqual');

    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      return (aVal !== bVal) ? 1 : 0;
    });
  }

  less(a: Tensor, b: Tensor): Tensor {
    this.assertNotComplex([a, b], 'less');

    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      return (aVal < bVal) ? 1 : 0;
    });
  }

  lessEqual(a: Tensor, b: Tensor): Tensor {
    this.assertNotComplex([a, b], 'lessEqual');

    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      return (aVal <= bVal) ? 1 : 0;
    });
  }

  greater(a: Tensor, b: Tensor): Tensor {
    this.assertNotComplex([a, b], 'greater');

    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      return (aVal > bVal) ? 1 : 0;
    });
  }

  greaterEqual(a: Tensor, b: Tensor): Tensor {
    this.assertNotComplex([a, b], 'greaterEqual');

    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      return (aVal >= bVal) ? 1 : 0;
    });
  }

  logicalNot<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'logicalNot');

    const values = x.dataSync();
    const newValues = new Uint8Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      newValues[i] = values[i] ? 0 : 1;
    }
    return Tensor.make(x.shape, {values: newValues}, 'bool') as T;
  }

  logicalAnd(a: Tensor, b: Tensor): Tensor {
    this.assertNotComplex([a, b], 'logicalAnd');

    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      return aVal && bVal;
    });
  }

  logicalOr(a: Tensor, b: Tensor): Tensor {
    this.assertNotComplex([a, b], 'logicalOr');

    return this.broadcastedBinaryOp(a, b, 'bool', (aVal, bVal) => {
      return aVal || bVal;
    });
  }

  select(condition: Tensor, a: Tensor, b: Tensor): Tensor {
    this.assertNotComplex([condition, a, b], 'select');

    const values = condition.dataSync();
    const aValues = a.dataSync();
    const bValues = b.dataSync();
    const result = ops.zeros(a.shape, upcastType(a.dtype, b.dtype));
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

  where(condition: Tensor): Tensor2D {
    this.assertNotComplex([condition], 'where');

    const condVals = condition.dataSync();
    return whereImpl(condition.shape, condVals);
  }

  topk<T extends Tensor>(x: T, k: number, sorted: boolean): [T, T] {
    this.assertNotComplex(x, 'topk');

    const xVals = x.dataSync();
    return topkImpl(xVals, x.shape, x.dtype as NumericDataType, k, sorted);
  }

  min(x: Tensor, axes: number[]): Tensor {
    this.assertNotComplex(x, 'min');

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
    this.assertNotComplex([a, b], 'minimum');

    return this.broadcastedBinaryOp(
        a, b, a.dtype, (aVal, bVal) => Math.min(aVal, bVal));
  }

  mod(a: Tensor, b: Tensor): Tensor {
    this.assertNotComplex([a, b], 'mod');

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
    this.assertNotComplex(x, 'max');

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
    this.assertNotComplex([a, b], 'maximum');

    return this.broadcastedBinaryOp(
        a, b, a.dtype, (aVal, bVal) => Math.max(aVal, bVal));
  }

  all(x: Tensor, axes: number[]): Tensor {
    this.assertNotComplex(x, 'all');

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

  any(x: Tensor, axes: number[]): Tensor {
    this.assertNotComplex(x, 'any');

    axis_util.assertAxesAreInnerMostDims('any', axes, x.rank);
    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const result = ops.zeros(outShape, x.dtype);
    const reduceSize = util.sizeFromShape(reduceShape);
    const vals = result.dataSync();

    const aVals = x.dataSync();
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
    this.assertNotComplex([a, b], 'squaredDifference');

    return this.broadcastedBinaryOp(a, b, a.dtype, (aVal, bVal) => {
      const diff = aVal - bVal;
      return diff * diff;
    });
  }

  ceil<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'ceil');

    const values = x.dataSync();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      newValues[i] = Math.ceil(values[i]);
    }
    return Tensor.make(x.shape, {values: newValues}) as T;
  }

  floor<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'floor');

    const values = x.dataSync();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      newValues[i] = Math.floor(values[i]);
    }
    return Tensor.make(x.shape, {values: newValues}) as T;
  }

  sign<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'x');

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

  isNaN<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'x');

    const values = x.dataSync();
    const newValues = new Uint8Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      if (Number.isNaN(values[i])) {
        newValues[i] = 1;
      }
    }
    return Tensor.make(x.shape, {values: newValues}, 'bool') as T;
  }

  isInf<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'x');

    const values = x.dataSync();
    const newValues = new Uint8Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      if (Math.abs(values[i]) === Infinity) {
        newValues[i] = 1;
      }
    }
    return Tensor.make(x.shape, {values: newValues}, 'bool') as T;
  }

  isFinite<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'x');

    const values = x.dataSync();
    const newValues = new Uint8Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      if (Number.isFinite(values[i])) {
        newValues[i] = 1;
      }
    }
    return Tensor.make(x.shape, {values: newValues}, 'bool') as T;
  }

  round<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'round');

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
    this.assertNotComplex(x, 'exp');

    const values = x.dataSync();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      newValues[i] = Math.exp(values[i]);
    }
    return Tensor.make(x.shape, {values: newValues}) as T;
  }

  expm1<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'expm1');

    const values = x.dataSync();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      newValues[i] = Math.expm1(values[i]);
    }
    return Tensor.make(x.shape, {values: newValues}) as T;
  }

  log<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'log');

    const values = x.dataSync();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      const value = values[i];
      newValues[i] = Math.log(value);
    }
    return Tensor.make(x.shape, {values: newValues}) as T;
  }

  log1p<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'log1p');

    const values = x.dataSync();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      const value = values[i];
      newValues[i] = Math.log1p(value);
    }
    return Tensor.make(x.shape, {values: newValues}) as T;
  }

  sqrt<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'sqrt');

    const values = x.dataSync();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      const value = values[i];
      newValues[i] = Math.sqrt(value);
    }
    return Tensor.make(x.shape, {values: newValues}) as T;
  }

  rsqrt<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'rsqrt');

    const values = x.dataSync();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      const value = values[i];
      newValues[i] = 1 / Math.sqrt(value);
    }
    return Tensor.make(x.shape, {values: newValues}) as T;
  }

  square<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'square');

    const values = x.dataSync();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      const value = values[i];
      newValues[i] = value * value;
    }
    return Tensor.make(x.shape, {values: newValues}) as T;
  }

  reciprocal<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'reciprocal');

    const values = x.dataSync();
    const newValues = new Float32Array(values.length);
    for (let i = 0; i < values.length; ++i) {
      newValues[i] = 1 / values[i];
    }
    return Tensor.make(x.shape, {values: newValues}) as T;
  }

  linear<T extends Tensor>(x: T): T {
    return x;
  }

  relu<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'relu');

    const res = ops.zeros(x.shape, x.dtype);
    const resVals = res.dataSync();
    const inVals = x.dataSync();
    for (let i = 0; i < inVals.length; ++i) {
      resVals[i] = Math.max(0, inVals[i]);
    }
    return res as T;
  }

  prelu<T extends Tensor>(x: T, a: T): T {
    this.assertNotComplex([x, a], 'prelu');

    return this.broadcastedBinaryOp(
               x, a, x.dtype,
               (xValue, aValue) => xValue < 0 ? aValue * xValue : xValue) as T;
  }

  elu<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'elu');

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
    this.assertNotComplex([dy, y], 'eluDer');

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
    this.assertNotComplex(x, 'selu');

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
    this.assertNotComplex(x, 'clip');

    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      const v = values[i];
      resultValues[i] = v > max ? max : (v < min ? min : v);
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

  complexAbs<T extends Tensor>(x: T): T {
    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();

    for (let i = 0; i < x.size; ++i) {
      const real = values[i * 2];
      const imag = values[i * 2 + 1];
      resultValues[i] = Math.hypot(real, imag);
    }
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  int<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'int');

    const resultValues = new Int32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = values[i];
    }
    return Tensor.make(x.shape, {values: resultValues}, 'int32');
  }

  sigmoid<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'sigmoid');

    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = 1 / (1 + Math.exp(-values[i]));
    }
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  softplus<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'softplus');

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
    this.assertNotComplex(x, 'sin');

    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.sin(values[i]);
    }
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  cos<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'cos');

    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.cos(values[i]);
    }
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  tan<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'tan');

    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.tan(values[i]);
    }
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  asin<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'asin');

    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.asin(values[i]);
    }
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  acos<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'acos');

    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.acos(values[i]);
    }
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  atan<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'atan');

    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.atan(values[i]);
    }
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  atan2<T extends Tensor>(a: T, b: T): T {
    this.assertNotComplex([a, b], 'atan2');

    return this.broadcastedBinaryOp(
               a, b, a.dtype, (aValue, bValue) => Math.atan2(aValue, bValue)) as
        T;
  }

  sinh<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'sinh');

    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.sinh(values[i]);
    }
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  cosh<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'cosh');

    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.cosh(values[i]);
    }
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  tanh<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'tanh');

    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = util.tanh(values[i]);
    }
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  asinh<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'asinh');

    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.asinh(values[i]);
    }
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  acosh<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'acosh');

    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.acosh(values[i]);
    }
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  atanh<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'atanh');

    const resultValues = new Float32Array(x.size);
    const values = x.dataSync();
    for (let i = 0; i < values.length; ++i) {
      resultValues[i] = Math.atanh(values[i]);
    }
    return Tensor.make(x.shape, {values: resultValues}) as T;
  }

  erf<T extends Tensor>(x: T): T {
    this.assertNotComplex(x, 'erf');

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
    this.assertNotComplex(x, 'step');

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
    this.assertNotComplex([x, filter], 'conv2d');

    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const padLeft = convInfo.padInfo.left;
    const padTop = convInfo.padInfo.top;
    const y = ops.buffer(convInfo.outShape, x.dtype as 'float32');

    const xVals = x.dataSync();
    const wVals = filter.dataSync();
    const yVals = y.values;

    for (let b = 0; b < convInfo.batchSize; ++b) {
      const xOffset1 = b * x.strides[0];
      const yOffset1 = b * y.strides[0];
      for (let yR = 0; yR < convInfo.outHeight; ++yR) {
        const yOffset2 = yOffset1 + yR * y.strides[1];
        const xRCorner = yR * convInfo.strideHeight - padLeft;
        for (let wR = 0; wR < filterHeight; wR++) {
          const xR = xRCorner + wR * dilationHeight;
          if (xR < 0 || xR >= convInfo.inHeight) {
            continue;
          }
          const wOffset1 = wR * filter.strides[0];
          const xOffset2 = xOffset1 + xR * x.strides[1];
          for (let yC = 0; yC < convInfo.outWidth; ++yC) {
            const yOffset3 = yOffset2 + yC * convInfo.outChannels;
            const xCCorner = yC * convInfo.strideWidth - padTop;
            for (let wC = 0; wC < filterWidth; wC++) {
              const xC = xCCorner + wC * dilationWidth;
              if (xC < 0 || xC >= convInfo.inWidth) {
                continue;
              }
              const wOffset2 = wOffset1 + wC * filter.strides[1];
              const xOffset3 = xOffset2 + xC * convInfo.inChannels;
              let wOffset3 = wOffset2;
              for (let d1 = 0; d1 < convInfo.inChannels; ++d1) {
                const xVal = xVals[xOffset3 + d1];
                for (let d2 = 0; d2 < convInfo.outChannels; ++d2) {
                  yVals[yOffset3 + d2] += xVal * wVals[wOffset3 + d2];
                }
                wOffset3 += convInfo.outChannels;
              }
            }
          }
        }
      }
    }
    return y.toTensor() as Tensor4D;
  }

  conv3d(x: Tensor5D, filter: Tensor5D, convInfo: Conv3DInfo): Tensor5D {
    const filterDepth = convInfo.filterDepth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const dilationDepth = convInfo.dilationDepth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const padFront = convInfo.padInfo.front;
    const padLeft = convInfo.padInfo.left;
    const padTop = convInfo.padInfo.top;
    const y = ops.buffer<Rank.R5>(convInfo.outShape, x.dtype as 'float32');

    const xVals = x.dataSync();
    const wVals = filter.dataSync();
    const yVals = y.values;

    for (let b = 0; b < convInfo.batchSize; ++b) {
      const xOffset1 = b * x.strides[0];
      const yOffset1 = b * y.strides[0];
      for (let yF = 0; yF < convInfo.outDepth; ++yF) {
        const yOffset2 = yOffset1 + yF * y.strides[1];
        const xFCorner = yF * convInfo.strideDepth - padFront;
        for (let wF = 0; wF < filterDepth; wF++) {
          const xF = xFCorner + wF * dilationDepth;
          if (xF < 0 || xF >= convInfo.inDepth) {
            continue;
          }
          const wOffset1 = wF * filter.strides[0];
          const xOffset2 = xOffset1 + xF * x.strides[1];

          for (let yR = 0; yR < convInfo.outHeight; ++yR) {
            const yOffset3 = yOffset2 + yR * y.strides[2];
            const xRCorner = yR * convInfo.strideHeight - padTop;
            for (let wR = 0; wR < filterHeight; wR++) {
              const xR = xRCorner + wR * dilationHeight;
              if (xR < 0 || xR >= convInfo.inHeight) {
                continue;
              }
              const wOffset2 = wOffset1 + wR * filter.strides[1];
              const xOffset3 = xOffset2 + xR * x.strides[2];
              for (let yC = 0; yC < convInfo.outWidth; ++yC) {
                const yOffset4 = yOffset3 + yC * convInfo.outChannels;
                const xCCorner = yC * convInfo.strideWidth - padLeft;
                for (let wC = 0; wC < filterWidth; wC++) {
                  const xC = xCCorner + wC * dilationWidth;
                  if (xC < 0 || xC >= convInfo.inWidth) {
                    continue;
                  }
                  const wOffset3 = wOffset2 + wC * filter.strides[2];
                  const xOffset4 = xOffset3 + xC * convInfo.inChannels;
                  let wOffset4 = wOffset3;
                  for (let d1 = 0; d1 < convInfo.inChannels; ++d1) {
                    const xVal = xVals[xOffset4 + d1];
                    for (let d2 = 0; d2 < convInfo.outChannels; ++d2) {
                      yVals[yOffset4 + d2] += xVal * wVals[wOffset4 + d2];
                    }
                    wOffset4 += convInfo.outChannels;
                  }
                }
              }
            }
          }
        }
      }
    }
    return y.toTensor();
  }

  conv2dDerInput(dy: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    this.assertNotComplex([dy, filter], 'conv2dDerInput');

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

  conv3dDerInput(dy: Tensor5D, filter: Tensor5D, convInfo: Conv3DInfo):
      Tensor5D {
    const dx = ops.buffer<Rank.R5>(convInfo.inShape, 'float32');
    const dxValues = dx.values;
    const [dxS0, dxS1, dxS2, dxS3] = dx.strides;
    const dyValues = dy.dataSync();
    const [dyS0, dyS1, dyS2, dyS3] = dy.strides;
    const fltValues = filter.dataSync();
    const [fltS0, fltS1, fltS2, fltS3] = filter.strides;
    const {
      batchSize,
      filterDepth,
      filterHeight,
      filterWidth,
      inChannels,
      inDepth,
      inHeight,
      inWidth,
      outChannels,
      outDepth,
      outHeight,
      outWidth,
      strideDepth,
      strideHeight,
      strideWidth
    } = convInfo;
    const frontPad = filterDepth - 1 - convInfo.padInfo.front;
    const topPad = filterHeight - 1 - convInfo.padInfo.top;
    const leftPad = filterWidth - 1 - convInfo.padInfo.left;

    for (let b = 0; b < batchSize; ++b) {
      for (let d1 = 0; d1 < inChannels; ++d1) {
        // Frames of depth
        for (let xF = 0; xF < inDepth; ++xF) {
          const xFCorner = xF - frontPad;
          const xFMin = Math.max(0, Math.ceil(xFCorner / strideDepth));
          const yFMax =
              Math.min(outDepth, (filterDepth + xFCorner) / strideDepth);

          // Rows as per standard 2d matrix notation
          for (let xR = 0; xR < inHeight; ++xR) {
            const xRCorner = xR - topPad;
            const xRMin = Math.max(0, Math.ceil(xRCorner / strideHeight));
            const yRMax =
                Math.min(outHeight, (filterHeight + xRCorner) / strideHeight);
            // Columns as per standard 2d matrix notation
            for (let xC = 0; xC < inWidth; ++xC) {
              const xCCorner = xC - leftPad;
              const xCMin = Math.max(0, Math.ceil(xCCorner / strideWidth));
              const yCMax =
                  Math.min(outWidth, (filterWidth + xCCorner) / strideWidth);

              let dotProd = 0;
              for (let yF = xFMin; yF < yFMax; ++yF) {
                const wF = yF * strideDepth - xFCorner;

                for (let yR = xRMin; yR < yRMax; ++yR) {
                  const wR = yR * strideHeight - xRCorner;

                  for (let yC = xCMin; yC < yCMax; ++yC) {
                    const wC = yC * strideWidth - xCCorner;
                    const dyOffset =
                        dyS0 * b + dyS1 * yF + dyS2 * yR + dyS3 * yC;
                    const fltOffset = fltS0 * (filterDepth - 1 - wF) +
                        fltS1 * (filterHeight - 1 - wR) +
                        fltS2 * (filterWidth - 1 - wC) + fltS3 * d1;

                    for (let d2 = 0; d2 < outChannels; ++d2) {
                      const pixel = dyValues[dyOffset + d2];
                      const weight = fltValues[fltOffset + d2];
                      dotProd += pixel * weight;
                    }
                  }
                }
              }
              dxValues[dxS0 * b + dxS1 * xF + dxS2 * xR + dxS3 * xC + d1] =
                  dotProd;
            }
          }
        }
      }
    }
    return dx.toTensor();
  }

  conv2dDerFilter(x: Tensor4D, dy: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    this.assertNotComplex([x, dy], 'conv2dDerFilter');

    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const dW = ops.buffer<Rank.R4>(convInfo.filterShape, 'float32');

    const leftPad = convInfo.padInfo.left;
    const topPad = convInfo.padInfo.top;
    const xBuf = x.bufferSync();
    const dyBuf = dy.bufferSync();
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
                  dotProd += xBuf.get(b, xR, xC, d1) * dyBuf.get(b, yR, yC, d2);
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

  conv3dDerFilter(x: Tensor5D, dy: Tensor5D, convInfo: Conv3DInfo): Tensor5D {
    const strideDepth = convInfo.strideDepth;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const filterDepth = convInfo.filterDepth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;

    const dw = ops.buffer<Rank.R5>(convInfo.filterShape, 'float32');
    const dwValues = dw.values;
    const [dwS0, dwS1, dwS2, dwS3] = dw.strides;
    const dyValues = dy.dataSync();
    const [dyS0, dyS1, dyS2, dyS3] = dy.strides;
    const xValues = x.dataSync();
    const [xS0, xS1, xS2, xS3] = x.strides;

    const frontPad = convInfo.padInfo.front;
    const leftPad = convInfo.padInfo.left;
    const topPad = convInfo.padInfo.top;

    for (let wF = 0; wF < filterDepth; ++wF) {
      const yFMin = Math.max(0, Math.ceil((frontPad - wF) / strideDepth));
      const yFMax = Math.min(
          convInfo.outDepth, (convInfo.inDepth + frontPad - wF) / strideDepth);
      const wOffset1 = wF * dwS0;

      for (let wR = 0; wR < filterHeight; ++wR) {
        const yRMin = Math.max(0, Math.ceil((topPad - wR) / strideHeight));
        const yRMax = Math.min(
            convInfo.outHeight,
            (convInfo.inHeight + topPad - wR) / strideHeight);
        const wOffset2 = wR * dwS1 + wOffset1;

        for (let wC = 0; wC < filterWidth; ++wC) {
          const yCMin = Math.max(0, Math.ceil((leftPad - wC) / strideWidth));
          const yCMax = Math.min(
              convInfo.outWidth,
              (convInfo.inWidth + leftPad - wC) / strideWidth);
          const wOffset3 = wC * dwS2 + wOffset2;

          for (let d1 = 0; d1 < convInfo.inChannels; ++d1) {
            const wOffset4 = d1 * dwS3 + wOffset3;

            for (let d2 = 0; d2 < convInfo.outChannels; ++d2) {
              let dotProd = 0;
              for (let b = 0; b < convInfo.batchSize; ++b) {
                const xOffset1 = b * xS0;
                const yOffset1 = b * dyS0;

                for (let yF = yFMin; yF < yFMax; ++yF) {
                  const xF = wF + yF * strideDepth - frontPad;
                  const xOffset2 = xF * xS1 + xOffset1;
                  const yOffset2 = yF * dyS1 + yOffset1;

                  for (let yR = yRMin; yR < yRMax; ++yR) {
                    const xR = wR + yR * strideHeight - topPad;
                    const xOffset3 = xR * xS2 + xOffset2;
                    const yOffset3 = yR * dyS2 + yOffset2;

                    for (let yC = yCMin; yC < yCMax; ++yC) {
                      const xC = wC + yC * strideWidth - leftPad;
                      const xOffset4 = xC * xS3 + xOffset3;
                      const yOffset4 = yC * dyS3 + yOffset3;

                      dotProd +=
                          xValues[xOffset4 + d1] * dyValues[yOffset4 + d2];
                    }
                  }
                }
              }
              dwValues[wOffset4 + d2] = dotProd;
            }
          }
        }
      }
    }
    return dw.toTensor();
  }

  depthwiseConv2D(x: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    this.assertNotComplex([x, filter], 'depthwiseConv2D');

    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const padLeft = convInfo.padInfo.left;
    const padTop = convInfo.padInfo.top;
    const chMul = convInfo.outChannels / convInfo.inChannels;
    const y = ops.buffer(convInfo.outShape, x.dtype as 'float32');
    const xVals = x.dataSync();
    const wVals = filter.dataSync();
    const yVals = y.values;

    for (let b = 0; b < convInfo.batchSize; ++b) {
      const xOffset1 = b * x.strides[0];
      const yOffset1 = b * y.strides[0];
      for (let yR = 0; yR < convInfo.outHeight; ++yR) {
        const yOffset2 = yOffset1 + yR * y.strides[1];
        const xRCorner = yR * convInfo.strideHeight - padLeft;
        for (let wR = 0; wR < filterHeight; ++wR) {
          const xR = xRCorner + wR * dilationHeight;
          if (xR < 0 || xR >= convInfo.inHeight) {
            continue;
          }
          const wOffset1 = wR * filter.strides[0];
          const xOffset2 = xOffset1 + xR * x.strides[1];
          for (let yC = 0; yC < convInfo.outWidth; ++yC) {
            const yOffset3 = yOffset2 + yC * y.strides[2];
            const xCCorner = yC * convInfo.strideWidth - padTop;
            for (let wC = 0; wC < filterWidth; ++wC) {
              const xC = xCCorner + wC * dilationWidth;
              if (xC < 0 || xC >= convInfo.inWidth) {
                continue;
              }
              const wOffset2 = wOffset1 + wC * filter.strides[1];
              const xOffset3 = xOffset2 + xC * convInfo.inChannels;
              let yOffset4 = yOffset3;
              let wOffset3 = wOffset2;
              for (let d1 = 0; d1 < convInfo.inChannels; ++d1) {
                const xVal = xVals[xOffset3 + d1];
                for (let q = 0; q < chMul; ++q) {
                  yVals[yOffset4 + q] += xVal * wVals[wOffset3 + q];
                }
                yOffset4 += chMul;
                wOffset3 += chMul;
              }
            }
          }
        }
      }
    }

    return y.toTensor() as Tensor4D;
  }

  depthwiseConv2DDerInput(dy: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    this.assertNotComplex([dy, filter], 'depthwiseConv2DDerInput');

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
    this.assertNotComplex([x, dy], 'depthwiseConv2DDerFilter');

    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const dW = ops.buffer<Rank.R4>(convInfo.filterShape, 'float32');

    const leftPad = convInfo.padInfo.left;
    const topPad = convInfo.padInfo.top;
    const chMul = convInfo.outChannels / convInfo.inChannels;

    const xBuf = x.bufferSync();
    const dyBuf = dy.bufferSync();
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
                dotProd += xBuf.get(b, xR, xC, d1) * dyBuf.get(b, yR, yC, d2);
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
    this.assertNotComplex(x, 'tile');

    const newShape: number[] = new Array(x.rank);
    for (let i = 0; i < newShape.length; i++) {
      newShape[i] = x.shape[i] * reps[i];
    }
    const result = ops.buffer(newShape, x.dtype);
    const xBuf = x.bufferSync();
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
    this.assertNotComplex(x, 'pad');

    const outShape = paddings.map(
        (p, i) => p[0] /* beforePad */ + x.shape[i] + p[1] /* afterPad */);
    const start = paddings.map(p => p[0]);
    const xBuffer = x.bufferSync();
    const buffer = ops.buffer(outShape, x.dtype as 'float32');
    if (constantValue !== 0) {
      buffer.values.fill(constantValue);
    }

    for (let i = 0; i < x.size; i++) {
      const coords = xBuffer.indexToLoc(i);
      const outCoords = coords.map((c, i) => c + start[i]);
      buffer.set(xBuffer.get(...coords), ...outCoords);
    }
    return buffer.toTensor() as T;
  }

  transpose<T extends Tensor>(x: T, perm: number[]): T {
    this.assertNotComplex(x, 'transpose');

    const newShape: number[] = new Array(x.rank);
    for (let i = 0; i < newShape.length; i++) {
      newShape[i] = x.shape[perm[i]];
    }
    const values = x.dataSync();
    const result = buffer(newShape, x.dtype);

    const xBuf = x.bufferSync();
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
    this.assertNotComplex([x, indices], 'gather');

    const newShape: number[] = x.shape.slice();
    const indicesValues = indices.dataSync();
    newShape[axis] = indicesValues.length;
    const result = buffer(newShape, x.dtype);
    const xBuf = x.bufferSync();

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
    this.assertNotComplex([x], 'batchToSpaceND');

    const prod = blockShape.reduce((a, b) => a * b);

    const reshaped = array_ops_util.getReshaped(x.shape, blockShape, prod);
    const permuted =
        array_ops_util.getPermuted(reshaped.length, blockShape.length);
    const reshapedPermuted =
        array_ops_util.getReshapedPermuted(x.shape, blockShape, prod);
    const sliceBeginCoords =
        array_ops_util.getSliceBeginCoords(crops, blockShape.length);
    const sliceSize =
        array_ops_util.getSliceSize(reshapedPermuted, crops, blockShape.length);

    return x.reshape(reshaped)
               .transpose(permuted)
               .reshape(reshapedPermuted)
               .slice(sliceBeginCoords, sliceSize) as T;
  }

  spaceToBatchND<T extends Tensor>(
      x: T, blockShape: number[], paddings: Array<[number, number]>): T {
    this.assertNotComplex([x], 'spaceToBatchND');

    const prod = blockShape.reduce((a, b) => a * b);

    const completePaddings: Array<[number, number]> = [[0, 0]];
    completePaddings.push(...paddings);
    for (let i = 1 + blockShape.length; i < x.shape.length; ++i) {
      completePaddings.push([0, 0]);
    }

    const paddedX = x.pad(completePaddings);

    const reshapedPaddedShape =
        array_ops_util.getReshaped(paddedX.shape, blockShape, prod, false);
    const permutedReshapedPaddedPermutation = array_ops_util.getPermuted(
        reshapedPaddedShape.length, blockShape.length, false);
    const flattenShape = array_ops_util.getReshapedPermuted(
        paddedX.shape, blockShape, prod, false);

    return paddedX.reshape(reshapedPaddedShape)
               .transpose(permutedReshapedPaddedPermutation)
               .reshape(flattenShape) as T;
  }

  private pool(x: Tensor4D, convInfo: Conv2DInfo, poolType: 'max'|'avg'):
      Tensor4D {
    this.assertNotComplex(x, 'pool');

    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const effectiveFilterHeight = convInfo.effectiveFilterHeight;
    const effectiveFilterWidth = convInfo.effectiveFilterWidth;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;

    const initialValue =
        (poolType === 'max' ? Number.NEGATIVE_INFINITY :
                              Number.POSITIVE_INFINITY);

    const xValues = x.dataSync();
    const output = ops.buffer(convInfo.outShape, x.dtype);
    const outputVals = output.values;

    const outputBatchStrides =
        convInfo.outShape[1] * convInfo.outShape[2] * convInfo.outShape[3];
    const outputRowStrides = convInfo.outShape[2] * convInfo.outShape[3];
    const outputColStrides = convInfo.outShape[3];

    for (let b = 0; b < convInfo.batchSize; ++b) {
      const outputBatchOffset = b * outputBatchStrides;
      const inputBatchOffset = b * x.strides[0];
      for (let d = 0; d < convInfo.inChannels; ++d) {
        for (let yR = 0; yR < convInfo.outHeight; ++yR) {
          const xRCorner = yR * strideHeight - padTop;
          const xRMin = Math.max(0, xRCorner);
          const xRMax =
              Math.min(convInfo.inHeight, effectiveFilterHeight + xRCorner);
          const outputRowOffset = outputBatchOffset + yR * outputRowStrides;
          for (let yC = 0; yC < convInfo.outWidth; ++yC) {
            const xCCorner = yC * strideWidth - padLeft;
            const xCMin = Math.max(0, xCCorner);
            const xCMax =
                Math.min(convInfo.inWidth, effectiveFilterWidth + xCCorner);
            let minMaxValue = initialValue;
            let avgValue = 0;
            let count = 0;
            for (let xR = xRMin; xR < xRMax; xR += dilationHeight) {
              const xROffset = inputBatchOffset + xR * x.strides[1];
              for (let xC = xCMin; xC < xCMax; xC += dilationWidth) {
                const xCOffset = xROffset + xC * x.strides[2];
                const pixel = xValues[xCOffset + d];
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
            const outputOffset = outputRowOffset + yC * outputColStrides + d;
            outputVals[outputOffset] =
                poolType === 'avg' ? avgValue / count : minMaxValue;
          }
        }
      }
    }
    return output.toTensor() as Tensor4D;
  }

  maxPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    return this.pool(x, convInfo, 'max');
  }

  private maxPoolPositions(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    const maxPositions = ops.buffer(convInfo.outShape, 'int32');
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const effectiveFilterHeight = convInfo.effectiveFilterHeight;
    const effectiveFilterWidth = convInfo.effectiveFilterWidth;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;

    const xBuf = x.bufferSync();
    for (let b = 0; b < convInfo.batchSize; ++b) {
      for (let d = 0; d < convInfo.inChannels; ++d) {
        for (let yR = 0; yR < convInfo.outHeight; ++yR) {
          const xRCorner = yR * strideHeight - padTop;
          let xRMin = xRCorner;
          while (xRMin < 0) {
            xRMin += dilationHeight;
          }
          // const xRMin = Math.max(0, xRCorner);
          const xRMax =
              Math.min(convInfo.inHeight, effectiveFilterHeight + xRCorner);
          for (let yC = 0; yC < convInfo.outWidth; ++yC) {
            const xCCorner = yC * strideWidth - padLeft;
            let xCMin = xCCorner;
            while (xCMin < 0) {
              xCMin += dilationWidth;
            }
            const xCMax =
                Math.min(convInfo.inWidth, effectiveFilterWidth + xCCorner);
            let maxValue = Number.NEGATIVE_INFINITY;
            let maxPosition = -1;

            for (let xR = xRMin; xR < xRMax; xR += dilationHeight) {
              const wR = xR - xRCorner;
              for (let xC = xCMin; xC < xCMax; xC += dilationWidth) {
                const wC = xC - xCCorner;
                const pixel = xBuf.get(b, xR, xC, d);
                if (pixel > maxValue) {
                  maxValue = pixel;
                  maxPosition = wR * effectiveFilterWidth + wC;
                }
              }
            }
            maxPositions.set(maxPosition, b, yR, yC, d);
          }
        }
      }
    }
    return maxPositions.toTensor() as Tensor4D;
  }

  maxPoolBackprop(dy: Tensor4D, x: Tensor4D, y: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    this.assertNotComplex([x, y], 'maxPoolBackprop');

    const maxPositions = this.maxPoolPositions(x, convInfo);
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const effectiveFilterHeight = convInfo.effectiveFilterHeight;
    const effectiveFilterWidth = convInfo.effectiveFilterWidth;
    const padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;
    const padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
    const dx = ops.buffer<Rank.R4>(x.shape, 'float32');

    const maxPosBuf = maxPositions.bufferSync();
    const dyBuf = dy.bufferSync();

    for (let b = 0; b < convInfo.batchSize; ++b) {
      for (let d = 0; d < convInfo.inChannels; ++d) {
        for (let dxR = 0; dxR < convInfo.inHeight; ++dxR) {
          for (let dxC = 0; dxC < convInfo.inWidth; ++dxC) {
            // Shader code begins.
            const dyRCorner = dxR - padTop;
            const dyCCorner = dxC - padLeft;
            let dotProd = 0;
            for (let wR = 0; wR < effectiveFilterHeight; wR += dilationHeight) {
              const dyR = (dyRCorner + wR) / strideHeight;
              if (dyR < 0 || dyR >= convInfo.outHeight ||
                  Math.floor(dyR) !== dyR) {
                continue;
              }
              for (let wC = 0; wC < effectiveFilterWidth; wC += dilationWidth) {
                const dyC = (dyCCorner + wC) / strideWidth;
                if (dyC < 0 || dyC >= convInfo.outWidth ||
                    Math.floor(dyC) !== dyC) {
                  continue;
                }
                const maxPos = effectiveFilterHeight * effectiveFilterWidth -
                    1 - maxPosBuf.get(b, dyR, dyC, d);
                const curPos = wR * effectiveFilterWidth + wC;

                const mask = maxPos === curPos ? 1 : 0;
                if (mask === 0) {
                  continue;
                }

                const pixel = dyBuf.get(b, dyR, dyC, d);
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
    this.assertNotComplex([dy, x], 'avgPoolBackprop');

    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const effectiveFilterHeight = convInfo.effectiveFilterHeight;
    const effectiveFilterWidth = convInfo.effectiveFilterWidth;
    const padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;
    const padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
    const dx = ops.buffer<Rank.R4>(x.shape, 'float32');

    const avgMultiplier = 1 / (filterHeight * filterWidth);

    const dyBuf = dy.bufferSync();

    for (let b = 0; b < convInfo.batchSize; ++b) {
      for (let d = 0; d < convInfo.inChannels; ++d) {
        for (let dxR = 0; dxR < convInfo.inHeight; ++dxR) {
          for (let dxC = 0; dxC < convInfo.inWidth; ++dxC) {
            // Shader code begins.
            const dyRCorner = dxR - padTop;
            const dyCCorner = dxC - padLeft;
            let dotProd = 0;
            for (let wR = 0; wR < effectiveFilterHeight; wR += dilationHeight) {
              const dyR = (dyRCorner + wR) / strideHeight;
              if (dyR < 0 || dyR >= convInfo.outHeight ||
                  Math.floor(dyR) !== dyR) {
                continue;
              }
              for (let wC = 0; wC < effectiveFilterWidth; wC += dilationWidth) {
                const dyC = (dyCCorner + wC) / strideWidth;
                if (dyC < 0 || dyC >= convInfo.outWidth ||
                    Math.floor(dyC) !== dyC) {
                  continue;
                }

                const pixel = dyBuf.get(b, dyR, dyC, d);
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

  cast<T extends Tensor>(x: T, dtype: DataType): T {
    return backend_util.castTensor(x, dtype, this);
  }

  reshape<R extends Rank>(x: Tensor, shape: ShapeMap[R]): Tensor<R> {
    return backend_util.reshapeTensor(x, shape);
  }

  avgPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    this.assertNotComplex(x, 'avgPool');

    return this.pool(x, convInfo, 'avg').toFloat();
  }

  resizeBilinear(
      x: Tensor4D, newHeight: number, newWidth: number,
      alignCorners: boolean): Tensor4D {
    this.assertNotComplex(x, 'resizeBilinear');

    const [batch, oldHeight, oldWidth, numChannels] = x.shape;
    const xValues = x.dataSync();
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
        const sourceFracRow = effectiveRowSizeRatio * r;
        const sourceRowFloor = Math.floor(sourceFracRow);
        const rowFrac = sourceFracRow - sourceRowFloor;
        const sourceRowCeil = Math.min(oldHeight - 1, Math.ceil(sourceFracRow));
        const topRowOffset = b * x.strides[0] + sourceRowFloor * x.strides[1];
        const botRowOffset = b * x.strides[0] + sourceRowCeil * x.strides[1];
        for (let c = 0; c < newWidth; c++) {
          const sourceFracCol = effectiveColSizeRatio * c;
          const sourceColFloor = Math.floor(sourceFracCol);
          const colFrac = sourceFracCol - sourceColFloor;
          const sourceColCeil =
              Math.min(oldWidth - 1, Math.ceil(sourceFracCol));
          const topLeftOffest = topRowOffset + sourceColFloor * x.strides[2];
          const botLeftOffset = botRowOffset + sourceColFloor * x.strides[2];
          const topRightOffset = topRowOffset + +sourceColCeil * x.strides[2];
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
    return ops.tensor(result, [batch, newHeight, newWidth, numChannels]);
  }

  resizeBilinearBackprop(dy: Tensor4D, x: Tensor4D, alignCorners: boolean) {
    this.assertNotComplex([dy, x], 'resizeBilinearBackprop');

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

    const dyValues = dy.dataSync();
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
    return ops.tensor4d(output, [batch, xWidth, xHeight, depth], x.dtype);
  }

  resizeNearestNeighbor(
      x: Tensor4D, newHeight: number, newWidth: number,
      alignCorners: boolean): Tensor4D {
    this.assertNotComplex(x, 'resizeNearestNeighbor');

    const [batch, oldHeight, oldWidth, numChannels] = x.shape;
    const xValues = x.dataSync();
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
    return ops.tensor(
        output, [batch, newHeight, newWidth, numChannels], x.dtype);
  }

  resizeNearestNeighborBackprop(
      dy: Tensor4D, x: Tensor4D, alignCorners: boolean) {
    this.assertNotComplex([dy, x], 'resizeNearestNeighborBackprop');

    const [batch, xHeight, xWidth, depth] = x.shape;
    const [, yHeight, yWidth] = dy.shape;

    const output = new Float32Array(batch * xHeight * xWidth * depth);
    const dyValues = dy.dataSync();

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
    return ops.tensor4d(output, x.shape, x.dtype);
  }

  batchNormalization(
      x: Tensor4D, mean: Tensor4D|Tensor1D, variance: Tensor4D|Tensor1D,
      varianceEpsilon: number, scale?: Tensor4D|Tensor1D,
      offset?: Tensor4D|Tensor1D): Tensor4D {
    this.assertNotComplex([x, mean, variance, scale, offset], 'batchNorm');

    const xVals = x.dataSync();
    const mVals = mean.dataSync();
    const varVals = variance.dataSync();
    const sVals = scale ? scale.dataSync() : new Float32Array([1]);
    const offVals = offset ? offset.dataSync() : new Float32Array([0]);
    const outVals = new Float32Array(xVals.length);

    const offValsLength = offVals.length;
    const sValsLength = sVals.length;
    const varValsLength = varVals.length;
    const mValsLength = mVals.length;

    let offi = 0;
    let mi = 0;
    let si = 0;
    let vi = 0;
    for (let i = 0; i < xVals.length; ++i) {
      outVals[i] = offVals[offi++] +
          (xVals[i] - mVals[mi++]) * sVals[si++] /
              Math.sqrt(varVals[vi++] + varianceEpsilon);
      if (offi >= offValsLength) {
        offi = 0;
      }
      if (mi >= mValsLength) {
        mi = 0;
      }
      if (si >= sValsLength) {
        si = 0;
      }
      if (vi >= varValsLength) {
        vi = 0;
      }
    }
    return tensor4d(outVals, x.shape);
  }

  localResponseNormalization4D(
      x: Tensor4D, depthRadius: number, bias: number, alpha: number,
      beta: number): Tensor4D {
    this.assertNotComplex(x, 'localResponseNormalization4D');

    const channels = x.shape[3];
    const maxD = channels - 1;
    const xValues = x.dataSync();
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

    return ops.tensor4d(result, x.shape);
  }

  LRNGrad(
      dy: Tensor4D, inputImage: Tensor4D, outputImage: Tensor4D,
      depthRadius: number, bias: number, alpha: number,
      beta: number): Tensor4D {
    this.assertNotComplex(dy, 'LRNGrad');
    const channels = dy.shape[3];
    const dyValues = dy.dataSync();
    const inputImageValues = inputImage.dataSync();
    const outputImageValues = outputImage.dataSync();
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
    return ops.tensor4d(result, dy.shape);
  }

  multinomial(
      logits: Tensor2D, normalized: boolean, numSamples: number,
      seed: number): Tensor2D {
    this.assertNotComplex(logits, 'multinomial');

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
    this.assertNotComplex(indices, 'oneHot');

    const res = new Float32Array(indices.size * depth);
    res.fill(offValue);
    const indicesVal = indices.dataSync();

    for (let event = 0; event < indices.size; ++event) {
      if (indicesVal[event] >= 0 && indicesVal[event] < depth) {
        res[event * depth + indicesVal[event]] = onValue;
      }
    }
    return ops.tensor2d(res, [indices.size, depth], 'int32');
  }

  nonMaxSuppression(
      boxes: Tensor2D, scores: Tensor1D, maxOutputSize: number,
      iouThreshold: number, scoreThreshold: number): Tensor1D {
    this.assertNotComplex(boxes, 'nonMaxSuppression');

    const boxesVals = boxes.dataSync();
    const scoresVals = scores.dataSync();
    return nonMaxSuppressionImpl(
        boxesVals, scoresVals, maxOutputSize, iouThreshold, scoreThreshold);
  }

  fft(x: Tensor2D): Tensor2D {
    return this.fftBatch(x, false);
  }

  ifft(x: Tensor2D): Tensor2D {
    return this.fftBatch(x, true);
  }

  /**
   * Calculate FFT of inner most elements of batch tensor.
   */
  private fftBatch(x: Tensor2D, inverse: boolean): Tensor2D {
    const batch = x.shape[0];
    const innerDim = x.shape[1];
    // Collects real and imaginary values separately.
    const realResult = ops.buffer(x.shape, 'float32');
    const imagResult = ops.buffer(x.shape, 'float32');

    const real = ops.real(x).as2D(batch, innerDim);
    const imag = ops.imag(x).as2D(batch, innerDim);

    for (let b = 0; b < batch; b++) {
      // TODO: Support slice ops for complex type.
      const r = real.slice([b, 0], [1, innerDim]);
      const i = imag.slice([b, 0], [1, innerDim]);
      const input = ops.complex(r, i);
      // Run FFT by batch element.
      const res = this.fftImpl(input, inverse).dataSync() as Float32Array;
      for (let d = 0; d < innerDim; d++) {
        const c = complex_util.getComplexWithIndex(res, d);
        realResult.values[b * innerDim + d] = c.real;
        imagResult.values[b * innerDim + d] = c.imag;
      }
    }

    const t = ops.complex(realResult.toTensor(), imagResult.toTensor());
    return t.as2D(batch, innerDim);
  }

  private fftImpl(x: Tensor2D, inverse: boolean): Tensor2D {
    const x1D = x.as1D();

    const n = x1D.size;

    if (this.isExponentOf2(n)) {
      let result = this.fftRadix2(x1D, n, inverse).as2D(x.shape[0], x.shape[1]);
      if (inverse) {
        result = ops.complex(
                     ops.real(result).div(scalar(n)),
                     ops.imag(result).div(scalar(n))) as Tensor2D;
      }
      return result;
    } else {
      const data = x.dataSync();
      const rawOutput =
          this.fourierTransformByMatmul(data, n, inverse) as Float32Array;
      const output = complex_util.splitRealAndImagArrays(rawOutput);
      return ops.complex(output.real, output.imag).as2D(x.shape[0], x.shape[1]);
    }
  }

  private isExponentOf2(size: number): boolean {
    return (size & size - 1) === 0;
  }

  // FFT using Cooley-Tukey algorithm on radix 2 dimensional input.
  private fftRadix2(input: Tensor1D, size: number, inverse: boolean): Tensor1D {
    if (size === 1) {
      return input;
    }
    const data = input.dataSync() as Float32Array;
    const half = size / 2;
    const evenComplex = complex_util.complexWithEvenIndex(data);
    let evenTensor = ops.complex(evenComplex.real, evenComplex.imag).as1D();
    const oddComplex = complex_util.complexWithOddIndex(data);
    let oddTensor = ops.complex(oddComplex.real, oddComplex.imag).as1D();

    // Recursive call for half part of original input.
    evenTensor = this.fftRadix2(evenTensor, half, inverse);
    oddTensor = this.fftRadix2(oddTensor, half, inverse);

    const e = complex_util.exponents(size, inverse);
    const exponent = ops.complex(e.real, e.imag).mul(oddTensor);

    const addPart = evenTensor.add(exponent);
    const subPart = evenTensor.sub(exponent);

    const realTensor = ops.real(addPart).concat(ops.real(subPart));
    const imagTensor = ops.imag(addPart).concat(ops.imag(subPart));

    return ops.complex(realTensor, imagTensor).as1D();
  }

  // Calculate fourier transform by multplying sinusoid matrix.
  private fourierTransformByMatmul(
      data: TypedArray, size: number, inverse: boolean): TypedArray {
    const ret = new Float32Array(size * 2);
    // TODO: Use matmul instead once it supports complex64 type.
    for (let r = 0; r < size; r++) {
      let real = 0.0;
      let imag = 0.0;
      for (let c = 0; c < size; c++) {
        const e = complex_util.exponent(r * c, size, inverse);
        const term = complex_util.getComplexWithIndex(data as Float32Array, c);
        real += term.real * e.real - term.imag * e.imag;
        imag += term.real * e.imag + term.imag * e.real;
      }
      if (inverse) {
        real /= size;
        imag /= size;
      }
      complex_util.assignToTypedArray(ret, real, imag, r);
    }
    return ret;
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

    const xValues = x.dataSync();
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
    return ops.tensor4d(
        result, [batchSize, outputHeight, outputWidth, outputDepth]);
  }

  private broadcastedBinaryOp(
      a: Tensor, b: Tensor, dtype: DataType,
      op: (a: number, b: number) => number): Tensor {
    const newShape =
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    const result = ops.buffer(newShape, dtype);
    const aVals = a.dataSync();
    const bVals = b.dataSync();
    const aBroadcastDims = broadcast_util.getBroadcastDims(a.shape, newShape);
    const bBroadcastDims = broadcast_util.getBroadcastDims(b.shape, newShape);

    const resVals = result.values;
    if (aBroadcastDims.length + bBroadcastDims.length === 0) {
      for (let i = 0; i < resVals.length; ++i) {
        resVals[i] = op(aVals[i % aVals.length], bVals[i % bVals.length]);
      }
    } else {
      const aBuf = a.bufferSync();
      const bBuf = b.bufferSync();
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

  private broadcastedBinaryComplexOp(
      a: Tensor, b: Tensor,
      op:
          (aReal: number, aImag: number, bReal: number,
           bImag: number) => {real: number, imag: number}): Tensor {
    const newShape =
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    const realResult = ops.buffer(newShape, 'float32');
    const imagResult = ops.buffer(newShape, 'float32');

    const aVals = a.dataSync();
    const bVals = b.dataSync();
    const aBroadcastDims = broadcast_util.getBroadcastDims(a.shape, newShape);
    const bBroadcastDims = broadcast_util.getBroadcastDims(b.shape, newShape);

    const realVals = realResult.values;
    const imagVals = imagResult.values;

    if (aBroadcastDims.length + bBroadcastDims.length === 0) {
      for (let i = 0; i < realVals.length; i++) {
        const aIdx = i % aVals.length;
        const bIdx = i % bVals.length;

        const result =
            op(aVals[aIdx * 2], aVals[aIdx * 2 + 1], bVals[bIdx * 2],
               bVals[bIdx * 2 + 1]);

        realVals[i] = result.real;
        imagVals[i] = result.imag;
      }
    } else {
      const aRealBuf = this.data.get(a.dataId).complexTensors.real.bufferSync();
      const bRealBuf = this.data.get(b.dataId).complexTensors.real.bufferSync();
      for (let i = 0; i < realVals.length; i++) {
        const loc = realResult.indexToLoc(i);

        const aLoc = loc.slice(-a.rank);
        aBroadcastDims.forEach(d => aLoc[d] = 0);
        const aIndex = aRealBuf.locToIndex(aLoc);

        const bLoc = loc.slice(-b.rank);
        bBroadcastDims.forEach(d => bLoc[d] = 0);
        const bIndex = bRealBuf.locToIndex(bLoc);

        const opResult =
            op(aVals[aIndex * 2], aVals[aIndex * 2 + 1], bVals[bIndex * 2],
               bVals[bIndex * 2 + 1]);

        realVals[i] = opResult.real;
        imagVals[i] = opResult.imag;
      }
    }
    return this.complex(realResult.toTensor(), imagResult.toTensor());
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
    return EPSILON_FLOAT32;
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
    const output = ops.buffer(
        [numBoxes, cropHeight, cropWidth, numChannels], images.dtype);

    const boxVals = boxes.dataSync();
    const boxIndVals = boxIndex.dataSync();
    const imageVals = images.dataSync();

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
        scatter_nd_util.calculateShapes(
            sparseValues, sparseIndices, outputShape);
    const sumDupeIndices = false;
    return this.scatter(
        sparseIndices, sparseValues, outputShape, outputSize, sliceSize,
        numUpdates, sliceRank, strides, defaultValue, sumDupeIndices);
  }

  gatherND(x: Tensor, indices: Tensor): Tensor {
    const indicesShape = indices.shape;
    const sliceRank = indicesShape[indicesShape.length - 1];

    const [resultShape, numSlices, sliceSize, strides] =
        gather_nd_util.prepareAndValidate(x, indices);
    if (numSlices === 0) {
      return tensor([], resultShape, x.dtype);
    }

    const buffer = new TensorBuffer([numSlices, sliceSize], x.dtype);
    const indicesData = indices.dataSync();
    const xData = x.dataSync();

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
        scatter_nd_util.calculateShapes(updates, indices, shape);
    const defaultValue = scalar(0);
    const sumDupeIndices = true;
    return this.scatter(
        indices, updates, shape, outputSize, sliceSize, numUpdates, sliceRank,
        strides, defaultValue, sumDupeIndices);
  }

  fill<R extends Rank>(
      shape: ShapeMap[R], value: number|string, dtype?: DataType): Tensor<R> {
    dtype = dtype || inferDtype(value);
    const values = getArrayFromDType(dtype, sizeFromShape(shape)) as TypedArray;
    values.fill(value as number);
    return Tensor.make(shape, {values}, dtype);
  }

  onesLike<R extends Rank>(x: Tensor<R>): Tensor<R> {
    if (x.dtype === 'string') {
      throw new Error('onesLike is not supported for string tensors');
    } else {
      return this.fill(x.shape, 1, x.dtype);
    }
  }

  zerosLike<R extends Rank>(x: Tensor<R>): Tensor<R> {
    const values =
        getArrayFromDType(x.dtype, sizeFromShape(x.shape)) as TypedArray;
    return Tensor.make(x.shape, {values}, x.dtype);
  }

  private scatter<R extends Rank>(
      indices: Tensor, updates: Tensor, shape: ShapeMap[R], outputSize: number,
      sliceSize: number, numUpdates: number, sliceRank: number,
      strides: number[], defaultValue: Scalar,
      sumDupeIndices: boolean): Tensor<R> {
    const flattenShape = [outputSize / sliceSize, sliceSize];
    const indicesData = indices.dataSync();
    const updatesData = updates.dataSync();

    if (outputSize === 0) {
      return tensor([], shape, updates.dtype);
    }

    const buffer = new TensorBuffer(flattenShape, updates.dtype as 'float32');
    buffer.values.fill(defaultValue.dataSync()[0]);

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

ENGINE.registerBackend('cpu', () => new MathBackendCPU(), 1 /* priority */);
