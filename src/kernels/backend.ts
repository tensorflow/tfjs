/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {Conv2DInfo} from '../ops/conv_util';
import {DataId, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D} from '../tensor';
import {DataType, Rank, ShapeMap, TypedArray} from '../types';

// Required information for all backends.
export interface BackendTimingInfo {
  kernelMs: number;
}

export interface TensorStorage {
  read(dataId: DataId): Promise<TypedArray>;
  readSync(dataId: DataId): TypedArray;
  disposeData(dataId: DataId): void;
  write(dataId: DataId, values: TypedArray): void;
  fromPixels(
      pixels: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      numChannels: number): Tensor3D;
  register(dataId: DataId, shape: number[], dtype: DataType): void;
  memory(): {unreliable: boolean;};  // Backend-specific information.
}

/** Convenient class for storing tensor-related data. */
export class DataStorage<T> {
  private data = new WeakMap<DataId, T>();

  constructor(private dataMover: DataMover) {}

  get(dataId: DataId) {
    if (!this.data.has(dataId)) {
      this.dataMover.moveData(dataId);
    }
    return this.data.get(dataId);
  }

  set(dataId: DataId, value: T): void {
    this.data.set(dataId, value);
  }

  has(dataId: DataId): boolean {
    return this.data.has(dataId);
  }

  delete(dataId: DataId): boolean {
    return this.data.delete(dataId);
  }
}

export interface DataMover {
  /**
   * To be called by backends whenever they see a dataId that they don't own.
   * Upon calling this method, the mover will fetch the tensor from another
   * backend and register it with the current active backend.
   */
  moveData(dataId: DataId): void;
}

export interface BackendTimer {
  time(f: () => void): Promise<BackendTimingInfo>;
}

/**
 * The interface that defines the kernels that should be implemented when
 * adding a new backend. New backends don't need to implement every one of the
 * methods, this can be done gradually (throw an error for unimplemented
 * methods).
 */
export interface KernelBackend extends TensorStorage, BackendTimer {
  /** Returns the highest precision for floats in bits (e.g. 16 or 32) */
  floatPrecision(): number;

  batchMatMul(
      a: Tensor3D, b: Tensor3D, transposeA: boolean,
      transposeB: boolean): Tensor3D;

  slice<T extends Tensor>(x: T, begin: number[], size: number[]): T;
  stridedSlice<T extends Tensor>(
      x: T, begin: number[], end: number[], strides: number[],
      beginMask: number, endMask: number, ellipsisMask: number,
      newAxisMask: number, shrinkAxisMask: number): T;
  reverse<T extends Tensor>(a: T, axis: number[]): T;

  concat(tensors: Tensor[], axis: number): Tensor;

  neg<T extends Tensor>(a: T): T;

  add(a: Tensor, b: Tensor): Tensor;
  addN<T extends Tensor>(tensors: T[]): T;
  subtract(a: Tensor, b: Tensor): Tensor;
  multiply(a: Tensor, b: Tensor): Tensor;
  realDivide(a: Tensor, b: Tensor): Tensor;
  floorDiv(a: Tensor, b: Tensor): Tensor;

  sum(x: Tensor, axes: number[]): Tensor;
  prod(x: Tensor, axes: number[]): Tensor;

  unsortedSegmentSum<T extends Tensor>(
      x: T, segmentIds: Tensor1D, numSegments: number): Tensor;

  argMin(x: Tensor, axis: number): Tensor;
  argMax(x: Tensor, axis: number): Tensor;

  equal(a: Tensor, b: Tensor): Tensor;
  notEqual(a: Tensor, b: Tensor): Tensor;

  less(a: Tensor, b: Tensor): Tensor;
  lessEqual(a: Tensor, b: Tensor): Tensor;

  greater(a: Tensor, b: Tensor): Tensor;
  greaterEqual(a: Tensor, b: Tensor): Tensor;

  logicalNot<T extends Tensor>(a: T): T;
  logicalAnd(a: Tensor, b: Tensor): Tensor;
  logicalOr(a: Tensor, b: Tensor): Tensor;

  where(condition: Tensor): Tensor2D;
  select(condition: Tensor, a: Tensor, b: Tensor): Tensor;

  topk<T extends Tensor>(x: T, k: number, sorted: boolean): [T, T];

  min(x: Tensor, axes: number[]): Tensor;
  minimum(a: Tensor, b: Tensor): Tensor;

  mod(a: Tensor, b: Tensor): Tensor;

  max(x: Tensor, axes: number[]): Tensor;
  maximum(a: Tensor, b: Tensor): Tensor;

  all(x: Tensor, axes: number[]): Tensor;
  any(x: Tensor, axes: number[]): Tensor;

  squaredDifference(a: Tensor, b: Tensor): Tensor;

  ceil<T extends Tensor>(x: T): T;
  floor<T extends Tensor>(x: T): T;
  round<T extends Tensor>(x: T): T;

  sign<T extends Tensor>(x: T): T;

  pow<T extends Tensor>(a: T, b: Tensor): T;
  exp<T extends Tensor>(x: T): T;
  expm1<T extends Tensor>(x: T): T;
  log<T extends Tensor>(x: T): T;
  log1p<T extends Tensor>(x: T): T;
  sqrt<T extends Tensor>(x: T): T;
  rsqrt<T extends Tensor>(x: T): T;

  square<T extends Tensor>(x: T): T;
  reciprocal<T extends Tensor>(x: T): T;

  relu<T extends Tensor>(x: T): T;
  elu<T extends Tensor>(x: T): T;
  eluDer<T extends Tensor>(dy: T, y: T): T;
  selu<T extends Tensor>(x: T): T;
  int<T extends Tensor>(x: T): T;

  clip<T extends Tensor>(x: T, min: number, max: number): T;

  abs<T extends Tensor>(x: T): T;

  sigmoid<T extends Tensor>(x: T): T;

  softplus<T extends Tensor>(x: T): T;

  sin<T extends Tensor>(x: T): T;
  cos<T extends Tensor>(x: T): T;
  tan<T extends Tensor>(x: T): T;

  asin<T extends Tensor>(x: T): T;
  acos<T extends Tensor>(x: T): T;
  atan<T extends Tensor>(x: T): T;
  atan2<T extends Tensor>(a: T, b: T): T;

  sinh<T extends Tensor>(x: T): T;
  cosh<T extends Tensor>(x: T): T;
  tanh<T extends Tensor>(x: T): T;

  asinh<T extends Tensor>(x: T): T;
  acosh<T extends Tensor>(x: T): T;
  atanh<T extends Tensor>(x: T): T;

  erf<T extends Tensor>(x: T): T;

  step<T extends Tensor>(x: T, alpha: number): T;

  conv2d(x: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo): Tensor4D;
  conv2dDerInput(dy: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D;
  conv2dDerFilter(x: Tensor4D, dY: Tensor4D, convInfo: Conv2DInfo): Tensor4D;

  depthwiseConv2D(input: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D;
  depthwiseConv2DDerInput(dy: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D;
  depthwiseConv2DDerFilter(x: Tensor4D, dY: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D;

  maxPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D;
  maxPoolBackprop(dy: Tensor4D, x: Tensor4D, y: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D;
  avgPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D;
  avgPoolBackprop(dy: Tensor4D, x: Tensor4D, convInfo: Conv2DInfo): Tensor4D;

  reshape<T extends Tensor, R extends Rank>(x: T, shape: ShapeMap[R]):
      Tensor<R>;
  cast<T extends Tensor>(x: T, dtype: DataType): T;

  tile<T extends Tensor>(x: T, reps: number[]): T;

  pad<T extends Tensor>(
      x: T, paddings: Array<[number, number]>, constantValue: number): T;

  transpose<T extends Tensor>(x: T, perm: number[]): T;

  gather<T extends Tensor>(x: T, indices: Tensor1D, axis: number): T;

  gatherND(x: Tensor, indices: Tensor): Tensor;

  scatterND<R extends Rank>(
      indices: Tensor, updates: Tensor, shape: ShapeMap[R]): Tensor<R>;

  batchToSpaceND<T extends Tensor>(
      x: T, blockShape: number[], crops: number[][]): T;

  spaceToBatchND<T extends Tensor>(
      x: T, blockShape: number[], paddings: number[][]): T;

  resizeBilinear(
      x: Tensor4D, newHeight: number, newWidth: number,
      alignCorners: boolean): Tensor4D;

  resizeBilinearBackprop(dy: Tensor4D, x: Tensor4D, alignCorners: boolean):
      Tensor4D;

  resizeNearestNeighbor(
      x: Tensor4D, newHEight: number, newWidth: number,
      alignCorners: boolean): Tensor4D;

  resizeNearestNeighborBackprop(
      dy: Tensor4D, x: Tensor4D, alignCorners: boolean): Tensor4D;

  batchNormalization(
      x: Tensor4D, mean: Tensor4D|Tensor1D, variance: Tensor4D|Tensor1D,
      varianceEpsilon: number, scale?: Tensor4D|Tensor1D,
      offset?: Tensor4D|Tensor1D): Tensor4D;

  localResponseNormalization4D(
      x: Tensor4D, radius: number, bias: number, alpha: number,
      beta: number): Tensor4D;

  LRNGrad(
      dy: Tensor4D, inputImage: Tensor4D, outputImage: Tensor4D, radius: number,
      bias: number, alpha: number, beta: number): Tensor4D;

  multinomial(
      logits: Tensor2D, normalized: boolean, numSamples: number,
      seed: number): Tensor2D;

  oneHot(indices: Tensor1D, depth: number, onValue: number, offValue: number):
      Tensor2D;

  cumsum(x: Tensor, axis: number, exclusive: boolean, reverse: boolean): Tensor;

  nonMaxSuppression(
      boxes: Tensor2D, scores: Tensor1D, maxOutputSize: number,
      iouThreshold: number, scoreThreshold?: number): Tensor1D;

  fft(x: Tensor1D): Tensor1D;
  complex<T extends Tensor>(real: T, imag: T): T;
  real<T extends Tensor>(input: T): T;
  imag<T extends Tensor>(input: T): T;

  cropAndResize(
      image: Tensor4D, boxes: Tensor2D, boxIndex: Tensor1D,
      cropSize: [number, number], method: 'bilinear'|'nearest',
      extrapolationValue: number): Tensor4D;

  depthToSpace(x: Tensor4D, blockSize: number, dataFormat: string): Tensor4D;

  // Aligns with the "SplitV" kernel in TensorFlow.
  split<T extends Tensor>(value: T, sizeSplits: number[], axis: number): T[];

  /**
   * Sets the data mover for this backend. Backends should use the mover to
   * move data from other backends to this backend.
   */
  setDataMover(dataMover: DataMover): void;

  dispose(): void;
}
