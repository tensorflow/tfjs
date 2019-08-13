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

import {Conv2DInfo, Conv3DInfo} from '../ops/conv_util';
import {Activation, FusedBatchMatMulConfig} from '../ops/fused_util';
import {Backend, DataId, Scalar, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D, Tensor5D} from '../tensor';
import {BackendValues, DataType, PixelData, Rank, ShapeMap} from '../types';

export const EPSILON_FLOAT32 = 1e-7;
export const EPSILON_FLOAT16 = 1e-4;

// Required information for all backends.
export interface BackendTimingInfo {
  kernelMs: number;
  getExtraProfileInfo?(): string;  // a field for additional timing information
                                   // e.g. packing / unpacking for WebGL backend
}

export interface TensorStorage {
  read(dataId: DataId): Promise<BackendValues>;
  readSync(dataId: DataId): BackendValues;
  disposeData(dataId: DataId): void;
  write(dataId: DataId, values: BackendValues): void;
  fromPixels(
      pixels: PixelData|ImageData|HTMLImageElement|HTMLCanvasElement|
      HTMLVideoElement,
      numChannels: number): Tensor3D;
  register(dataId: DataId, shape: number[], dtype: DataType): void;
  memory(): {unreliable: boolean;};  // Backend-specific information.
}

/** Convenient class for storing tensor-related data. */
export class DataStorage<T> {
  private data = new WeakMap<DataId, T>();

  constructor(private backend: KernelBackend, private dataMover: DataMover) {}

  get(dataId: DataId) {
    if (!this.data.has(dataId)) {
      this.dataMover.moveData(this.backend, dataId);
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
  moveData(backend: KernelBackend, dataId: DataId): void;
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
export class KernelBackend implements TensorStorage, Backend, BackendTimer {
  time(f: () => void): Promise<BackendTimingInfo> {
    throw new Error('Not yet implemented.');
  }
  read(dataId: object): Promise<BackendValues> {
    throw new Error('Not yet implemented.');
  }
  readSync(dataId: object): BackendValues {
    throw new Error('Not yet implemented.');
  }
  disposeData(dataId: object): void {
    throw new Error('Not yet implemented.');
  }
  write(dataId: object, values: BackendValues): void {
    throw new Error('Not yet implemented.');
  }
  fromPixels(
      pixels: PixelData|ImageData|HTMLImageElement|HTMLCanvasElement|
      HTMLVideoElement,
      numChannels: number): Tensor<Rank.R3> {
    throw new Error('Not yet implemented.');
  }
  register(dataId: object, shape: number[], dtype: DataType): void {
    throw new Error('Not yet implemented.');
  }
  memory(): {unreliable: boolean; reasons?: string[]} {
    throw new Error('Not yet implemented.');
  }
  /** Returns the highest precision for floats in bits (e.g. 16 or 32) */
  floatPrecision(): 16|32 {
    throw new Error('Not yet implemented');
  }
  /** Returns the smallest representable number.  */
  epsilon(): number {
    return this.floatPrecision() === 32 ? EPSILON_FLOAT32 : EPSILON_FLOAT16;
  }

  batchMatMul(
      a: Tensor3D, b: Tensor3D, transposeA: boolean,
      transposeB: boolean): Tensor3D {
    throw new Error('Not yet implemented');
  }

  fusedBatchMatMul(
      {a, b, transposeA, transposeB, bias, activation, preluActivationWeights}:
          FusedBatchMatMulConfig): Tensor3D {
    throw new Error('Not yet implemented');
  }

  slice<T extends Tensor>(x: T, begin: number[], size: number[]): T {
    throw new Error('Not yet implemented');
  }
  stridedSlice<T extends Tensor>(
      x: T, begin: number[], end: number[], strides: number[],
      beginMask: number, endMask: number, ellipsisMask: number,
      newAxisMask: number, shrinkAxisMask: number): T {
    throw new Error('Not yet implemented');
  }
  unstack(x: Tensor, axis: number): Tensor[] {
    throw new Error('Not yet implemented');
  }
  reverse<T extends Tensor>(a: T, axis: number[]): T {
    throw new Error('Not yet implemented');
  }

  concat(tensors: Tensor[], axis: number): Tensor {
    throw new Error('Not yet implemented');
  }

  neg<T extends Tensor>(a: T): T {
    throw new Error('Not yet implemented');
  }

  add(a: Tensor, b: Tensor): Tensor {
    throw new Error('Not yet implemented');
  }
  addN<T extends Tensor>(tensors: T[]): T {
    throw new Error('Not yet implemented');
  }
  subtract(a: Tensor, b: Tensor): Tensor {
    throw new Error('Not yet implemented');
  }
  multiply(a: Tensor, b: Tensor): Tensor {
    throw new Error('Not yet implemented');
  }
  realDivide(a: Tensor, b: Tensor): Tensor {
    throw new Error('Not yet implemented');
  }
  floorDiv(a: Tensor, b: Tensor): Tensor {
    throw new Error('Not yet implemented');
  }

  sum(x: Tensor, axes: number[]): Tensor {
    throw new Error('Not yet implemented');
  }
  prod(x: Tensor, axes: number[]): Tensor {
    throw new Error('Not yet implemented');
  }

  unsortedSegmentSum<T extends Tensor>(
      x: T, segmentIds: Tensor1D, numSegments: number): Tensor {
    throw new Error('Not yet implemented');
  }

  argMin(x: Tensor, axis: number): Tensor {
    throw new Error('Not yet implemented');
  }
  argMax(x: Tensor, axis: number): Tensor {
    throw new Error('Not yet implemented');
  }

  equal(a: Tensor, b: Tensor): Tensor {
    throw new Error('Not yet implemented');
  }
  notEqual(a: Tensor, b: Tensor): Tensor {
    throw new Error('Not yet implemented');
  }

  less(a: Tensor, b: Tensor): Tensor {
    throw new Error('Not yet implemented');
  }
  lessEqual(a: Tensor, b: Tensor): Tensor {
    throw new Error('Not yet implemented');
  }

  greater(a: Tensor, b: Tensor): Tensor {
    throw new Error('Not yet implemented');
  }
  greaterEqual(a: Tensor, b: Tensor): Tensor {
    throw new Error('Not yet implemented');
  }

  logicalNot<T extends Tensor>(a: T): T {
    throw new Error('Not yet implemented');
  }
  logicalAnd(a: Tensor, b: Tensor): Tensor {
    throw new Error('Not yet implemented');
  }
  logicalOr(a: Tensor, b: Tensor): Tensor {
    throw new Error('Not yet implemented');
  }

  where(condition: Tensor): Tensor2D {
    throw new Error('Not yet implemented');
  }
  select(condition: Tensor, a: Tensor, b: Tensor): Tensor {
    throw new Error('Not yet implemented');
  }

  topk<T extends Tensor>(x: T, k: number, sorted: boolean): [T, T] {
    throw new Error('Not yet implemented');
  }

  min(x: Tensor, axes: number[]): Tensor {
    throw new Error('Not yet implemented');
  }
  minimum(a: Tensor, b: Tensor): Tensor {
    throw new Error('Not yet implemented');
  }

  mod(a: Tensor, b: Tensor): Tensor {
    throw new Error('Not yet implemented');
  }

  max(x: Tensor, axes: number[]): Tensor {
    throw new Error('Not yet implemented');
  }
  maximum(a: Tensor, b: Tensor): Tensor {
    throw new Error('Not yet implemented');
  }

  all(x: Tensor, axes: number[]): Tensor {
    throw new Error('Not yet implemented');
  }
  any(x: Tensor, axes: number[]): Tensor {
    throw new Error('Not yet implemented');
  }

  squaredDifference(a: Tensor, b: Tensor): Tensor {
    throw new Error('Not yet implemented');
  }

  ceil<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }
  floor<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }
  round<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }

  sign<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }

  isNaN<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }
  isInf<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }
  isFinite<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }

  pow<T extends Tensor>(a: T, b: Tensor): T {
    throw new Error('Not yet implemented');
  }
  exp<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }
  expm1<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }
  log<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }
  log1p<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }
  sqrt<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }
  rsqrt<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }

  square<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }
  reciprocal<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }
  relu<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }
  prelu<T extends Tensor>(x: T, a: T): T {
    throw new Error('Not yet implemented');
  }
  elu<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }
  eluDer<T extends Tensor>(dy: T, y: T): T {
    throw new Error('Not yet implemented');
  }
  selu<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }
  int<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }

  clip<T extends Tensor>(x: T, min: number, max: number): T {
    throw new Error('Not yet implemented');
  }

  abs<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }
  complexAbs<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }

  sigmoid<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }

  softplus<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }

  sin<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }
  cos<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }
  tan<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }

  asin<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }
  acos<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }
  atan<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }
  atan2<T extends Tensor>(a: T, b: T): T {
    throw new Error('Not yet implemented');
  }

  sinh<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }
  cosh<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }
  tanh<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }

  asinh<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }
  acosh<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }
  atanh<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }

  erf<T extends Tensor>(x: T): T {
    throw new Error('Not yet implemented');
  }

  step<T extends Tensor>(x: T, alpha: number): T {
    throw new Error('Not yet implemented');
  }

  fusedConv2d(
      x: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo, bias?: Tensor4D,
      activation?: Activation, preluActivationWeights?: Tensor): Tensor4D {
    throw new Error('Not yet implemented');
  }

  conv2d(x: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    throw new Error('Not yet implemented');
  }
  conv2dDerInput(dy: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    throw new Error('Not yet implemented');
  }
  conv2dDerFilter(x: Tensor4D, dY: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    throw new Error('Not yet implemented');
  }

  depthwiseConv2D(input: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    throw new Error('Not yet implemented');
  }
  depthwiseConv2DDerInput(dy: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    throw new Error('Not yet implemented');
  }
  depthwiseConv2DDerFilter(x: Tensor4D, dY: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    throw new Error('Not yet implemented');
  }
  conv3d(x: Tensor5D, filter: Tensor5D, convInfo: Conv3DInfo): Tensor5D {
    throw new Error('Not yet implemented');
  }
  conv3dDerInput(dy: Tensor5D, filter: Tensor5D, convInfo: Conv3DInfo):
      Tensor5D {
    throw new Error('Not yet implemented');
  }
  conv3dDerFilter(x: Tensor5D, dY: Tensor5D, convInfo: Conv3DInfo): Tensor5D {
    throw new Error('Not yet implemented');
  }
  maxPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    throw new Error('Not yet implemented');
  }
  maxPoolBackprop(dy: Tensor4D, x: Tensor4D, y: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    throw new Error('Not yet implemented');
  }
  avgPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    throw new Error('Not yet implemented');
  }
  avgPoolBackprop(dy: Tensor4D, x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    throw new Error('Not yet implemented');
  }
  avgPool3d(x: Tensor5D, convInfo: Conv3DInfo): Tensor5D {
    throw new Error('Not yet implemented');
  }
  avgPool3dBackprop(dy: Tensor5D, x: Tensor5D, convInfo: Conv3DInfo): Tensor5D {
    throw new Error('Not yet implemented');
  }
  maxPool3d(x: Tensor5D, convInfo: Conv3DInfo): Tensor5D {
    throw new Error('Not yet implemented');
  }
  maxPool3dBackprop(
      dy: Tensor5D, x: Tensor5D, y: Tensor5D, convInfo: Conv3DInfo): Tensor5D {
    throw new Error('Not yet implemented');
  }

  reshape<T extends Tensor, R extends Rank>(x: T, shape: ShapeMap[R]):
      Tensor<R> {
    throw new Error('Not yet implemented');
  }
  cast<T extends Tensor>(x: T, dtype: DataType): T {
    throw new Error('Not yet implemented');
  }

  tile<T extends Tensor>(x: T, reps: number[]): T {
    throw new Error('Not yet implemented');
  }

  pad<T extends Tensor>(
      x: T, paddings: Array<[number, number]>, constantValue: number): T {
    throw new Error('Not yet implemented');
  }

  transpose<T extends Tensor>(x: T, perm: number[]): T {
    throw new Error('Not yet implemented');
  }

  gather<T extends Tensor>(x: T, indices: Tensor1D, axis: number): T {
    throw new Error('Not yet implemented');
  }

  gatherND(x: Tensor, indices: Tensor): Tensor {
    throw new Error('Not yet implemented');
  }

  scatterND<R extends Rank>(
      indices: Tensor, updates: Tensor, shape: ShapeMap[R]): Tensor<R> {
    throw new Error('Not yet implemented');
  }

  batchToSpaceND<T extends Tensor>(
      x: T, blockShape: number[], crops: number[][]): T {
    throw new Error('Not yet implemented');
  }

  spaceToBatchND<T extends Tensor>(
      x: T, blockShape: number[], paddings: number[][]): T {
    throw new Error('Not yet implemented');
  }

  resizeBilinear(
      x: Tensor4D, newHeight: number, newWidth: number,
      alignCorners: boolean): Tensor4D {
    throw new Error('Not yet implemented');
  }

  resizeBilinearBackprop(dy: Tensor4D, x: Tensor4D, alignCorners: boolean):
      Tensor4D {
    throw new Error('Not yet implemented');
  }

  resizeNearestNeighbor(
      x: Tensor4D, newHEight: number, newWidth: number,
      alignCorners: boolean): Tensor4D {
    throw new Error('Not yet implemented');
  }

  resizeNearestNeighborBackprop(
      dy: Tensor4D, x: Tensor4D, alignCorners: boolean): Tensor4D {
    throw new Error('Not yet implemented');
  }

  batchNormalization(
      x: Tensor4D, mean: Tensor4D|Tensor1D, variance: Tensor4D|Tensor1D,
      varianceEpsilon: number, scale?: Tensor4D|Tensor1D,
      offset?: Tensor4D|Tensor1D): Tensor4D {
    throw new Error('Not yet implemented');
  }

  localResponseNormalization4D(
      x: Tensor4D, radius: number, bias: number, alpha: number,
      beta: number): Tensor4D {
    throw new Error('Not yet implemented');
  }

  LRNGrad(
      dy: Tensor4D, inputImage: Tensor4D, outputImage: Tensor4D, radius: number,
      bias: number, alpha: number, beta: number): Tensor4D {
    throw new Error('Not yet implemented');
  }

  multinomial(
      logits: Tensor2D, normalized: boolean, numSamples: number,
      seed: number): Tensor2D {
    throw new Error('Not yet implemented');
  }

  oneHot(indices: Tensor1D, depth: number, onValue: number, offValue: number):
      Tensor2D {
    throw new Error('Not yet implemented');
  }

  cumsum(x: Tensor, axis: number, exclusive: boolean, reverse: boolean):
      Tensor {
    throw new Error('Not yet implemented');
  }

  nonMaxSuppression(
      boxes: Tensor2D, scores: Tensor1D, maxOutputSize: number,
      iouThreshold: number, scoreThreshold?: number): Tensor1D {
    throw new Error('Not yet implemented');
  }

  fft(x: Tensor2D): Tensor2D {
    throw new Error('Not yet implemented');
  }
  ifft(x: Tensor2D): Tensor2D {
    throw new Error('Not yet implemented');
  }
  complex<T extends Tensor>(real: T, imag: T): T {
    throw new Error('Not yet implemented');
  }
  real<T extends Tensor>(input: T): T {
    throw new Error('Not yet implemented');
  }
  imag<T extends Tensor>(input: T): T {
    throw new Error('Not yet implemented');
  }

  cropAndResize(
      image: Tensor4D, boxes: Tensor2D, boxIndex: Tensor1D,
      cropSize: [number, number], method: 'bilinear'|'nearest',
      extrapolationValue: number): Tensor4D {
    throw new Error('Not yet implemented');
  }

  depthToSpace(x: Tensor4D, blockSize: number, dataFormat: string): Tensor4D {
    throw new Error('Not yet implemented');
  }

  // Aligns with the "SplitV" kernel in TensorFlow.
  split<T extends Tensor>(value: T, sizeSplits: number[], axis: number): T[] {
    throw new Error('Not yet implemented');
  }

  sparseToDense<R extends Rank>(
      sparseIndices: Tensor, sparseValues: Tensor, outputShape: ShapeMap[R],
      defaultValue: Scalar): Tensor<R> {
    throw new Error('Not yet implemented');
  }

  diag(x: Tensor): Tensor {
    throw new Error('Not yet implemented');
  }

  fill<R extends Rank>(
      shape: ShapeMap[R], value: number|string, dtype?: DataType): Tensor<R> {
    throw new Error('Not yet implemented.');
  }

  onesLike<R extends Rank>(x: Tensor<R>): Tensor<R> {
    throw new Error('Not yet implemented');
  }

  zerosLike<R extends Rank>(x: Tensor<R>): Tensor<R> {
    throw new Error('Not yet implemented');
  }

  linspace(start: number, stop: number, num: number): Tensor1D {
    throw new Error('Not yet implemented');
  }

  dispose(): void {
    throw new Error('Not yet implemented');
  }
}
