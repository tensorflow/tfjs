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
import {FusedBatchMatMulConfig, FusedConv2DConfig} from '../ops/fused_util';
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
  write(values: BackendValues, shape: number[], dtype: DataType): DataId;
  move(dataId: DataId, values: BackendValues, shape: number[], dtype: DataType):
      void;
  fromPixels(
      pixels: PixelData|ImageData|HTMLImageElement|HTMLCanvasElement|
      HTMLVideoElement,
      numChannels: number): Tensor3D;
  memory(): {unreliable: boolean;};  // Backend-specific information.
  /** Returns number of data ids currently in the storage. */
  numDataIds(): number;
}

/** Convenient class for storing tensor-related data. */
export class DataStorage<T> {
  private data = new WeakMap<DataId, T>();
  private dataIdsCount = 0;

  constructor(private backend: KernelBackend, private dataMover: DataMover) {}

  get(dataId: DataId) {
    if (!this.data.has(dataId)) {
      this.dataMover.moveData(this.backend, dataId);
    }
    return this.data.get(dataId);
  }

  set(dataId: DataId, value: T): void {
    this.dataIdsCount++;
    this.data.set(dataId, value);
  }

  has(dataId: DataId): boolean {
    return this.data.has(dataId);
  }

  delete(dataId: DataId): boolean {
    this.dataIdsCount--;
    return this.data.delete(dataId);
  }

  numDataIds(): number {
    return this.dataIdsCount;
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
    return notYetImplemented();
  }
  read(dataId: object): Promise<BackendValues> {
    return notYetImplemented();
  }
  readSync(dataId: object): BackendValues {
    return notYetImplemented();
  }
  numDataIds(): number {
    return notYetImplemented();
  }
  disposeData(dataId: object): void {
    return notYetImplemented();
  }
  fromPixels(
      pixels: PixelData|ImageData|HTMLImageElement|HTMLCanvasElement|
      HTMLVideoElement,
      numChannels: number): Tensor<Rank.R3> {
    return notYetImplemented();
  }
  write(values: BackendValues, shape: number[], dtype: DataType): DataId {
    return notYetImplemented();
  }
  move(dataId: DataId, values: BackendValues, shape: number[], dtype: DataType):
      void {
    return notYetImplemented();
  }
  memory(): {unreliable: boolean; reasons?: string[]} {
    return notYetImplemented();
  }
  /** Returns the highest precision for floats in bits (e.g. 16 or 32) */
  floatPrecision(): 16|32 {
    return notYetImplemented();
  }
  /** Returns the smallest representable number.  */
  epsilon(): number {
    return this.floatPrecision() === 32 ? EPSILON_FLOAT32 : EPSILON_FLOAT16;
  }

  batchMatMul(
      a: Tensor3D, b: Tensor3D, transposeA: boolean,
      transposeB: boolean): Tensor3D {
    return notYetImplemented();
  }

  fusedBatchMatMul(
      {a, b, transposeA, transposeB, bias, activation, preluActivationWeights}:
          FusedBatchMatMulConfig): Tensor3D {
    return notYetImplemented();
  }

  slice<T extends Tensor>(x: T, begin: number[], size: number[]): T {
    return notYetImplemented();
  }
  stridedSlice<T extends Tensor>(
      x: T, begin: number[], end: number[], strides: number[]): T {
    return notYetImplemented();
  }
  unstack(x: Tensor, axis: number): Tensor[] {
    return notYetImplemented();
  }
  reverse<T extends Tensor>(a: T, axis: number[]): T {
    return notYetImplemented();
  }

  concat(tensors: Tensor[], axis: number): Tensor {
    return notYetImplemented();
  }

  neg<T extends Tensor>(a: T): T {
    return notYetImplemented();
  }

  add(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented();
  }
  addN<T extends Tensor>(tensors: T[]): T {
    return notYetImplemented();
  }
  subtract(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented();
  }
  multiply(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented();
  }
  realDivide(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented();
  }
  floorDiv(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented();
  }

  sum(x: Tensor, axes: number[]): Tensor {
    return notYetImplemented();
  }
  prod(x: Tensor, axes: number[]): Tensor {
    return notYetImplemented();
  }

  unsortedSegmentSum<T extends Tensor>(
      x: T, segmentIds: Tensor1D, numSegments: number): Tensor {
    return notYetImplemented();
  }

  argMin(x: Tensor, axis: number): Tensor {
    return notYetImplemented();
  }
  argMax(x: Tensor, axis: number): Tensor {
    return notYetImplemented();
  }

  equal(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented();
  }
  notEqual(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented();
  }

  less(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented();
  }
  lessEqual(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented();
  }

  greater(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented();
  }
  greaterEqual(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented();
  }

  logicalNot<T extends Tensor>(a: T): T {
    return notYetImplemented();
  }
  logicalAnd(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented();
  }
  logicalOr(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented();
  }

  where(condition: Tensor): Tensor2D {
    return notYetImplemented();
  }
  select(condition: Tensor, a: Tensor, b: Tensor): Tensor {
    return notYetImplemented();
  }

  topk<T extends Tensor>(x: T, k: number, sorted: boolean): [T, T] {
    return notYetImplemented();
  }

  min(x: Tensor, axes: number[]): Tensor {
    return notYetImplemented();
  }
  minimum(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented();
  }

  mod(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented();
  }

  max(x: Tensor, axes: number[]): Tensor {
    return notYetImplemented();
  }
  maximum(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented();
  }

  all(x: Tensor, axes: number[]): Tensor {
    return notYetImplemented();
  }
  any(x: Tensor, axes: number[]): Tensor {
    return notYetImplemented();
  }

  squaredDifference(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented();
  }

  ceil<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }
  floor<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }
  round<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }

  sign<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }

  isNaN<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }
  isInf<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }
  isFinite<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }

  pow<T extends Tensor>(a: T, b: Tensor): T {
    return notYetImplemented();
  }
  exp<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }
  expm1<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }
  log<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }
  log1p<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }
  sqrt<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }
  rsqrt<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }
  square<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }
  reciprocal<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }
  relu<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }
  relu6<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }
  prelu<T extends Tensor>(x: T, a: T): T {
    return notYetImplemented();
  }
  elu<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }
  eluDer<T extends Tensor>(dy: T, y: T): T {
    return notYetImplemented();
  }
  selu<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }
  int<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }

  clip<T extends Tensor>(x: T, min: number, max: number): T {
    return notYetImplemented();
  }

  abs<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }
  complexAbs<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }

  sigmoid<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }

  softplus<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }

  sin<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }
  cos<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }
  tan<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }

  asin<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }
  acos<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }
  atan<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }
  atan2<T extends Tensor>(a: T, b: T): T {
    return notYetImplemented();
  }

  sinh<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }
  cosh<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }
  tanh<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }

  asinh<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }
  acosh<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }
  atanh<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }

  erf<T extends Tensor>(x: T): T {
    return notYetImplemented();
  }

  step<T extends Tensor>(x: T, alpha: number): T {
    return notYetImplemented();
  }

  fusedConv2d(
      {input, filter, convInfo, bias, activation, preluActivationWeights}:
          FusedConv2DConfig): Tensor4D {
    return notYetImplemented();
  }

  conv2d(x: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    return notYetImplemented();
  }
  conv2dDerInput(dy: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    return notYetImplemented();
  }
  conv2dDerFilter(x: Tensor4D, dY: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    return notYetImplemented();
  }

  fusedDepthwiseConv2D(
      {input, filter, convInfo, bias, activation, preluActivationWeights}:
          FusedConv2DConfig): Tensor4D {
    return notYetImplemented();
  }

  depthwiseConv2D(input: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    return notYetImplemented();
  }
  depthwiseConv2DDerInput(dy: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    return notYetImplemented();
  }
  depthwiseConv2DDerFilter(x: Tensor4D, dY: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    return notYetImplemented();
  }
  conv3d(x: Tensor5D, filter: Tensor5D, convInfo: Conv3DInfo): Tensor5D {
    return notYetImplemented();
  }
  conv3dDerInput(dy: Tensor5D, filter: Tensor5D, convInfo: Conv3DInfo):
      Tensor5D {
    return notYetImplemented();
  }
  conv3dDerFilter(x: Tensor5D, dY: Tensor5D, convInfo: Conv3DInfo): Tensor5D {
    return notYetImplemented();
  }
  maxPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    return notYetImplemented();
  }
  maxPoolBackprop(dy: Tensor4D, x: Tensor4D, y: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    return notYetImplemented();
  }
  avgPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    return notYetImplemented();
  }
  avgPoolBackprop(dy: Tensor4D, x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    return notYetImplemented();
  }
  avgPool3d(x: Tensor5D, convInfo: Conv3DInfo): Tensor5D {
    return notYetImplemented();
  }
  avgPool3dBackprop(dy: Tensor5D, x: Tensor5D, convInfo: Conv3DInfo): Tensor5D {
    return notYetImplemented();
  }
  maxPool3d(x: Tensor5D, convInfo: Conv3DInfo): Tensor5D {
    return notYetImplemented();
  }
  maxPool3dBackprop(
      dy: Tensor5D, x: Tensor5D, y: Tensor5D, convInfo: Conv3DInfo): Tensor5D {
    return notYetImplemented();
  }

  reshape<T extends Tensor, R extends Rank>(x: T, shape: ShapeMap[R]):
      Tensor<R> {
    return notYetImplemented();
  }
  cast<T extends Tensor>(x: T, dtype: DataType): T {
    return notYetImplemented();
  }

  tile<T extends Tensor>(x: T, reps: number[]): T {
    return notYetImplemented();
  }

  pad<T extends Tensor>(
      x: T, paddings: Array<[number, number]>, constantValue: number): T {
    return notYetImplemented();
  }

  transpose<T extends Tensor>(x: T, perm: number[]): T {
    return notYetImplemented();
  }

  gather<T extends Tensor>(x: T, indices: Tensor1D, axis: number): T {
    return notYetImplemented();
  }

  gatherND(x: Tensor, indices: Tensor): Tensor {
    return notYetImplemented();
  }

  scatterND<R extends Rank>(
      indices: Tensor, updates: Tensor, shape: ShapeMap[R]): Tensor<R> {
    return notYetImplemented();
  }

  batchToSpaceND<T extends Tensor>(
      x: T, blockShape: number[], crops: number[][]): T {
    return notYetImplemented();
  }

  spaceToBatchND<T extends Tensor>(
      x: T, blockShape: number[], paddings: number[][]): T {
    return notYetImplemented();
  }

  resizeBilinear(
      x: Tensor4D, newHeight: number, newWidth: number,
      alignCorners: boolean): Tensor4D {
    return notYetImplemented();
  }

  resizeBilinearBackprop(dy: Tensor4D, x: Tensor4D, alignCorners: boolean):
      Tensor4D {
    return notYetImplemented();
  }

  resizeNearestNeighbor(
      x: Tensor4D, newHEight: number, newWidth: number,
      alignCorners: boolean): Tensor4D {
    return notYetImplemented();
  }

  resizeNearestNeighborBackprop(
      dy: Tensor4D, x: Tensor4D, alignCorners: boolean): Tensor4D {
    return notYetImplemented();
  }

  batchNormalization(
      x: Tensor4D, mean: Tensor4D|Tensor1D, variance: Tensor4D|Tensor1D,
      varianceEpsilon: number, scale?: Tensor4D|Tensor1D,
      offset?: Tensor4D|Tensor1D): Tensor4D {
    return notYetImplemented();
  }

  localResponseNormalization4D(
      x: Tensor4D, radius: number, bias: number, alpha: number,
      beta: number): Tensor4D {
    return notYetImplemented();
  }

  LRNGrad(
      dy: Tensor4D, inputImage: Tensor4D, outputImage: Tensor4D, radius: number,
      bias: number, alpha: number, beta: number): Tensor4D {
    return notYetImplemented();
  }

  multinomial(
      logits: Tensor2D, normalized: boolean, numSamples: number,
      seed: number): Tensor2D {
    return notYetImplemented();
  }

  oneHot(indices: Tensor1D, depth: number, onValue: number, offValue: number):
      Tensor2D {
    return notYetImplemented();
  }

  cumsum(x: Tensor, axis: number, exclusive: boolean, reverse: boolean):
      Tensor {
    return notYetImplemented();
  }

  nonMaxSuppression(
      boxes: Tensor2D, scores: Tensor1D, maxOutputSize: number,
      iouThreshold: number, scoreThreshold?: number): Tensor1D {
    return notYetImplemented();
  }

  fft(x: Tensor2D): Tensor2D {
    return notYetImplemented();
  }
  ifft(x: Tensor2D): Tensor2D {
    return notYetImplemented();
  }
  complex<T extends Tensor>(real: T, imag: T): T {
    return notYetImplemented();
  }
  real<T extends Tensor>(input: T): T {
    return notYetImplemented();
  }
  imag<T extends Tensor>(input: T): T {
    return notYetImplemented();
  }

  cropAndResize(
      image: Tensor4D, boxes: Tensor2D, boxIndex: Tensor1D,
      cropSize: [number, number], method: 'bilinear'|'nearest',
      extrapolationValue: number): Tensor4D {
    return notYetImplemented();
  }

  depthToSpace(x: Tensor4D, blockSize: number, dataFormat: string): Tensor4D {
    return notYetImplemented();
  }

  // Aligns with the "SplitV" kernel in TensorFlow.
  split<T extends Tensor>(value: T, sizeSplits: number[], axis: number): T[] {
    return notYetImplemented();
  }

  sparseToDense<R extends Rank>(
      sparseIndices: Tensor, sparseValues: Tensor, outputShape: ShapeMap[R],
      defaultValue: Scalar): Tensor<R> {
    return notYetImplemented();
  }

  diag(x: Tensor): Tensor {
    return notYetImplemented();
  }

  fill<R extends Rank>(
      shape: ShapeMap[R], value: number|string, dtype?: DataType): Tensor<R> {
    throw new Error('Not yet implemented.');
  }

  onesLike<R extends Rank>(x: Tensor<R>): Tensor<R> {
    return notYetImplemented();
  }

  zerosLike<R extends Rank>(x: Tensor<R>): Tensor<R> {
    return notYetImplemented();
  }

  linspace(start: number, stop: number, num: number): Tensor1D {
    return notYetImplemented();
  }

  dispose(): void {
    return notYetImplemented();
  }
}

function notYetImplemented(): never {
  throw new Error(
      'Not yet implemented or not found in the registry. ' +
      'Did you forget to import the kernel?');
}
