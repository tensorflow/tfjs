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
import {BackendValues, DataType, Rank, ShapeMap} from '../types';

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
    return notYetImplemented('time');
  }
  read(dataId: object): Promise<BackendValues> {
    return notYetImplemented('read');
  }
  readSync(dataId: object): BackendValues {
    return notYetImplemented('readSync');
  }
  numDataIds(): number {
    return notYetImplemented('numDataIds');
  }
  disposeData(dataId: object): void {
    return notYetImplemented('disposeData');
  }
  write(values: BackendValues, shape: number[], dtype: DataType): DataId {
    return notYetImplemented('write');
  }
  move(dataId: DataId, values: BackendValues, shape: number[], dtype: DataType):
      void {
    return notYetImplemented('move');
  }
  memory(): {unreliable: boolean; reasons?: string[]} {
    return notYetImplemented('memory');
  }
  /** Returns the highest precision for floats in bits (e.g. 16 or 32) */
  floatPrecision(): 16|32 {
    return notYetImplemented('floatPrecision');
  }
  /** Returns the smallest representable number.  */
  epsilon(): number {
    return this.floatPrecision() === 32 ? EPSILON_FLOAT32 : EPSILON_FLOAT16;
  }

  batchMatMul(
      a: Tensor3D, b: Tensor3D, transposeA: boolean,
      transposeB: boolean): Tensor3D {
    return notYetImplemented('batchMatMul');
  }

  fusedBatchMatMul(
      {a, b, transposeA, transposeB, bias, activation, preluActivationWeights}:
          FusedBatchMatMulConfig): Tensor3D {
    return notYetImplemented('fusedBatchMatMul');
  }

  slice<T extends Tensor>(x: T, begin: number[], size: number[]): T {
    return notYetImplemented('slice');
  }
  stridedSlice<T extends Tensor>(
      x: T, begin: number[], end: number[], strides: number[]): T {
    return notYetImplemented('stridedSlice');
  }
  unstack(x: Tensor, axis: number): Tensor[] {
    return notYetImplemented('unstack');
  }
  reverse<T extends Tensor>(a: T, axis: number[]): T {
    return notYetImplemented('reverse');
  }

  concat(tensors: Tensor[], axis: number): Tensor {
    return notYetImplemented('concat');
  }

  neg<T extends Tensor>(a: T): T {
    return notYetImplemented('neg');
  }

  add(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented('add');
  }
  addN<T extends Tensor>(tensors: T[]): T {
    return notYetImplemented('addN');
  }
  subtract(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented('subtract');
  }
  multiply(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented('multiply');
  }
  realDivide(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented('realDivide');
  }
  floorDiv(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented('floorDiv');
  }

  sum(x: Tensor, axes: number[]): Tensor {
    return notYetImplemented('sum');
  }
  prod(x: Tensor, axes: number[]): Tensor {
    return notYetImplemented('prod');
  }

  unsortedSegmentSum<T extends Tensor>(
      x: T, segmentIds: Tensor1D, numSegments: number): Tensor {
    return notYetImplemented('unsortedSegmentSum');
  }

  argMin(x: Tensor, axis: number): Tensor {
    return notYetImplemented('argMin');
  }
  argMax(x: Tensor, axis: number): Tensor {
    return notYetImplemented('argMax');
  }

  equal(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented('equal');
  }
  notEqual(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented('notEqual');
  }

  less(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented('less');
  }
  lessEqual(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented('lessEqual');
  }

  greater(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented('greater');
  }
  greaterEqual(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented('greaterEqual');
  }

  logicalNot<T extends Tensor>(a: T): T {
    return notYetImplemented('logicalNot');
  }
  logicalAnd(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented('logicalAnd');
  }
  logicalOr(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented('logicalOr');
  }

  where(condition: Tensor): Tensor2D {
    return notYetImplemented('where');
  }
  select(condition: Tensor, a: Tensor, b: Tensor): Tensor {
    return notYetImplemented('select');
  }

  topk<T extends Tensor>(x: T, k: number, sorted: boolean): [T, T] {
    return notYetImplemented('topk');
  }

  min(x: Tensor, axes: number[]): Tensor {
    return notYetImplemented('min');
  }
  minimum(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented('minimum');
  }

  mod(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented('mod');
  }

  max(x: Tensor, axes: number[]): Tensor {
    return notYetImplemented('max');
  }
  maximum(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented('maximum');
  }

  all(x: Tensor, axes: number[]): Tensor {
    return notYetImplemented('all');
  }
  any(x: Tensor, axes: number[]): Tensor {
    return notYetImplemented('any');
  }

  squaredDifference(a: Tensor, b: Tensor): Tensor {
    return notYetImplemented('squaredDifference');
  }

  ceil<T extends Tensor>(x: T): T {
    return notYetImplemented('ceil');
  }
  floor<T extends Tensor>(x: T): T {
    return notYetImplemented('floor');
  }
  round<T extends Tensor>(x: T): T {
    return notYetImplemented('round');
  }

  sign<T extends Tensor>(x: T): T {
    return notYetImplemented('sign');
  }

  isNaN<T extends Tensor>(x: T): T {
    return notYetImplemented('isNaN');
  }
  isInf<T extends Tensor>(x: T): T {
    return notYetImplemented('isInf');
  }
  isFinite<T extends Tensor>(x: T): T {
    return notYetImplemented('isFinite');
  }

  pow<T extends Tensor>(a: T, b: Tensor): T {
    return notYetImplemented('pow');
  }
  exp<T extends Tensor>(x: T): T {
    return notYetImplemented('exp');
  }
  expm1<T extends Tensor>(x: T): T {
    return notYetImplemented('expm1');
  }
  log<T extends Tensor>(x: T): T {
    return notYetImplemented('log');
  }
  log1p<T extends Tensor>(x: T): T {
    return notYetImplemented('log1p');
  }
  sqrt<T extends Tensor>(x: T): T {
    return notYetImplemented('sqrt');
  }
  rsqrt<T extends Tensor>(x: T): T {
    return notYetImplemented('rsqrt');
  }
  square<T extends Tensor>(x: T): T {
    return notYetImplemented('square');
  }
  reciprocal<T extends Tensor>(x: T): T {
    return notYetImplemented('reciprocal');
  }
  relu<T extends Tensor>(x: T): T {
    return notYetImplemented('relu');
  }
  relu6<T extends Tensor>(x: T): T {
    return notYetImplemented('relu6');
  }
  prelu<T extends Tensor>(x: T, a: T): T {
    return notYetImplemented('prelu');
  }
  elu<T extends Tensor>(x: T): T {
    return notYetImplemented('elu');
  }
  eluDer<T extends Tensor>(dy: T, y: T): T {
    return notYetImplemented('eluDer');
  }
  selu<T extends Tensor>(x: T): T {
    return notYetImplemented('selu');
  }
  int<T extends Tensor>(x: T): T {
    return notYetImplemented('int');
  }

  clip<T extends Tensor>(x: T, min: number, max: number): T {
    return notYetImplemented('clip');
  }

  abs<T extends Tensor>(x: T): T {
    return notYetImplemented('abs');
  }
  complexAbs<T extends Tensor>(x: T): T {
    return notYetImplemented('complexAbs');
  }

  sigmoid<T extends Tensor>(x: T): T {
    return notYetImplemented('sigmoid');
  }

  softplus<T extends Tensor>(x: T): T {
    return notYetImplemented('softplus');
  }

  sin<T extends Tensor>(x: T): T {
    return notYetImplemented('sin');
  }
  cos<T extends Tensor>(x: T): T {
    return notYetImplemented('cos');
  }
  tan<T extends Tensor>(x: T): T {
    return notYetImplemented('tan');
  }

  asin<T extends Tensor>(x: T): T {
    return notYetImplemented('asin');
  }
  acos<T extends Tensor>(x: T): T {
    return notYetImplemented('acos');
  }
  atan<T extends Tensor>(x: T): T {
    return notYetImplemented('atan');
  }
  atan2<T extends Tensor>(a: T, b: T): T {
    return notYetImplemented('atan2');
  }

  sinh<T extends Tensor>(x: T): T {
    return notYetImplemented('sinh');
  }
  cosh<T extends Tensor>(x: T): T {
    return notYetImplemented('cosh');
  }
  tanh<T extends Tensor>(x: T): T {
    return notYetImplemented('tanh');
  }

  asinh<T extends Tensor>(x: T): T {
    return notYetImplemented('asinh');
  }
  acosh<T extends Tensor>(x: T): T {
    return notYetImplemented('acosh');
  }
  atanh<T extends Tensor>(x: T): T {
    return notYetImplemented('atanh');
  }

  erf<T extends Tensor>(x: T): T {
    return notYetImplemented('erf');
  }

  step<T extends Tensor>(x: T, alpha: number): T {
    return notYetImplemented('step');
  }

  fusedConv2d(
      {input, filter, convInfo, bias, activation, preluActivationWeights}:
          FusedConv2DConfig): Tensor4D {
    return notYetImplemented('fusedConv2d');
  }

  conv2d(x: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    return notYetImplemented('conv2d');
  }
  conv2dDerInput(dy: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    return notYetImplemented('conv2dDerInput');
  }
  conv2dDerFilter(x: Tensor4D, dY: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    return notYetImplemented('conv2dDerFilter');
  }

  fusedDepthwiseConv2D(
      {input, filter, convInfo, bias, activation, preluActivationWeights}:
          FusedConv2DConfig): Tensor4D {
    return notYetImplemented('fusedDepthwiseConv2D');
  }

  depthwiseConv2D(input: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    return notYetImplemented('depthwiseConv2D');
  }
  depthwiseConv2DDerInput(dy: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    return notYetImplemented('depthwiseConv2DDerInput');
  }
  depthwiseConv2DDerFilter(x: Tensor4D, dY: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    return notYetImplemented('depthwiseConv2DDerFilter');
  }
  conv3d(x: Tensor5D, filter: Tensor5D, convInfo: Conv3DInfo): Tensor5D {
    return notYetImplemented('conv3d');
  }
  conv3dDerInput(dy: Tensor5D, filter: Tensor5D, convInfo: Conv3DInfo):
      Tensor5D {
    return notYetImplemented('conv3dDerInput');
  }
  conv3dDerFilter(x: Tensor5D, dY: Tensor5D, convInfo: Conv3DInfo): Tensor5D {
    return notYetImplemented('conv3dDerFilter');
  }
  maxPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    return notYetImplemented('maxPool');
  }
  maxPoolBackprop(dy: Tensor4D, x: Tensor4D, y: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    return notYetImplemented('maxPoolBackprop');
  }
  avgPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    return notYetImplemented('avgPool');
  }
  avgPoolBackprop(dy: Tensor4D, x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    return notYetImplemented('avgPoolBackprop');
  }
  avgPool3d(x: Tensor5D, convInfo: Conv3DInfo): Tensor5D {
    return notYetImplemented('avgPool3d');
  }
  avgPool3dBackprop(dy: Tensor5D, x: Tensor5D, convInfo: Conv3DInfo): Tensor5D {
    return notYetImplemented('avgPool3dBackprop');
  }
  maxPool3d(x: Tensor5D, convInfo: Conv3DInfo): Tensor5D {
    return notYetImplemented('maxPool3d');
  }
  maxPool3dBackprop(
      dy: Tensor5D, x: Tensor5D, y: Tensor5D, convInfo: Conv3DInfo): Tensor5D {
    return notYetImplemented('maxPool3dBackprop');
  }

  reshape<T extends Tensor, R extends Rank>(x: T, shape: ShapeMap[R]):
      Tensor<R> {
    return notYetImplemented('reshape');
  }
  cast<T extends Tensor>(x: T, dtype: DataType): T {
    return notYetImplemented('cast');
  }

  tile<T extends Tensor>(x: T, reps: number[]): T {
    return notYetImplemented('tile');
  }

  pad<T extends Tensor>(
      x: T, paddings: Array<[number, number]>, constantValue: number): T {
    return notYetImplemented('pad');
  }

  transpose<T extends Tensor>(x: T, perm: number[]): T {
    return notYetImplemented('transpose');
  }

  gather<T extends Tensor>(x: T, indices: Tensor1D, axis: number): T {
    return notYetImplemented('gather');
  }

  gatherND(x: Tensor, indices: Tensor): Tensor {
    return notYetImplemented('gatherND');
  }

  scatterND<R extends Rank>(
      indices: Tensor, updates: Tensor, shape: ShapeMap[R]): Tensor<R> {
    return notYetImplemented('scatterND');
  }

  batchToSpaceND<T extends Tensor>(
      x: T, blockShape: number[], crops: number[][]): T {
    return notYetImplemented('batchToSpaceND');
  }

  spaceToBatchND<T extends Tensor>(
      x: T, blockShape: number[], paddings: number[][]): T {
    return notYetImplemented('spaceToBatchND');
  }

  resizeBilinear(
      x: Tensor4D, newHeight: number, newWidth: number,
      alignCorners: boolean): Tensor4D {
    return notYetImplemented('resizeBilinear');
  }

  resizeBilinearBackprop(dy: Tensor4D, x: Tensor4D, alignCorners: boolean):
      Tensor4D {
    return notYetImplemented('resizeBilinearBackprop');
  }

  resizeNearestNeighbor(
      x: Tensor4D, newHEight: number, newWidth: number,
      alignCorners: boolean): Tensor4D {
    return notYetImplemented('resizeNearestNeighbor');
  }

  resizeNearestNeighborBackprop(
      dy: Tensor4D, x: Tensor4D, alignCorners: boolean): Tensor4D {
    return notYetImplemented('resizeNearestNeighborBackprop');
  }

  batchNormalization(
      x: Tensor4D, mean: Tensor4D|Tensor1D, variance: Tensor4D|Tensor1D,
      varianceEpsilon: number, scale?: Tensor4D|Tensor1D,
      offset?: Tensor4D|Tensor1D): Tensor4D {
    return notYetImplemented('batchNormalization');
  }

  localResponseNormalization4D(
      x: Tensor4D, radius: number, bias: number, alpha: number,
      beta: number): Tensor4D {
    return notYetImplemented('localResponseNormalization4D');
  }

  LRNGrad(
      dy: Tensor4D, inputImage: Tensor4D, outputImage: Tensor4D, radius: number,
      bias: number, alpha: number, beta: number): Tensor4D {
    return notYetImplemented('LRNGrad');
  }

  multinomial(
      logits: Tensor2D, normalized: boolean, numSamples: number,
      seed: number): Tensor2D {
    return notYetImplemented('multinomial');
  }

  oneHot(indices: Tensor1D, depth: number, onValue: number, offValue: number):
      Tensor2D {
    return notYetImplemented('oneHot');
  }

  cumsum(x: Tensor, axis: number, exclusive: boolean, reverse: boolean):
      Tensor {
    return notYetImplemented('cumsum');
  }

  nonMaxSuppression(
      boxes: Tensor2D, scores: Tensor1D, maxOutputSize: number,
      iouThreshold: number, scoreThreshold?: number): Tensor1D {
    return notYetImplemented('nonMaxSuppression');
  }

  fft(x: Tensor2D): Tensor2D {
    return notYetImplemented('fft');
  }
  ifft(x: Tensor2D): Tensor2D {
    return notYetImplemented('ifft');
  }
  complex<T extends Tensor>(real: T, imag: T): T {
    return notYetImplemented('complex');
  }
  real<T extends Tensor>(input: T): T {
    return notYetImplemented('real');
  }
  imag<T extends Tensor>(input: T): T {
    return notYetImplemented('imag');
  }

  cropAndResize(
      image: Tensor4D, boxes: Tensor2D, boxIndex: Tensor1D,
      cropSize: [number, number], method: 'bilinear'|'nearest',
      extrapolationValue: number): Tensor4D {
    return notYetImplemented('cropAndResize');
  }

  depthToSpace(x: Tensor4D, blockSize: number, dataFormat: string): Tensor4D {
    return notYetImplemented('depthToSpace');
  }

  // Aligns with the "SplitV" kernel in TensorFlow.
  split<T extends Tensor>(value: T, sizeSplits: number[], axis: number): T[] {
    return notYetImplemented('split');
  }

  sparseToDense<R extends Rank>(
      sparseIndices: Tensor, sparseValues: Tensor, outputShape: ShapeMap[R],
      defaultValue: Scalar): Tensor<R> {
    return notYetImplemented('sparseToDense');
  }

  diag(x: Tensor): Tensor {
    return notYetImplemented('diag');
  }

  fill<R extends Rank>(
      shape: ShapeMap[R], value: number|string, dtype?: DataType): Tensor<R> {
    return notYetImplemented('fill');
  }

  onesLike<R extends Rank>(x: Tensor<R>): Tensor<R> {
    return notYetImplemented('onesLike');
  }

  zerosLike<R extends Rank>(x: Tensor<R>): Tensor<R> {
    return notYetImplemented('zerosLike');
  }

  linspace(start: number, stop: number, num: number): Tensor1D {
    return notYetImplemented('linspace');
  }

  dispose(): void {
    return notYetImplemented('dispose');
  }
}

function notYetImplemented(kernelName: string): never {
  throw new Error(
      `'${kernelName}' not yet implemented or not found in the registry. ` +
      `Did you forget to import the kernel?`);
}
