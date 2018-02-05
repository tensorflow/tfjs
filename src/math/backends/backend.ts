
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

import {Conv2DInfo} from '../conv_util';
// tslint:disable-next-line:max-line-length
import {Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D} from '../tensor';
import {DataType, Rank, TypedArray} from '../types';

import {MatrixOrientation} from './types/matmul';

export interface TensorStorage {
  read(dataId: number): Promise<TypedArray>;
  readSync(dataId: number): TypedArray;
  disposeData(dataId: number): void;
  write(dataId: number, values: TypedArray): void;
  fromPixels(
      pixels: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      numChannels: number): Tensor3D;
  time(query: () => void): Promise<number>;
  register(dataId: number, shape: number[], dtype: DataType): void;
}

export interface BackendTimer { time(f: () => void): Promise<number>; }

/**
 * The interface that defines the kernels that should be implemented when
 * adding a new backend. New backends don't need to implement every one of the
 * methods, this can be done gradually (throw an error for unimplemented
 * methods).
 */
export interface MathBackend extends TensorStorage, BackendTimer {
  matMul(
      a: Tensor2D, b: Tensor2D, aOrientation: MatrixOrientation,
      bOrientation: MatrixOrientation): Tensor2D;

  slice1D(x: Tensor1D, begin: number, size: number): Tensor1D;
  slice2D(x: Tensor2D, begin: [number, number], size: [number, number]):
      Tensor2D;
  slice3D(x: Tensor3D, begin: [number, number, number], size: [
    number, number, number
  ]): Tensor3D;
  slice4D(x: Tensor4D, begin: [number, number, number, number], size: [
    number, number, number, number
  ]): Tensor4D;

  reverse4D(a: Tensor4D, axis: number[]): Tensor4D;

  // Any concat of n-dimensional tensors across any axis can be reduced to
  // a concatenation of two-dimensional tensors across the axis 1 by first
  // partitioning the axes of the original tensors into those less than the axis
  // to be concatenated across and the rest. Then reshape the tensors into a
  // two-dimensional tensor by collapsing these two sets of axes and concatenate
  // the resulting matrices across the axis 1, finally reshaping the result to
  // have the proper shape.
  // This method always take a rank-2 tensor (i.e a matrix) and concatenate it
  // along the axis 1 ("putting them next to each other" as opposed to
  // "putting them on top of one another").
  concat(a: Tensor2D, b: Tensor2D): Tensor2D;

  neg<T extends Tensor>(a: T): T;

  add(a: Tensor, b: Tensor): Tensor;
  subtract(a: Tensor, b: Tensor): Tensor;
  multiply(a: Tensor, b: Tensor): Tensor;
  divide(a: Tensor, b: Tensor): Tensor;

  sum(x: Tensor, axes: number[]): Tensor;

  argMin(x: Tensor, axes: number[]): Tensor;
  argMax(x: Tensor, axes: number[]): Tensor;

  equal(a: Tensor, b: Tensor): Tensor;
  notEqual(a: Tensor, b: Tensor): Tensor;

  less(a: Tensor, b: Tensor): Tensor;
  lessEqual(a: Tensor, b: Tensor): Tensor;

  greater(a: Tensor, b: Tensor): Tensor;
  greaterEqual(a: Tensor, b: Tensor): Tensor;

  logicalNot(a: Tensor): Tensor;
  logicalAnd(a: Tensor, b: Tensor): Tensor;
  logicalOr(a: Tensor, b: Tensor): Tensor;
  logicalXor(a: Tensor, b: Tensor): Tensor;

  where(condition: Tensor, a: Tensor, b: Tensor, dtype: DataType): Tensor;

  topKValues<T extends Tensor>(x: T, k: number): Tensor1D;
  topKIndices(x: Tensor, k: number): Tensor1D;

  min(x: Tensor, axes: number[]): Tensor;
  minimum(a: Tensor, b: Tensor): Tensor;

  max(x: Tensor, axes: number[]): Tensor;
  maximum(a: Tensor, b: Tensor): Tensor;

  ceil<T extends Tensor>(x: T): T;
  floor<T extends Tensor>(x: T): T;

  pow<T extends Tensor>(a: T, b: Tensor): T;
  exp<T extends Tensor>(x: T): T;
  log<T extends Tensor>(x: T): T;
  sqrt<T extends Tensor>(x: T): T;

  square<T extends Tensor>(x: T): T;

  relu<T extends Tensor>(x: T): T;
  elu<T extends Tensor>(x: T): T;
  eluDer<T extends Tensor>(x: T): T;
  selu<T extends Tensor>(x: T): T;
  leakyRelu<T extends Tensor>(x: T, alpha: number): T;
  prelu<T extends Tensor>(x: T, alpha: T): T;
  preluDer<T extends Tensor>(x: T, alpha: T): T;
  int<R extends Rank>(x: Tensor<R>): Tensor<R>;

  clip<T extends Tensor>(x: T, min: number, max: number): T;

  abs<T extends Tensor>(x: T): T;

  sigmoid<T extends Tensor>(x: T): T;

  sin<T extends Tensor>(x: T): T;
  cos<T extends Tensor>(x: T): T;
  tan<T extends Tensor>(x: T): T;

  asin<T extends Tensor>(x: T): T;
  acos<T extends Tensor>(x: T): T;
  atan<T extends Tensor>(x: T): T;

  sinh<T extends Tensor>(x: T): T;
  cosh<T extends Tensor>(x: T): T;
  tanh<T extends Tensor>(x: T): T;

  step<T extends Tensor>(x: T, alpha: number): T;

  conv2d(
      x: Tensor4D, filter: Tensor4D, bias: Tensor1D|null,
      convInfo: Conv2DInfo): Tensor4D;
  conv2dDerInput(dy: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D;
  conv2dDerFilter(x: Tensor4D, dY: Tensor4D, convInfo: Conv2DInfo): Tensor4D;
  conv2dDerBias(dY: Tensor4D): Tensor1D;

  depthwiseConv2D(input: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D;

  maxPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D;
  maxPoolBackprop(dy: Tensor4D, x: Tensor4D, convInfo: Conv2DInfo): Tensor4D;

  minPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D;
  avgPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D;
  avgPoolBackprop(dy: Tensor4D, x: Tensor4D, convInfo: Conv2DInfo): Tensor4D;

  tile<T extends Tensor>(x: T, reps: number[]): T;

  pad1D(x: Tensor1D, paddings: [number, number], constantValue: number):
      Tensor1D;
  pad2D(
      x: Tensor2D, paddings: [[number, number], [number, number]],
      constantValue: number): Tensor2D;

  transpose<T extends Tensor>(x: T, perm: number[]): T;

  gather<T extends Tensor>(x: T, indices: Tensor1D, axis: number): T;

  resizeBilinear(
      x: Tensor4D, newHeight: number, newWidth: number,
      alignCorners: boolean): Tensor4D;

  batchNormalization2D(
      x: Tensor2D, mean: Tensor2D|Tensor1D, variance: Tensor2D|Tensor1D,
      varianceEpsilon: number, scale?: Tensor2D|Tensor1D,
      offset?: Tensor2D|Tensor1D): Tensor2D;
  batchNormalization3D(
      x: Tensor3D, mean: Tensor3D|Tensor1D, variance: Tensor3D|Tensor1D,
      varianceEpsilon: number, scale?: Tensor3D|Tensor1D,
      offset?: Tensor3D|Tensor1D): Tensor3D;
  batchNormalization4D(
      x: Tensor4D, mean: Tensor4D|Tensor1D, variance: Tensor4D|Tensor1D,
      varianceEpsilon: number, scale?: Tensor4D|Tensor1D,
      offset?: Tensor4D|Tensor1D): Tensor4D;

  localResponseNormalization4D(
      x: Tensor4D, radius: number, bias: number, alpha: number, beta: number,
      normRegion: 'acrossChannels'|'withinChannel'): Tensor4D;

  multinomial(probabilities: Tensor2D, numSamples: number, seed: number):
      Tensor2D;

  oneHot(indices: Tensor1D, depth: number, onValue: number, offValue: number):
      Tensor2D;

  dispose(): void;
}
