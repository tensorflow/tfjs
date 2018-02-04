
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
import {Array1D, Array2D, Array3D, Array4D, NDArray} from '../ndarray';
import {DataType, Rank, TypedArray} from '../types';

import {MatrixOrientation} from './types/matmul';

export interface NDArrayStorage {
  read(dataId: number): Promise<TypedArray>;
  readSync(dataId: number): TypedArray;
  disposeData(dataId: number): void;
  write(dataId: number, values: TypedArray): void;
  fromPixels(
      pixels: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      numChannels: number): Array3D;
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
export interface MathBackend extends NDArrayStorage, BackendTimer {
  matMul(
      a: Array2D, b: Array2D, aOrientation: MatrixOrientation,
      bOrientation: MatrixOrientation): Array2D;

  slice1D(x: Array1D, begin: number, size: number): Array1D;
  slice2D(x: Array2D, begin: [number, number], size: [number, number]): Array2D;
  slice3D(x: Array3D, begin: [number, number, number], size: [
    number, number, number
  ]): Array3D;
  slice4D(x: Array4D, begin: [number, number, number, number], size: [
    number, number, number, number
  ]): Array4D;

  reverse4D(a: Array4D, axis: number[]): Array4D;

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
  concat(a: Array2D, b: Array2D): Array2D;

  neg<T extends NDArray>(a: T): T;

  add(a: NDArray, b: NDArray): NDArray;
  subtract(a: NDArray, b: NDArray): NDArray;
  multiply(a: NDArray, b: NDArray): NDArray;
  divide(a: NDArray, b: NDArray): NDArray;

  sum(x: NDArray, axes: number[]): NDArray;

  argMin(x: NDArray, axes: number[]): NDArray;
  argMax(x: NDArray, axes: number[]): NDArray;

  equal(a: NDArray, b: NDArray): NDArray;
  notEqual(a: NDArray, b: NDArray): NDArray;

  less(a: NDArray, b: NDArray): NDArray;
  lessEqual(a: NDArray, b: NDArray): NDArray;

  greater(a: NDArray, b: NDArray): NDArray;
  greaterEqual(a: NDArray, b: NDArray): NDArray;

  logicalAnd(a: NDArray, b: NDArray): NDArray;
  logicalOr(a: NDArray, b: NDArray): NDArray;
  logicalXor(a: NDArray, b: NDArray): NDArray;

  where(condition: NDArray, a: NDArray, b: NDArray, dtype: DataType): NDArray;

  topKValues<T extends NDArray>(x: T, k: number): Array1D;
  topKIndices(x: NDArray, k: number): Array1D;

  min(x: NDArray, axes: number[]): NDArray;
  minimum(a: NDArray, b: NDArray): NDArray;

  max(x: NDArray, axes: number[]): NDArray;
  maximum(a: NDArray, b: NDArray): NDArray;

  ceil<T extends NDArray>(x: T): T;
  floor<T extends NDArray>(x: T): T;

  pow<T extends NDArray>(a: T, b: NDArray): T;
  exp<T extends NDArray>(x: T): T;
  log<T extends NDArray>(x: T): T;
  sqrt<T extends NDArray>(x: T): T;

  square<T extends NDArray>(x: T): T;

  relu<T extends NDArray>(x: T): T;
  elu<T extends NDArray>(x: T): T;
  eluDer<T extends NDArray>(x: T): T;
  selu<T extends NDArray>(x: T): T;
  leakyRelu<T extends NDArray>(x: T, alpha: number): T;
  prelu<T extends NDArray>(x: T, alpha: T): T;
  preluDer<T extends NDArray>(x: T, alpha: T): T;
  int<R extends Rank>(x: NDArray<R>): NDArray<R>;

  clip<T extends NDArray>(x: T, min: number, max: number): T;

  abs<T extends NDArray>(x: T): T;

  sigmoid<T extends NDArray>(x: T): T;

  sin<T extends NDArray>(x: T): T;
  cos<T extends NDArray>(x: T): T;
  tan<T extends NDArray>(x: T): T;

  asin<T extends NDArray>(x: T): T;
  acos<T extends NDArray>(x: T): T;
  atan<T extends NDArray>(x: T): T;

  sinh<T extends NDArray>(x: T): T;
  cosh<T extends NDArray>(x: T): T;
  tanh<T extends NDArray>(x: T): T;

  step<T extends NDArray>(x: T, alpha: number): T;

  conv2d(x: Array4D, filter: Array4D, bias: Array1D|null, convInfo: Conv2DInfo):
      Array4D;
  conv2dDerInput(dy: Array4D, filter: Array4D, convInfo: Conv2DInfo): Array4D;
  conv2dDerFilter(x: Array4D, dY: Array4D, convInfo: Conv2DInfo): Array4D;
  conv2dDerBias(dY: Array4D): Array1D;

  depthwiseConv2D(input: Array4D, filter: Array4D, convInfo: Conv2DInfo):
      Array4D;

  maxPool(x: Array4D, convInfo: Conv2DInfo): Array4D;
  maxPoolBackprop(dy: Array4D, x: Array4D, convInfo: Conv2DInfo): Array4D;

  minPool(x: Array4D, convInfo: Conv2DInfo): Array4D;
  avgPool(x: Array4D, convInfo: Conv2DInfo): Array4D;
  avgPoolBackprop(dy: Array4D, x: Array4D, convInfo: Conv2DInfo): Array4D;

  tile<T extends NDArray>(x: T, reps: number[]): T;

  pad1D(x: Array1D, paddings: [number, number], constantValue: number): Array1D;
  pad2D(
      x: Array2D, paddings: [[number, number], [number, number]],
      constantValue: number): Array2D;

  transpose<T extends NDArray>(x: T, perm: number[]): T;

  gather<T extends NDArray>(x: T, indices: Array1D, axis: number): T;

  resizeBilinear(
      x: Array4D, newHeight: number, newWidth: number,
      alignCorners: boolean): Array4D;

  batchNormalization2D(
      x: Array2D, mean: Array2D|Array1D, variance: Array2D|Array1D,
      varianceEpsilon: number, scale?: Array2D|Array1D,
      offset?: Array2D|Array1D): Array2D;
  batchNormalization3D(
      x: Array3D, mean: Array3D|Array1D, variance: Array3D|Array1D,
      varianceEpsilon: number, scale?: Array3D|Array1D,
      offset?: Array3D|Array1D): Array3D;
  batchNormalization4D(
      x: Array4D, mean: Array4D|Array1D, variance: Array4D|Array1D,
      varianceEpsilon: number, scale?: Array4D|Array1D,
      offset?: Array4D|Array1D): Array4D;

  localResponseNormalization4D(
      x: Array4D, radius: number, bias: number, alpha: number, beta: number,
      normRegion: 'acrossChannels'|'withinChannel'): Array4D;

  multinomial(probabilities: Array2D, numSamples: number, seed: number):
      Array2D;

  oneHot(indices: Array1D, depth: number, onValue: number, offValue: number):
      Array2D;

  dispose(): void;
}
