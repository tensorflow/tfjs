
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
import {Array1D, Array2D, Array3D, Array4D, DataTypes, NDArray} from '../ndarray';
import {SumTypes} from '../types';
import {MatrixOrientation} from './types/matmul';

export interface NDArrayStorage {
  read<T extends keyof DataTypes>(id: number): Promise<DataTypes[T]>;
  readSync<T extends keyof DataTypes>(id: number): DataTypes[T];
  disposeData(id: number): void;
  write<T extends keyof DataTypes>(
      id: number, values: DataTypes[T], dtype: T, shape: number[]): void;
  writePixels(
      id: number,
      pixels: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      numChannels: number): void;
  time(query: () => NDArray): Promise<number>;
}

/**
 * The interface that defines the kernels that should be implemented when
 * adding a new backend. New backends don't need to implement every one of the
 * methods, this can be done gradually (throw an error for unimplemented
 * methods).
 */
export interface MathBackend extends NDArrayStorage {
  matMul(
      a: Array2D, b: Array2D, aOrientation: MatrixOrientation,
      bOrientation: MatrixOrientation): Array2D;

  clone<T extends NDArray>(ndarray: T): T;

  slice1D(x: Array1D, begin: number, size: number): Array1D;
  slice2D(x: Array2D, begin: [number, number], size: [number, number]): Array2D;
  slice3D(x: Array3D, begin: [number, number, number], size: [
    number, number, number
  ]): Array3D;
  slice4D(x: Array4D, begin: [number, number, number, number], size: [
    number, number, number, number
  ]): Array4D;

  concat1D(a: Array1D, b: Array1D): Array1D;
  concat2D(a: Array2D, b: Array2D, axis: number): Array2D;
  concat3D(a: Array3D, b: Array3D, axis: number): Array3D;
  concat4D(a: Array4D, b: Array4D, axis: number): Array4D;

  neg<T extends NDArray>(a: T): T;

  add<G extends keyof DataTypes>(a: NDArray<G>, b: NDArray<G>): NDArray<G>;
  subtract<G extends keyof DataTypes>(a: NDArray<G>, b: NDArray<G>): NDArray<G>;
  multiply<G extends keyof DataTypes>(a: NDArray<G>, b: NDArray<G>): NDArray<G>;
  divide<G extends keyof DataTypes>(a: NDArray<G>, b: NDArray<G>):
      NDArray<'float32'>;

  sum<T extends keyof DataTypes>(x: NDArray<T>, axes: number[]):
      NDArray<SumTypes[T]>;

  argMin(x: NDArray, axes: number[]): NDArray<'int32'>;
  argMax(x: NDArray, axes: number[]): NDArray<'int32'>;

  equal(a: NDArray, b: NDArray): NDArray<'bool'>;

  topKValues<D extends keyof DataTypes, T extends NDArray<D>>(x: T, k: number):
      Array1D<D>;
  topKIndices(x: NDArray, k: number): Array1D<'int32'>;

  min<G extends keyof DataTypes>(x: NDArray<G>, axes: number[]): NDArray<G>;
  max<G extends keyof DataTypes>(x: NDArray<G>, axes: number[]): NDArray<G>;

  ceil<T extends NDArray>(x: T): T;

  floor<T extends NDArray>(x: T): T;

  pow<T extends NDArray>(a: T, b: NDArray<'int32'>): T;
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

  tile<D extends keyof DataTypes, T extends NDArray<D>>(x: T, reps: number[]):
      T;

  transpose<D extends keyof DataTypes, T extends NDArray<D>>(
      x: T, perm: number[]): T;

  resizeBilinear3D(
      x: Array3D, newShape2D: [number, number], alignCorners: boolean): Array3D;

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

  multinomial(probabilities: Array2D, numSamples: number, seed: number):
      Array2D<'int32'>;

  oneHot(indices: Array1D, depth: number, onValue: number, offValue: number):
      Array2D;

  dispose(): void;
}
