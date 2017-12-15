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

import {Backends, ENV} from '../environment';
import * as util from '../util';

import * as axis_util from './axis_util';
// tslint:disable-next-line:max-line-length
import {BACKEND_REGISTRY, NDArrayStorage} from './backends/backend';
import {MathBackend} from './backends/backend';
import {BackendEngine} from './backends/backend_engine';
import {MatrixOrientation} from './backends/types/matmul';
import * as broadcast_util from './broadcast_util';
import * as concat_util from './concat_util';
import * as conv_util from './conv_util';
// tslint:disable-next-line:max-line-length
import {Array1D, Array2D, Array3D, Array4D, DataTypes, NDArray, Scalar} from './ndarray';
import * as slice_util from './slice_util';
import {SumTypes} from './types';

export type ScopeResultImmediate =
    void|NDArray|NDArray[]|{[key: string]: NDArray};
export type ScopeResult = ScopeResultImmediate|Promise<ScopeResultImmediate>;

export interface LSTMCell {
  (data: Array2D, c: Array2D, h: Array2D): [Array2D, Array2D];
}

export interface NDArrayManager {
  getNumArrays(): number;
  register<T extends keyof DataTypes>(a: NDArray<T>): void;
}

export class NDArrayMath implements NDArrayStorage, NDArrayManager {
  protected backendEngine: BackendEngine;
  private numArrays = 0;
  private backend: MathBackend;
  private customBackend = false;

  getNumArrays() {
    return this.numArrays;
  }

  register<T extends keyof DataTypes>(a: NDArray<T>): void {
    this.track(a);
    this.numArrays++;
  }

  writePixels(
      id: number,
      pixels: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      numChannels: number): void {
    this.backend.writePixels(id, pixels, numChannels);
  }
  write<T extends keyof DataTypes>(
      id: number, values: DataTypes[T], dtype: T, shape: number[]): void {
    this.backend.write(id, values, dtype, shape);
  }
  readSync<T extends keyof DataTypes>(id: number): DataTypes[T] {
    return this.backend.readSync(id);
  }
  read<T extends keyof DataTypes>(id: number): Promise<DataTypes[T]> {
    return this.backend.read(id);
  }

  private ndarrayScopes: NDArray[][] = [];
  private activeScope: NDArray[];

  private ndarraysToKeep: NDArray[][] = [];
  private activeScopeNDArraysToKeep: NDArray[] = [];

  /**
   * @param safeMode In safe mode, you must use math operations inside
   *     a math.scope() which will automatically clean up intermediate NDArrays.
   */
  constructor(backend: Backends|MathBackend, private safeMode: boolean) {
    if (typeof backend === 'string') {
      this.backend = BACKEND_REGISTRY[backend];
    } else {
      this.customBackend = true;
      this.backend = backend;
    }
    this.backendEngine = new BackendEngine(this.backend);
    ENV.setGlobalMath(this);
  }

  /**
   * Create a new math scope. Put chained math operations inside a scope
   * function closure so that the library automatically cleans up NDArrays
   * from intermediate math operations. You must create a scope in safe mode
   * to call math operations. If a result is returned from the scope, it will
   * also be tracked, which means there must be yet another wrapping scope.
   * @param scopeFn The function to execute with chained math operations.
   */
  scope<T extends ScopeResult>(
      scopeFn:
          (keep: <D1 extends keyof DataTypes, T1 extends NDArray<D1>>(
               ndarray: T1) => T1,
           track: <D2 extends keyof DataTypes, T2 extends NDArray<D2>>(
               ndarray: T2) => T2) => T): T {
    this.startScope();

    const keepFn = <T extends NDArray>(ndarray: T): T => this.keep(ndarray);
    const trackFn = <T extends NDArray>(ndarray: T): T => this.track(ndarray);
    const result = scopeFn(keepFn, trackFn);

    if (result instanceof Promise) {
      result.then(r => this.endScope(r));
      return result;
    } else {
      this.endScope(result as ScopeResultImmediate);
      return result;
    }
  }

  /**
   * In debug mode, the output of every math call will be downloaded to the CPU
   * and checked for NaNs. This significantly impacts performance.
   */
  enableDebugMode() {
    this.backendEngine.enableDebugMode();
    console.warn(
        'Debugging mode is ON. The output of every math call will ' +
        'be downloaded to CPU and checked for NaNs. ' +
        'This significantly impacts performance.');
  }

  /**
   * Start a scope. Use this with endScope() to achieve the same functionality
   * as scope() without the need for a function closure.
   */
  startScope() {
    const newScope: NDArray[] = [];
    this.ndarrayScopes.push(newScope);
    this.activeScope = newScope;

    const newNDArraysToKeep: NDArray[] = [];
    this.ndarraysToKeep.push(newNDArraysToKeep);
    this.activeScopeNDArraysToKeep = newNDArraysToKeep;
  }

  private extractNDArraysFromScopeResult(result: ScopeResultImmediate):
      NDArray[] {
    if (result == null) {
      return [];
    }
    if (result instanceof NDArray) {
      return [result];
    }

    const list: NDArray[] = [];
    const resultObj = result as {[key: string]: NDArray};
    // Iteration over keys works also for arrays.
    for (const k in resultObj) {
      const val = resultObj[k];
      if (val instanceof NDArray) {
        list.push(val);
      }
    }
    return list;
  }

  /**
   * End a scope. Use this with startScope() to achieve the same functionality
   * as scope() without the need for a function closure.
   */
  endScope(result: ScopeResultImmediate) {
    let arraysToKeep = this.activeScopeNDArraysToKeep;
    const resultArrays = this.extractNDArraysFromScopeResult(result);
    arraysToKeep = arraysToKeep.concat(resultArrays);

    // Dispose the current scope.
    for (let i = 0; i < this.activeScope.length; i++) {
      const ndarray = this.activeScope[i];
      if (this.isNDArrayDataInList(ndarray, arraysToKeep)) {
        continue;
      }
      ndarray.dispose();
    }

    // Pop the current scope.
    this.ndarrayScopes.pop();
    this.activeScope = this.ndarrayScopes.length === 0 ?
        null :
        this.ndarrayScopes[this.ndarrayScopes.length - 1];

    // Track the current result in the parent scope.
    resultArrays.forEach(val => {
      if (!this.isNDArrayDataInList(val, this.activeScopeNDArraysToKeep)) {
        this.track(val);
      }
    });

    this.ndarraysToKeep.pop();
    this.activeScopeNDArraysToKeep = this.ndarraysToKeep.length === 0 ?
        null :
        this.ndarraysToKeep[this.ndarraysToKeep.length - 1];
  }

  private isNDArrayDataInList(ndarray: NDArray, ndarrayList: NDArray[]) {
    for (let i = 0; i < ndarrayList.length; i++) {
      if (ndarrayList[i].id === ndarray.id) {
        return true;
      }
    }
    return false;
  }

  /**
   * Keeps an NDArray in the current scope from being disposed automatically.
   * @param result The NDArray to keep from being disposed.
   */
  keep<T extends NDArray>(result: T): T {
    if (this.activeScope == null) {
      if (this.safeMode) {
        throw new Error(
            'You are using math in safe mode. Enclose all ' +
            'math.method() calls inside a scope: ' +
            'math.scope(() => {math.method();...}) to avoid memory ' +
            'leaks.');
      }
      return result;
    }
    this.activeScopeNDArraysToKeep.push(result);
    return result;
  }

  /**
   * Tracks an NDArray in the current scope to be automatically cleaned up
   * when the current scope ends, and returns the value.
   *
   * @param result The NDArray to track in the current scope.
   */
  track<G extends keyof DataTypes, T extends NDArray<G>>(result: T): T {
    if (this.activeScope == null) {
      if (this.safeMode) {
        throw new Error(
            'You are using math in safe mode. Enclose all ' +
            'math.method() calls inside a scope: ' +
            'math.scope(() => {math.method();...}) to avoid memory ' +
            'leaks.');
      }
      return result;
    }
    this.activeScope.push(result);
    return result;
  }

  dispose() {
    if (this.customBackend) {
      this.backend.dispose();
    }
  }

  /**
   * Computes the dot product of two matrices, A * B. These must be matrices,
   * use matrixTimesVector and vectorTimesMatrix, dotProduct, and outerProduct
   * in other cases.
   * @param a First matrix in dot product operation.
   * @param b Second matrix in dot product operation.
   * @param aOrientation The MatrixOrientation of A. If using TRANSPOSED, will
   * compute A^T * B.
   * @param bOrientation The MatrixOrientation of B. If using TRANSPOSED, will
   * compute A * B^T.
   */
  matMul(
      a: Array2D, b: Array2D, aOrientation = MatrixOrientation.REGULAR,
      bOrientation = MatrixOrientation.REGULAR): Array2D {
    const innerShapeA =
        (aOrientation === MatrixOrientation.REGULAR) ? a.shape[1] : a.shape[0];
    const innerShapeB =
        (bOrientation === MatrixOrientation.REGULAR) ? b.shape[0] : b.shape[1];

    util.assert(
        a.rank === 2 && b.rank === 2,
        `Error in matMul: inputs must be rank 2, got ranks ${a.rank}` +
            ` and ${b.rank}.`);

    util.assert(
        innerShapeA === innerShapeB,
        `Error in matMul: inner shapes (${innerShapeA}) and (` +
            `${innerShapeB}) of NDArrays with shapes ${a.shape} and ` +
            `${b.shape} and orientations ${MatrixOrientation[aOrientation]}` +
            ` and ${MatrixOrientation[bOrientation]} must match.`);

    return this.backendEngine.executeKernel(
        'MatMul', {inputs: {a, b}, args: {aOrientation, bOrientation}},
        (dy: Array2D, y: Array2D) => {
          return {
            a: () => this.matMul(
                dy, b, MatrixOrientation.REGULAR, MatrixOrientation.TRANSPOSED),
            b: () => this.matMul(
                a, dy, MatrixOrientation.TRANSPOSED, MatrixOrientation.REGULAR)
          };
        });
  }

  private executeOp<G extends keyof DataTypes, T extends NDArray<G>>(
      name: string, f: () => T): T {
    // TODO(nsthorat): Do operation logging and performance profiling.
    return f();
  }

  /**
   * Computes the dot product of a vector and a matrix, v * B.
   * @param v The vector in dot product operation.
   * @param matrix The matrix in dot product operation.
   */
  vectorTimesMatrix(v: Array1D, matrix: Array2D): Array1D {
    util.assert(
        v.rank === 1,
        `Error in vectorTimesMatrix: first input must be rank 1, but got ` +
            `rank ${v.rank}.`);
    util.assert(
        matrix.rank === 2,
        `Error in vectorTimesMatrix: second input must be rank 2, but got ` +
            `rank ${matrix.rank}.`);
    util.assert(
        v.size === matrix.shape[0],
        `Error in vectorTimesMatrix: size of vector (${v.size}) ` +
            `must match first dimension of matrix (${matrix.shape[0]})`);

    return this.matMul(v.as2D(1, -1), matrix).as1D();
  }

  /**
   * Computes the dot product of a matrix and vector, A * v.
   * @param matrix The matrix in dot product operation.
   * @param v The vector in dot product operation.
   */
  matrixTimesVector(matrix: Array2D, v: Array1D): Array1D {
    util.assert(
        v.rank === 1,
        `Error in matrixTimesVector: second input must rank 1, but got ` +
            `rank ${v.rank}.`);
    util.assert(
        matrix.rank === 2,
        `Error in matrixTimesVector: first input must be a rank 2, but got ` +
            `rank ${matrix.rank}.`);
    util.assert(
        v.size === matrix.shape[1],
        `Error in matrixTimesVector: size of first rank 1 input ${v.size} ` +
            `must match inner dimension of second rank 2 input, but got ` +
            `shape ${matrix.shape}.`);

    return this.matMul(matrix, v.as2D(-1, 1)).as1D();
  }

  /**
   * Computes the dot product of two vectors, v1 * v2.
   * @param v1 The first vector in the dot product operation.
   * @param v2 The second vector in the dot product operation.
   */
  dotProduct(v1: Array1D, v2: Array1D): Scalar {
    util.assert(
        v1.rank === 1 && v2.rank === 1,
        `Error in dotProduct: inputs must be rank 1, but got ranks ` +
            `${v1.rank} and ${v2.rank}.`);
    util.assert(
        v1.size === v2.size,
        `Error in dotProduct: size of inputs (${v1.size}) and (` +
            `${v2.size}) must match.`);
    return this.matMul(v1.as2D(1, -1), v2.as2D(-1, 1)).asScalar();
  }

  /**
   * Computes the outer product of two vectors, v1 and v2.
   * @param v1 The first vector in the outer product operation.
   * @param v2 The second vector in the dot product operation.
   */
  outerProduct(v1: Array1D, v2: Array1D): Array2D {
    util.assert(
        v1.rank === 1 && v2.rank === 1,
        `Error in outerProduct: inputs must be rank 1, but got ranks ` +
            `${v1.rank} and ${v2.rank}.`);

    return this.matMul(v1.as2D(-1, 1), v2.as2D(1, -1));
  }

  ///////////////
  // Shape ops //
  ///////////////

  /**
   * Clones an NDArray of any shape.
   * @param x The NDArray to clone.
   */
  clone<T extends NDArray>(x: T): T {
    return this.backendEngine.executeKernel('Clone', {inputs: {x}}) as T;
  }

  /**
   * @deprecated Please call reshape() directly on the ndarray object.
   */
  reshape<T1 extends NDArray, T2 extends NDArray>(
      ndarray: T1, newShape: number[]): T2 {
    console.warn(
        'math.reshape() is deprecated. Please call reshape() ' +
        'directly on the ndarray object');
    return ndarray.reshape(newShape) as T2;
  }

  /**
   * Extracts a 1D slice from 1D array starting at coordinates `begin` and is
   * of length `size`.
   *
   * @param x The input array to slice from.
   * @param begin The offset to start the slice from.
   * @param size The size of the slice.
   */
  slice1D(x: Array1D, begin: number, size: number): Array1D {
    slice_util.assertParamsValid(x, [begin], [size]);
    return this.backendEngine.executeKernel(
        'Slice1D', {inputs: {x}, args: {begin, size}});
  }

  /**
   * Extracts a 2D slice from a 2D array starting at coordinates `begin` and
   * is of size `size`.
   *
   * @param x The input array to slice from.
   * @param begin The [row, col] 2d coordinates to start the slice from.
   * @param size The size of the slice.
   */
  slice2D(x: Array2D, begin: [number, number], size: [number, number]):
      Array2D {
    slice_util.assertParamsValid(x, begin, size);
    return this.backendEngine.executeKernel(
        'Slice2D', {inputs: {x}, args: {begin, size}});
  }

  /**
   * Extracts a 3D slice from a 3D array starting at coordinates `begin` and
   * is of size `size`.
   *
   * @param x The input array to slice from.
   * @param begin The [row, col, depth] 3d coordinates to start the slice from.
   * @param size The size of the slice.
   */
  slice3D(x: Array3D, begin: [number, number, number], size: [
    number, number, number
  ]): Array3D {
    slice_util.assertParamsValid(x, begin, size);
    return this.backendEngine.executeKernel(
        'Slice3D', {inputs: {x}, args: {begin, size}});
  }

  /**
   * Extracts a 4D slice from a 4D array starting at coordinates `begin` and
   * is of size `size`.
   *
   * @param x The input array to slice from.
   * @param begin The [row, col, depth, depth2] 4d coordinates to start the
   *              slice from.
   * @param size The size of the slice.
   */
  slice4D(x: Array4D, begin: [number, number, number, number], size: [
    number, number, number, number
  ]): Array4D {
    slice_util.assertParamsValid(x, begin, size);
    return this.backendEngine.executeKernel(
        'Slice4D', {inputs: {x}, args: {begin, size}});
  }

  /**
   * Concatenates two 1D arrays.
   *
   * For example, if:
   * A: shape(3) = |r1, g1, b1|
   * B: shape(2) = |r2, g2|
   * C = concat1D(A, B) == |r1, g1, b1, r2, g2|
   *
   * @param a The first array.
   * @param b The second array.
   * @return The concatenated array.
   */
  concat1D(a: Array1D, b: Array1D): Array1D {
    concat_util.assertParams(a.shape, b.shape, 0);
    return this.backendEngine.executeKernel('Concat1D', {inputs: {a, b}});
  }

  /**
   * Concatenates two 2D arrays along a given axis.
   *
   * For example, if:
   * A: shape(2, 3) = | r1, g1, b1 |
   *                  | r2, g2, b2 |
   *
   * B: shape(2, 3) = | r3, g3, b3 |
   *                  | r4, g4, b4 |
   *
   * C = concat2D(A, B, axis)
   *
   * if axis = 0:
   * C: shape(4, 3) = | r1, g1, b1 |
   *                  | r2, g2, b2 |
   *                  | r3, g3, b3 |
   *                  | r4, g4, b4 |
   *
   * if axis = 1:
   * C = shape(2, 6) = | r1, g1, b1, r3, g3, b3 |
   *                   | r2, g2, b2, r4, g4, b4 |
   *
   *
   * @param a The first array.
   * @param b The second array.
   * @param axis The axis to concatenate along.
   * @return The concatenated array.
   */
  concat2D(a: Array2D, b: Array2D, axis: number): Array2D {
    concat_util.assertParams(a.shape, b.shape, axis);
    return this.backendEngine.executeKernel(
        'Concat2D', {inputs: {a, b}, args: {axis}});
  }

  /**
   * Concatenates two 3D ndarrays along a given axis.
   *
   * For example, if:
   * A: shape(2, 1, 3) = | r1, g1, b1 |
   *                     | r2, g2, b2 |
   *
   * B: shape(2, 1, 3) = | r3, g3, b3 |
   *                     | r4, g4, b4 |
   *
   * C = concat3D(A, B, axis)
   *
   * if axis = 0:
   * C: shape(4, 1, 3) = | r1, g1, b1 |
   *                     | r2, g2, b2 |
   *                     | r3, g3, b3 |
   *                     | r4, g4, b4 |
   *
   * if axis = 1:
   * C: shape(2, 2, 3) = | r1, g1, b1, r3, g3, b3 |
   *                     | r2, g2, b2, r4, g4, b4 |
   *
   * if axis = 2:
   * C = shape(2, 1, 6) = | r1, g1, b1, r3, g3, b3 |
   *                      | r2, g2, b2, r4, g4, b4 |
   *
   * @param a The first array to concat.
   * @param b The second array to conat.
   * @param axis The axis to concate along.
   * @return The concatenated array.
   */
  concat3D(a: Array3D, b: Array3D, axis: number): Array3D {
    concat_util.assertParams(a.shape, b.shape, axis);
    return this.backendEngine.executeKernel(
        'Concat3D', {inputs: {a, b}, args: {axis}});
  }

  /**
   * Concatenates two 4D ndarrays along a given axis. See math.concat2D() for
   * documentation.
   *
   * @param a The first array to concat.
   * @param b The second array to conat.
   * @param axis The axis to concate along.
   * @return The concatenated array.
   */
  concat4D(a: Array4D, b: Array4D, axis: number): Array4D {
    concat_util.assertParams(a.shape, b.shape, axis);
    return this.backendEngine.executeKernel(
        'Concat4D', {inputs: {a, b}, args: {axis}});
  }

  ///////////////////
  // Reduction ops //
  ///////////////////

  /**
   * Computes the log(sum(exp(elements across the reduction dimensions)).
   *
   * Reduces the input along the dimensions given in `axis`. Unless `keepDims`
   * is true, the rank of the array is reduced by 1 for each entry in `axis`.
   * If `keepDims` is true, the reduced dimensions are retained with length 1.
   * If `axis` has no entries, all dimensions are reduced, and an array with a
   * single element is returned.
   *
   * @param input The input NDArray.
   * @param axis Optional. The dimension(s) to reduce. If null (the default),
   *     reduces all dimensions.
   * @param keepDims Optional. If true, retains reduced dimensions with length
   *     of 1. Defaults to false.
   */
  logSumExp(input: NDArray, axis: number|number[] = null, keepDims = false):
      NDArray {
    const axes = axis_util.parseAxisParam(axis, input.shape);
    return this.executeOp('logSumExp', () => {
      const xMax = this.max(input, axes, true /* keepDims */);
      const a = this.subtract(input, xMax);
      const b = this.exp(a);
      const c = this.sum(b, axes);
      const d = this.log(c);
      const res = this.add(xMax.reshape(d.shape), d);

      if (keepDims) {
        const newShape = axis_util.expandShapeToKeepDim(res.shape, axes);
        return res.reshape(newShape);
      }
      return res;
    });
  }

  /**
   * Computes the sum of elements across dimensions of an array.
   *
   * Reduces the input along the dimensions given in `axes`. Unless `keepDims`
   * is true, the rank of the array is reduced by 1 for each entry in `axes`.
   * If `keepDims` is true, the reduced dimensions are retained with length 1.
   * If axes has no entries, all dimensions are reduced, and an array with a
   * single element is returned.
   *
   * @param x The input array to compute the sum over.
   * @param axis Optional. The dimension(s) to reduce. By default it reduces
   *     all dimensions.
   * @param keepDims Optional. If true, retains reduced dimensions with size 1.
   */
  sum<T extends keyof DataTypes>(
      x: NDArray<T>, axis: number|number[] = null,
      keepDims = false): NDArray<SumTypes[T]> {
    const origAxes = axis_util.parseAxisParam(axis, x.shape);
    let axes = origAxes;
    const permutedAxes = axis_util.getPermutedAxes(axes, x.rank);
    return this.executeOp('sum', () => {
      if (permutedAxes != null) {
        x = this.transpose(x, permutedAxes);
        axes = axis_util.getInnerMostAxes(axes.length, x.rank);
      }
      const res = this.backendEngine.executeKernel(
          'Sum', {inputs: {x}, args: {axes}},
          (dy: NDArray<SumTypes[T]>, y: NDArray<SumTypes[T]>) => {
            return {
              x: () => {
                // TODO(nsthorat): Fix gradients for sum when using axis
                // reduction.
                if (axis != null) {
                  throw new Error(
                      `Gradients for sum with axis reduction not yet ` +
                      `supported.`);
                }
                return this.multiply(dy, NDArray.onesLike(x));
              }
            };
          });
      if (keepDims) {
        const newShape = axis_util.expandShapeToKeepDim(res.shape, origAxes);
        return res.reshape(newShape);
      }
      return res;
    });
  }

  /**
   * Computes the mean of elements across dimensions of an array.
   *
   * Reduces `x` along the dimensions given in `axis`. Unless `keepDims` is
   * true, the rank of the array is reduced by 1 for each entry in `axis`.
   * If `keepDims` is true, the reduced dimensions are retained with length 1.
   * If `axis` has no entries, all dimensions are reduced, and an array with a
   * single element is returned.
   *
   * @param x The input array.
   * @param axis Optional. The dimension(s) to reduce. By default it reduces
   *     all dimensions.
   * @param keepDims Optional. If true, retains reduced dimensions with size 1.
   */
  mean(x: NDArray, axis: number|number[] = null, keepDims = false):
      NDArray<'float32'> {
    const axes = axis_util.parseAxisParam(axis, x.shape);
    const shapes = axis_util.computeOutAndReduceShapes(x.shape, axes);
    const reduceShape = shapes[1];
    const reduceSize = util.sizeFromShape(reduceShape);
    return this.executeOp('mean', () => {
      return this.scope((keep, track) => {
        const res = this.divide(x, track(Scalar.new(reduceSize)));
        return this.sum(res, axis, keepDims);
      });
    });
  }

  /**
   * Returns the indices of the minimum values along an `axis`. The result has
   * the same shape as `input` with the dimension along `axis` removed.
   *
   * @param x The input array.
   * @param axis Optional. The dimension to reduce. By default it reduces
   * across all axes and returns the flat index.
   *
   */
  argMin(x: NDArray, axis: number = null): NDArray<'int32'> {
    let axes = axis_util.parseAxisParam(axis, x.shape);
    const permutedAxes = axis_util.getPermutedAxes(axes, x.rank);
    return this.executeOp('argMin', () => {
      if (permutedAxes != null) {
        x = this.transpose(x, permutedAxes);
        axes = axis_util.getInnerMostAxes(axes.length, x.rank);
      }
      return this.backendEngine.executeKernel(
          'ArgMin', {inputs: {x}, args: {axes}});
    });
  }

  /**
   * Returns the indices of the maximum values along an `axis`. The result has
   * the same shape as `input` with the dimension along `axis` removed.
   *
   * @param x The input array.
   * @param axis Optional. The dimension to reduce. By default it reduces
   *     across all axes and returns the flat index
   */
  argMax(x: NDArray, axis: number = null): NDArray<'int32'> {
    let axes = axis_util.parseAxisParam(axis, x.shape);
    const permutedAxes = axis_util.getPermutedAxes(axes, x.rank);
    return this.executeOp('argMax', () => {
      if (permutedAxes != null) {
        x = this.transpose(x, permutedAxes);
        axes = axis_util.getInnerMostAxes(axes.length, x.rank);
      }
      return this.backendEngine.executeKernel(
          'ArgMax', {inputs: {x}, args: {axes}});
    });
  }

  /**
   * Returns a 1 if the argMax of x1 and x2 are the same, otherwise 0.
   * @param x1 The first input NDArray.
   * @param x2 The second input NDArray.
   */
  argMaxEquals(x1: NDArray, x2: NDArray): Scalar<'bool'> {
    util.assertShapesMatch(x1.shape, x2.shape, 'Error in argMaxEquals: ');
    return this.executeOp('argMaxEquals', () => this.scope(() => {
      return this.equal(this.argMax(x1), this.argMax(x2));
    }));
  }

  /**
   * Returns the truth value of (a == b) element-wise. Supports broadcasting.
   * For a stricter version without broadcasting use math.equalStrict().
   */
  equal(a: NDArray, b: NDArray): NDArray<'bool'> {
    return this.backendEngine.executeKernel('Equal', {inputs: {a, b}});
  }

  equalStrict<D extends keyof DataTypes, T extends NDArray<D>>(a: T, b: T):
      NDArray<'bool'> {
    util.assertShapesMatch(a.shape, b.shape, 'Error in equalStrict: ');
    return this.equal(a, b);
  }

  /**
   * Computes the top K values and flattened indices.
   * @param x The input NDArray.
   * @param k How many top values to compute.
   */
  topK(x: NDArray, k: number): {values: Array1D, indices: Array1D<'int32'>} {
    util.assert(
        k <= x.size,
        `Error in topK: k value (${k}) must be less than size of input ` +
            `ndarray, got shape ${x.shape}.`);
    let values: Array1D;
    let indices: Array1D<'int32'>;
    this.executeOp('topK', () => {
      values = this.backendEngine.executeKernel(
          'TopKValues', {inputs: {x}, args: {k}});
      indices = this.backendEngine.executeKernel(
          'TopKIndices', {inputs: {x}, args: {k}});
      return values;
    });
    const result = {values, indices};
    return result;
  }

  /**
   * Computes the minimum value from the input.
   *
   * Reduces the input along the dimensions given in `axes`. Unless `keepDims`
   * is true, the rank of the array is reduced by 1 for each entry in `axes`.
   * If `keepDims` is true, the reduced dimensions are retained with length 1.
   * If `axes` has no entries, all dimensions are reduced, and an array with a
   * single element is returned.
   *
   * @param x The input NDArray.
   * @param axis Optional. The dimension(s) to reduce. By default it reduces
   *     all dimensions.
   * @param keepDims Optional. If true, retains reduced dimensions with size 1.
   */
  min<G extends keyof DataTypes>(
      x: NDArray<G>, axis: number|number[] = null,
      keepDims = false): NDArray<G> {
    const origAxes = axis_util.parseAxisParam(axis, x.shape);
    let axes = origAxes;
    const permutedAxes = axis_util.getPermutedAxes(axes, x.rank);
    return this.executeOp('min', () => {
      if (permutedAxes != null) {
        x = this.transpose(x, permutedAxes);
        axes = axis_util.getInnerMostAxes(axes.length, x.rank);
      }
      const res = this.backendEngine.executeKernel(
                      'Min', {inputs: {x}, args: {axes}}) as NDArray<G>;
      if (keepDims) {
        const newShape = axis_util.expandShapeToKeepDim(res.shape, origAxes);
        return res.reshape(newShape);
      }
      return res;
    });
  }

  /**
   * Computes the maximum of elements across dimensions of an array.
   *
   * Reduces the input along the dimensions given in `axes`. Unless `keepDims`
   * is true, the rank of the array is reduced by 1 for each entry in `axes`.
   * If `keepDims` is true, the reduced dimensions are retained with length 1.
   * If `axes` has no entries, all dimensions are reduced, and an array with a
   * single element is returned.
   *
   * @param x The input array.
   * @param axis Optional. The dimension(s) to reduce. By default it reduces
   *     all dimensions.
   * @param keepDims Optional. If true, retains reduced dimensions with size 1.
   */
  max<G extends keyof DataTypes>(
      x: NDArray<G>, axis: number|number[] = null,
      keepDims = false): NDArray<G> {
    const origAxes = axis_util.parseAxisParam(axis, x.shape);
    let axes = origAxes;
    const permutedAxes = axis_util.getPermutedAxes(axes, x.rank);
    return this.executeOp('max', () => {
      if (permutedAxes != null) {
        x = this.transpose(x, permutedAxes);
        axes = axis_util.getInnerMostAxes(axes.length, x.rank);
      }
      const res = this.backendEngine.executeKernel(
                      'Max', {inputs: {x}, args: {axes}}) as NDArray<G>;
      if (keepDims) {
        const newShape = axis_util.expandShapeToKeepDim(res.shape, origAxes);
        return res.reshape(newShape);
      }
      return res;
    });
  }

  /**
   * Computes the softmax normalized vector given the logits.
   * @param logits The logits array.
   * @param dim The dimension softmax would be performed on. Defaults to -1
   *     which indicates the last dimension.
   */
  softmax<T extends NDArray>(logits: T, dim = -1): T {
    if (dim === -1) {
      dim = logits.rank - 1;
    }
    if (dim !== logits.rank - 1) {
      throw Error(
          'Softmax along a non-last dimension is not yet supported. ' +
          `Logits was rank ${logits.rank} and dim was ${dim}`);
    }
    return this.executeOp('softmax', () => {
      return this.scope(() => {
        // Do it in log space for numerical stability.
        // exp(X - logSumExp(X))
        const lse = this.logSumExp(logits, [dim], true /* keepDims */);
        const logResult = this.subtract(logits, lse);
        return this.exp(logResult) as T;
      });
    });
  }

  //////////////////////
  // Element-wise ops //
  //////////////////////

  /** @deprecated Use math.transpose() instead. */
  switchDim<T extends NDArray>(a: T, newDim: number[]): T {
    return this.transpose(a, newDim);
  }

  /**
   * Construct an array by repeating it the number of times given by reps.
   *
   * This operation creates a new array by replicating `input` `reps`
   * times. The output tensor's i'th dimension has `input.shape[i] *
   * reps[i]` elements, and the values of `input` are replicated
   * `reps[i]` times along the i'th dimension. For example, tiling
   * `[a, b, c, d]` by `[2]` produces `[a, b, c, d, a, b, c, d]`.
   *
   * @param x The array to transpose.
   * @param reps Determines the number of replications per dimension.
   */
  tile<D extends keyof DataTypes, T extends NDArray<D>>(x: T, reps: number[]):
      T {
    util.assert(
        x.rank === reps.length,
        `Error in transpose: rank of input ${x.rank} ` +
            `must match length of reps ${reps}.`);
    return this.backendEngine.executeKernel(
               'Tile', {inputs: {x}, args: {reps}}) as T;
  }

  /**
   * Transposes the array. Permutes the dimensions according to `perm`.
   *
   * The returned array's dimension `i` will correspond to the input dimension
   * `perm[i]`. If `perm` is not given, it is set to `[n-1...0]`, where `n` is
   * the rank of the input array. Hence by default, this operation performs a
   * regular matrix transpose on 2-D input arrays.
   *
   * @param x The array to transpose.
   * @param perm Optional. The permutation of the dimensions of a.
   */
  transpose<D extends keyof DataTypes, T extends NDArray<D>>(
      x: T, perm?: number[]): T {
    if (perm == null) {
      perm = x.shape.map((s, i) => i).reverse();
    }
    util.assert(
        x.rank === perm.length,
        `Error in transpose: rank of input ${x.rank} ` +
            `must match length of perm ${perm}.`);
    return this.backendEngine.executeKernel(
               'Transpose', {inputs: {x}, args: {perm}}) as T;
  }

  /** @deprecated Use math.add(c, A) instead. */
  scalarPlusArray<T extends NDArray>(c: Scalar, a: T): T {
    util.assert(
        c.size === 1,
        `Error in scalarPlusArray: first argument must be rank 0, but got ` +
            `rank ${c.rank}.`);
    return this.add(c, a) as T;
  }

  /** @deprecated Use math.sub(c, A) instead. */
  scalarMinusArray<T extends NDArray>(c: Scalar, a: T): T {
    util.assert(
        c.size === 1,
        `Error in scalarMinusArray: first argument must be rank 0, but got ` +
            `rank ${c.rank}.`);
    return this.subtract(c, a) as T;
  }

  /** @deprecated Use math.sub(A, c) instead. */
  arrayMinusScalar<T extends NDArray>(a: T, c: Scalar): T {
    util.assert(
        c.size === 1,
        `Error in arrayMinusScalar: second argument must be rank 0, but ` +
            `got rank ${c.rank}.`);
    return this.subtract(a, c) as T;
  }

  /**
   * Computes -1 * A element-wise.
   * @param x The input array.
   */
  neg<T extends NDArray>(x: T): T {
    return this.backendEngine.executeKernel('Neg', {inputs: {x}}) as T;
  }

  /**
   * Adds two NDArrays element-wise, A + B. Supports broadcasting.
   * For a stricter version without broadcasting use math.addStrict().
   *
   * @param a The first NDArray to add element-wise.
   * @param b The second NDArray to add element-wise.
   */
  add<G extends keyof DataTypes>(a: NDArray<G>, b: NDArray<G>): NDArray<G> {
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return this.backendEngine.executeKernel('Add', {inputs: {a, b}}) as
        NDArray<G>;
  }

  /**
   * Adds two NDArrays element-wise, A + B. Inputs must
   * be the same shape. For broadcasting support, use math.add() instead.
   *
   * @param a The first NDArray to multiply element-wise.
   * @param b The second NDArray to multiply element-wise.
   */
  addStrict<D extends keyof DataTypes, T extends NDArray<D>>(a: T, b: T): T {
    util.assertShapesMatch(a.shape, b.shape, 'Error in addStrict: ');
    return this.add(a, b) as T;
  }

  /**
   * Subtracts two NDArrays element-wise, A - B. Supports broadcasting.
   * For a stricter version without broadcasting use math.subStrict().
   *
   * @param a The first NDArray to subtract element-wise.
   * @param b The second NDArray to subtract element-wise.
   */
  subtract<G extends keyof DataTypes>(a: NDArray<G>, b: NDArray<G>):
      NDArray<G> {
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return this.backendEngine.executeKernel('Sub', {inputs: {a, b}}) as
        NDArray<G>;
  }

  /**
   * Computes the power of one value to another.
   * Given a tensor x and a tensor y, this operation computes x^y for
   * corresponding elements in x and y. For example:
   * x = tf.constant([[2, 2], [3, 3]])
   * y = tf.constant([[8, 16], [2, 3]])
   * pow(x, y)  # [[256, 65536], [9, 27]]
   *
   * @param a The base NDArray to pow element-wise.
   * @param b The exponent NDArray to pow element-wise.
   */
  pow<G extends keyof DataTypes>(a: NDArray<G>, b: NDArray<'int32'>):
      NDArray<G> {
    util.assert(
        b.dtype === 'int32',
        'only supports int32 data type for the exponent parameter.');
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return this.backendEngine.executeKernel('Pow', {inputs: {a, b}}) as
        NDArray<G>;
  }

  /**
   * Computes the power of one value to another. Inputs must
   * be the same shape. For broadcasting support, use math.pow() instead.
   *
   * @param a The base NDArray to pow element-wise.
   * @param b The exponent NDArray to pow element-wise.
   */
  powStrict<G extends keyof DataTypes>(a: NDArray<G>, b: NDArray<'int32'>):
      NDArray<G> {
    util.assertShapesMatch(a.shape, b.shape, 'Error in powStrict: ');
    return this.pow(a, b);
  }

  /** @deprecated Use math.subtract instead. */
  sub<G extends keyof DataTypes>(a: NDArray<G>, b: NDArray<G>): NDArray<G> {
    return this.subtract(a, b);
  }

  /**
   * Subtracts two NDArrays element-wise, A - B. Inputs must
   * be the same shape. For broadcasting support, use math.sub() instead.
   *
   * @param a The first NDArray to multiply element-wise.
   * @param b The second NDArray to multiply element-wise.
   */
  subStrict<D extends keyof DataTypes, T extends NDArray<D>>(a: T, b: T): T {
    util.assertShapesMatch(a.shape, b.shape, 'Error in subStrict: ');
    return this.subtract(a, b) as T;
  }

  /**
   * Multiplies two NDArrays element-wise, A * B. Supports broadcasting.
   * For a stricter version without broadcasting use math.multiplyStrict().
   *
   * @param a The first NDArray to multiply element-wise.
   * @param b The second NDArray to multiply element-wise.
   */
  multiply(a: NDArray, b: NDArray): NDArray {
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return this.backendEngine.executeKernel('Mul', {inputs: {a, b}});
  }

  /**
   * @deprecated Use math.multiplyStrict() instead.
   */
  elementWiseMul<T extends NDArray>(a: T, b: T): T {
    return this.multiplyStrict(a, b);
  }

  /**
   * Multiplies two NDArrays element-wise, A * B. Inputs must
   * be the same shape. For broadcasting support, use math.multiply() instead.
   *
   * @param a The first NDArray to multiply element-wise.
   * @param b The second NDArray to multiply element-wise.
   */
  multiplyStrict<T extends NDArray>(a: T, b: T): T {
    util.assertShapesMatch(a.shape, b.shape, 'Error in multiplyStrict: ');
    return this.multiply(a, b) as T;
  }

  /**
   * Divides two NDArrays element-wise, A / B. Supports broadcasting.
   * For a stricter version without broadcasting use math.divideStrict().
   *
   * @param a The first NDArray to divide element-wise.
   * @param b The second NDArray to divide element-wise.
   */
  divide(a: NDArray, b: NDArray): NDArray<'float32'> {
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return this.backendEngine.executeKernel('Div', {inputs: {a, b}}) as
        NDArray<'float32'>;
  }

  /**
   * Divides two NDArrays element-wise, A / B. Inputs must
   * be the same shape. For broadcasting support, use math.divide() instead.
   *
   * @param a The first NDArray to multiply element-wise.
   * @param b The second NDArray to multiply element-wise.
   */
  divideStrict<T extends NDArray>(a: T, b: T): T {
    util.assertShapesMatch(a.shape, b.shape, 'Error in divideStrict: ');
    return this.divide(a, b) as T;
  }

  /** @deprecated Use math.divide(c, A) instead. */
  scalarDividedByArray<T extends NDArray>(c: Scalar, a: T): T {
    util.assert(
        c.size === 1,
        `Error in scalarDividedByArray: first argument must be rank 0, but ` +
            `got NDArray of rank ${c.rank}.`);
    return this.divide(c, a) as T;
  }

  /** @deprecated Use math.divide(A, c) instead. */
  arrayDividedByScalar<T extends NDArray>(a: T, c: Scalar): T {
    util.assert(
        c.size === 1,
        `Error in arrayDividedByScalar: second argument must be rank 0, ` +
            `but got NDArray of rank ${c.rank}.`);
    return this.divide(a, c) as T;
  }

  /**
   * Computes ceiling of input NDArray element-wise. y = ceil(x)
   * TODO(nsthorat): Make this return an int32 when we add rank as a
   * generic.
   * @param x The input NDArray.
   */
  ceil<T extends NDArray>(x: T): T {
    return this.backendEngine.executeKernel('Ceil', {inputs: {x}}) as T;
  }

  /**
   * Computes floor of input NDArray element-wise. y = floor(x)
   * TODO(nsthorat): Make this return an int32 when we add rank as a generic.
   * @param x The input NDArray.
   */
  floor<T extends NDArray>(x: T): T {
    return this.backendEngine.executeKernel('Floor', {inputs: {x}}) as T;
  }

  /**
   * Computes exponential of the input NDArray element-wise. y = e ^ x
   * @param x The input NDArray.
   */
  exp<T extends NDArray>(x: T): T {
    return this.backendEngine.executeKernel('Exp', {inputs: {x}}) as T;
  }

  /**
   * Computes natural logarithm of the input NDArray element-wise. y = ln(x)
   * @param x The input NDArray.
   */
  log<T extends NDArray>(x: T): T {
    return this.backendEngine.executeKernel('Log', {inputs: {x}}) as T;
  }

  /**
   * Computes square root of the input NDArray element-wise. y = sqrt(x)
   * @param x The input NDArray.
   */
  sqrt<T extends NDArray>(x: T): T {
    return this.backendEngine.executeKernel('Sqrt', {inputs: {x}}) as T;
  }

  /**
   * Computes square of `x` element-wise.
   *
   * @param x The input array.
   */
  square<T extends NDArray>(x: T): T {
    return this.backendEngine.executeKernel('Square', {inputs: {x}}) as T;
  }

  /**
   * Computes absolute value element-wise.
   * @param x The input NDArray.
   */
  abs<T extends NDArray>(x: T): T {
    return this.backendEngine.executeKernel('Abs', {inputs: {x}}) as T;
  }

  /**
   * Clips values element-wise.
   * @param x The input NDArray.
   * @param min Lower-bound of range to be clipped to.
   * @param max Upper-bound of range to be clipped to.
   */
  clip<T extends NDArray>(x: T, min: number, max: number): T {
    util.assert(
        (min <= max),
        `Error in clip: min (${min}) must be` +
            `less than or equal to max (${max}).`);
    return this.backendEngine.executeKernel(
               'Clip', {inputs: {x}, args: {min, max}}) as T;
  }

  /**
   * Computes rectified linear element-wise, max(x, 0).
   * @param x The input NDArray.
   */
  relu<T extends NDArray>(x: T): T {
    return this.backendEngine.executeKernel(
               'Relu', {inputs: {x}}, (dy: T, y: T) => {
                 return {x: () => this.step(x)};
               }) as T;
  }

  /**
   * Computes exponential linear element-wise
   * @param {T} x the input NDArray
   */
  elu<T extends NDArray>(x: T): T {
    return this.backendEngine.executeKernel('Elu', {inputs: {x}}) as T;
  }

  /**
   * Computes the derivatice of elu which is used ly
   * @hidden
   */
  eluDer<T extends NDArray>(x: T): T {
    return this.backendEngine.executeKernel('EluDer', {inputs: {x}}) as T;
  }

  /**
   * Computes scaled exponential linear element-wise.
   * @hidden
   */
  selu<T extends NDArray>(x: T): T {
    return this.backendEngine.executeKernel('Selu', {inputs: {x}}) as T;
  }

  /**
   * Computes leaky rectified linear element-wise
   * @param {T} x the input NDArray
   * @param alpha scaleing factor for negative values, defaults to 0.2
   * @return {NDArray}
   */
  leakyRelu<T extends NDArray>(x: T, alpha = 0.2): T {
    return this.backendEngine.executeKernel(
               'LeakyRelu', {inputs: {x}, args: {alpha}}) as T;
  }

  /**
   * Computes sigmoid element-wise, y = 1 / (1 + exp(-x)).
   * @param x The input NDArray.
   */
  sigmoid<T extends NDArray>(x: T): T {
    return this.backendEngine.executeKernel('Sigmoid', {inputs: {x}}) as T;
  }

  /**
   * Computes sin of the input NDArray element-wise, y = sin(x).
   * @param x The input NDArray.
   */
  sin<T extends NDArray>(x: T): T {
    return this.backendEngine.executeKernel('Sin', {inputs: {x}}) as T;
  }

  /**
   * Computes cos of the input NDArray element-wise, y = cos(x).
   * @param x The input NDArray.
   */
  cos<T extends NDArray>(x: T): T {
    return this.backendEngine.executeKernel('Cos', {inputs: {x}}) as T;
  }

  /**
   * Computes tan of the input NDArray element-wise, y = tan(x).
   * @param x The input NDArray.
   */
  tan<T extends NDArray>(x: T): T {
    return this.backendEngine.executeKernel('Tan', {inputs: {x}}) as T;
  }

  /**
   * Computes asin of the input NDArray element-wise, y = asin(x).
   * @param x The input NDArray.
   */
  asin<T extends NDArray>(x: T): T {
    return this.backendEngine.executeKernel('Asin', {inputs: {x}}) as T;
  }

  /**
   * Computes acos of the input NDArray element-wise, y = acos(x).
   * @param x The input NDArray.
   */
  acos<T extends NDArray>(x: T): T {
    return this.backendEngine.executeKernel('Acos', {inputs: {x}}) as T;
  }

  /**
   * Computes atan of the input NDArray element-wise, y = atan(x).
   * @param x The input NDArray.
   */
  atan<T extends NDArray>(x: T): T {
    return this.backendEngine.executeKernel('Atan', {inputs: {x}}) as T;
  }

  /**
   * Computes hyperbolic sin of the input NDArray element-wise, y = sinh(x).
   * @param x The input NDArray.
   */
  sinh<T extends NDArray>(x: T): T {
    return this.backendEngine.executeKernel('Sinh', {inputs: {x}}) as T;
  }

  /**
   * Computes hyperbolic cos of the input NDArray element-wise, y = cosh(x).
   * @param x The input NDArray.
   */
  cosh<T extends NDArray>(x: T): T {
    return this.backendEngine.executeKernel('Cosh', {inputs: {x}}) as T;
  }

  /**
   * Computes hyperbolic tangent of the input NDArray element-wise.
   * @param x The input NDArray.
   */
  tanh<T extends NDArray>(x: T): T {
    return this.backendEngine.executeKernel('Tanh', {inputs: {x}}) as T;
  }

  /**
   * Computes step of the input NDArray element-wise,
   * y=1 if x>0|alpha*x if x<=0.
   *
   * @param x The input NDArray.
   * @param alpha The gradient when input is negative.
   */
  step<T extends NDArray>(x: T, alpha = 0.0): T {
    return this.backendEngine.executeKernel(
               'Step', {inputs: {x}, args: {alpha}}) as T;
  }

  /**
   * Computes a scaled array add operation, c1 * A + c2 * B.
   * @param c1 The first scalar in the scaled array add computation.
   * @param a The first NDArray in the scaled array add computation.
   * @param c2 The second scalar in the scaled array add computation.
   * @param cb The second NDArray in the scaled array add computation.
   */
  scaledArrayAdd<T extends NDArray>(c1: Scalar, a: T, c2: Scalar, b: T): T {
    util.assert(
        c1.size === 1,
        `Error in scaledArrayAdd: first argument must rank 0, but got ` +
            ` rank ${c1.rank}.`);
    util.assert(
        c2.size === 1,
        `Error in scaledArrayAdd: third argument must be rank 0, but got ` +
            `NDArray of rank ${c2.rank}.`);
    util.assertShapesMatch(a.shape, b.shape, 'Error in scaledArrayAdd: ');

    return this.executeOp('scaledArrayAdd', () => {
      return this.scope(() => {
        // TODO(nsthorat): Add an SGEMM kernel and then update this.
        return this.add(this.multiply(c1, a), this.multiply(c2, b)) as T;
      });
    });
  }

  /** @deprecated Use math.multiply(c, A) instead. */
  scalarTimesArray<T extends NDArray>(c: Scalar, a: T): T {
    util.assert(
        c.size === 1,
        `Error in arrayDividedByScalar: first argument must be rank 0, but ` +
            `got rank ${c.rank}.`);
    return this.multiply(c, a) as T;
  }

  /**
   * @deprecated Use math.multiply() instead.
   */
  elementWiseMulBroadcast(a: Array2D, b: Array2D): Array2D {
    util.assert(
        a.rank === 2,
        `Error in elementWiseMulBroadcast: first argument must be ` +
            `rank 2, but got rank ${a.rank}.`);
    util.assert(
        b.rank === 2,
        `Error in elementWiseMulBroadcast: second argument must be ` +
            `rank 2, but got rank ${b.rank}.`);
    return this.multiply(a, b) as Array2D;
  }

  /////////////////////
  // Convolution ops //
  /////////////////////

  /**
   * Computes a 1D convolution over the input x.
   * @param input The input ndarray, of rank 3 or rank 2, of shape
   *     `[batch, width, inChannels]`. If rank 2, batch of 1 is assumed.
   * @param filter The filter, rank 3, of shape
   *     [filterWidth, inDepth, outDepth].
   * @param bias Optional bias, rank 1 of shape [outDepth].
   * @param stride The number of entries by which the filter is moved right at
   *     each step.
   * @param pad A string from: 'same', 'valid'. The type of padding algorithm.
   *    - 'same' pad and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *    - 'valid' pad: output will be smaller than input if filter is larger
   *       than 1x1.
   *   - For more info, see this guide:
   *     https://www.tensorflow.org/api_guides/python/nn#Convolution
   */
  conv1d<T extends NDArray>(
      input: T, filter: Array3D, bias: Array1D|null, stride: number,
      pad: 'valid'|'same'|number): T {
    let input3D = input as NDArray as Array3D;
    let reshapedTo3D = false;
    if (input.rank === 2) {
      reshapedTo3D = true;
      input3D = input.as3D(1, input.shape[0], input.shape[1]);
    }

    util.assert(
        input3D.rank === 3,
        `Error in conv1d: input must be rank 3, but got rank ${input3D.rank}.`);
    util.assert(
        filter.rank === 3,
        `Error in conv1d: filter must be rank 3, but got rank ` +
            `${filter.rank}.`);
    if (bias != null) {
      util.assert(
          bias.rank === 1,
          `Error in conv1d: bias must be rank 1, but got rank ` +
              `${bias.rank}.`);
    }

    util.assert(
        input3D.shape[2] === filter.shape[1],
        `Error in conv1d: depth of input (${input3D.shape[2]}) must match  ` +
            `input depth for filter ${filter.shape[1]}.`);

    const filter4D =
        filter.as4D(1, filter.shape[0], filter.shape[1], filter.shape[2]);
    const input4D =
        input3D.as4D(input3D.shape[0], 1, input3D.shape[1], input3D.shape[2]);
    const strides: [number, number] = [1, stride];

    return this.executeOp('Conv1D', () => {
      const res = this.conv2d(input4D, filter4D, bias, strides, pad);
      if (reshapedTo3D) {
        return res.as2D(res.shape[2], res.shape[3]) as NDArray as T;
      }
      return res.as3D(res.shape[0], res.shape[2], res.shape[3]) as NDArray as T;
    });
  }

  /**
   * Computes a 2D convolution over the input x.
   *
   * @param input The input ndarray, of rank 4 or rank 3, of shape
   *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
   * @param filter The filter, rank 4, of shape
   *     [filterHeight, filterWidth, inDepth, outDepth].
   * @param bias Optional bias, rank 1 of shape [outDepth].
   * @param strides The strides of the convolution: [strideHeight, strideWidth].
   * @param pad A string from: 'same', 'valid'. The type of padding algorithm.
   *    - 'same' pad and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *    - 'valid' pad: output will be smaller than input if filter is larger
   *       than 1x1.
   *   - For more info, see this guide:
   *     https://www.tensorflow.org/api_guides/python/nn#Convolution
   */
  conv2d<T extends NDArray>(
      input: T, filter: Array4D, bias: Array1D|null,
      strides: [number, number]|number, pad: 'valid'|'same'|number): T {
    let input4D = input as NDArray as Array4D;
    let reshapedTo4D = false;
    if (input.rank === 3) {
      reshapedTo4D = true;
      input4D = input.as4D(1, input.shape[0], input.shape[1], input.shape[2]);
    }
    util.assert(
        input4D.rank === 4,
        `Error in conv2d: input must be rank 4, but got rank ${input4D.rank}.`);
    util.assert(
        filter.rank === 4,
        `Error in conv2d: filter must be rank 4, but got rank ` +
            `${filter.rank}.`);
    if (bias != null) {
      util.assert(
          bias.rank === 1,
          `Error in conv2d: bias must be rank 1, but got rank ` +
              `${bias.rank}.`);
    }

    util.assert(
        input4D.shape[3] === filter.shape[2],
        `Error in conv2d: depth of input (${input4D.shape[3]}) must match  ` +
            `input depth for filter ${filter.shape[2]}.`);

    const convInfo =
        conv_util.computeConv2DInfo(input4D.shape, filter.shape, strides, pad);
    return this.executeOp('Conv2D', () => {
      const res = this.backendEngine.executeKernel(
          'Conv2D', {inputs: {x: input4D, filter, bias}, args: {convInfo}});
      if (reshapedTo4D) {
        return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as NDArray as
            T;
      }
      return res as NDArray as T;
    });
  }

  /**
   * Computes the derivative of the input of a 2D convolution.
   *
   * @param inShape The shape of the input: [batch, height, width, inDepth]. If
   *     length of 3, batch of 1 is assumed.
   * @param dy The derivative of the output, of rank 4 or rank 3 of shape
   *   [batch, outHeight, outWidth, outDepth]. If rank 3, batch of 1 is assumed.
   * @param filter The filter, rank 4, of shape
   *     [filterHeight, filterWidth, inDepth, outDepth].
   * @param strides The strides of the convolution: [strideHeight, strideWidth].
   * @param pad A string from: 'same', 'valid'. The type of padding algorithm
   *     used in the forward prop of the op.
   */
  conv2dDerInput<T extends NDArray>(
      inShape: [number, number, number, number]|[number, number, number], dy: T,
      filter: Array4D, strides: [number, number]|number,
      pad: 'valid'|'same'|number): T {
    util.assert(
        inShape.length === dy.rank,
        `Length of inShape ` +
            `(${inShape.length}) and rank of dy (${dy.rank}) must match`);

    let inShape4D = inShape as [number, number, number, number];
    let dy4D = dy as NDArray as Array4D;
    let reshapedTo4D = false;
    if (dy.rank === 3) {
      reshapedTo4D = true;
      dy4D = dy.as4D(1, dy.shape[0], dy.shape[1], dy.shape[2]);
      inShape4D = [1, inShape[0], inShape[1], inShape[2]];
    }

    const inDepth = inShape4D[3];
    const outDepth = dy4D.shape[3];
    util.assert(
        inShape4D.length === 4,
        `Error in conv2dDerInput: inShape must be length 4, but got length ` +
            `${inShape4D.length}.`);
    util.assert(
        dy4D.rank === 4,
        `Error in conv2dDerInput: dy must be rank 4, but got ` +
            `rank ${dy4D.rank}`);
    util.assert(
        filter.rank === 4,
        `Error in conv2dDerInput: filter must be rank 4, but got ` +
            `rank ${filter.rank}`);
    util.assert(
        inDepth === filter.shape[2],
        `Error in conv2dDerInput: depth of input (${inDepth}) must ` +
            `match input depth for filter ${filter.shape[2]}.`);
    util.assert(
        outDepth === filter.shape[3],
        `Error in conv2dDerInput: depth of output (${outDepth}) must` +
            `match output depth for filter ${filter.shape[3]}.`);

    const convInfo =
        conv_util.computeConv2DInfo(inShape4D, filter.shape, strides, pad);
    return this.executeOp('conv2dDerInput', () => {
      const res = this.backendEngine.executeKernel(
          'Conv2DDerInput', {inputs: {dy: dy4D, filter}, args: {convInfo}});
      if (reshapedTo4D) {
        return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as NDArray as
            T;
      }
      return res as NDArray as T;
    });
  }

  /**
   * Computes the derivative of the bias of a 2D convolution.
   *
   * @param dy The gradient for the output of this op, of rank 4 or rank 3 of
   *   shape [batch, height, width, outDepth]. If rank 3, batch of 1 is assumed.
   */
  conv2dDerBias(dy: Array3D|Array4D): Array1D {
    let dy4D = dy as Array4D;
    if (dy.rank === 3) {
      dy4D = dy.as4D(1, dy.shape[0], dy.shape[1], dy.shape[2]);
    }
    return this.backendEngine.executeKernel(
        'Conv2DDerBias', {inputs: {dy: dy4D}});
  }

  /**
   * Computes the derivative of the filter of a 2D convolution.
   *
   * @param input The input ndarray, of rank 4 or rank 3 of shape
   *     [batch, height, width, inChannels]. If rank 3, batch of 1 is assumed.
   * @param dy The dy image, of rank 4 or rank 3, of shape
   *     [batch, height, width, outDepth]. If rank 3, batch of 1 is assumed.
   * @param filterShape The shape of the filter, length 4,
   *     [filterHeight, filterWidth, inDepth, outDepth].
   * @param strides The strides of the convolution: [strideHeight, strideWidth].
   * @param pad A string from: 'same', 'valid'. The type of padding algorithm
   *     used in the forward prop of the op.
   */
  conv2dDerFilter<T extends NDArray>(
      input: T, dy: T, filterShape: [number, number, number, number],
      strides: [number, number]|number, pad: 'valid'|'same'|number): Array4D {
    let input4D = input as NDArray as Array4D;
    if (input.rank === 3) {
      input4D = input.as4D(1, input.shape[0], input.shape[1], input.shape[2]);
    }
    let dy4D = dy as NDArray as Array4D;
    if (dy4D.rank === 3) {
      dy4D = dy.as4D(1, dy.shape[0], dy.shape[1], dy.shape[2]);
    }
    util.assert(
        input4D.rank === 4,
        `Error in conv2dDerFilter: input must be rank 4, but got shape ` +
            `${input4D.shape}.`);
    util.assert(
        dy4D.rank === 4,
        `Error in conv2dDerFilter: dy must be rank 4, but got shape ` +
            `${dy4D.shape}.`);
    util.assert(
        filterShape.length === 4,
        `Error in conv2dDerFilter: filterShape must be length 4, but got ` +
            `${filterShape}.`);
    util.assert(
        input4D.shape[3] === filterShape[2],
        `Error in conv2dDerFilter: depth of input ${input4D.shape[3]}) must ` +
            `match input depth in filter (${filterShape[2]}.`);
    util.assert(
        dy4D.shape[3] === filterShape[3],
        `Error in conv2dDerFilter: depth of dy (${dy4D.shape[3]}) must ` +
            `match output depth for filter (${filterShape[3]}).`);

    const convInfo =
        conv_util.computeConv2DInfo(input4D.shape, filterShape, strides, pad);
    return this.backendEngine.executeKernel(
        'Conv2DDerFilter', {inputs: {x: input4D, dy: dy4D}, args: {convInfo}});
  }

  /**
   * Computes the transposed 2D convolution of an image, also known as a
   * deconvolution.
   *
   * @param x The input image, of rank 4 or rank 3, of shape
   *   [batch, height, width, inDepth]. If rank 3, batch of 1 is assumed.
   * @param filter The filter, rank 4, of shape
   *     `[filterHeight, filterWidth, outDepth, inDepth]`.
   *     `inDepth` must match `inDepth` in `x`.
   * @param outputShape Output shape, of rank 4 or rank 3:
   *     [batch, height, width, outDepth]. If rank 3, batch of 1 is assumed.
   * @param strides The strides of the original convolution:
   *     `[strideHeight, strideWidth]`.
   * @param pad A string from: 'same', 'valid'. The type of padding algorithm
   *     used in the non-transpose version of the op.
   */
  conv2dTranspose<T extends NDArray>(
      x: T, filter: Array4D,
      outputShape: [number, number, number, number]|[number, number, number],
      strides: [number, number]|number, pad: 'valid'|'same'|number): T {
    return this.conv2dDerInput(outputShape, x, filter, strides, pad);
  }

  /**
   * Depthwise 2D convolution.
   *
   * Given a 4D `input` array and a `filter` array of shape
   * `[filterHeight, filterWidth, inChannels, channelMultiplier]` containing
   * `inChannels` convolutional filters of depth 1, this op applies a different
   * filter to each input channel (expanding from 1 channel to
   * `channelMultiplier` channels for each), then concatenates the results
   * together. The output has `inChannels * channelMultiplier` channels.
   *
   * See https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d for
   * more details.
   *
   * @param input The input ndarray, of rank 4 or rank 3, of shape
   *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
   * @param filter The filter ndarray, rank 4, of shape
   *     `[filterHeight, filterWidth, inChannels, channelMultiplier]`.
   * @param strides The strides of the convolution: [strideHeight, strideWidth].
   *     If strides is a single number, then `strideHeight == strideWidth`.
   * @param pad A string from: 'same', 'valid'. The type of padding algorithm.
   *   - 'same' pad and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *   - 'valid' pad: output will be smaller than input if filter is larger
   *       than 1x1.
   *   - For more info, see this guide:
   *     https://www.tensorflow.org/api_guides/python/nn#Convolution
   * @param rates The dilation rates: `[rateHeight, rateWidth]` in which we
   *     sample input values across the height and width dimensions in atrous
   *     convolution. Defaults to `[1, 1]`. If `rate` is a single number, then
   *     `rateHeight == rateWidth`. If it is greater than 1, then all values of
   *     `strides` must be 1.
   */
  depthwiseConv2D<T extends NDArray>(
      input: T, filter: Array4D, strides: [number, number]|number,
      pad: 'valid'|'same'|number, rates: [number, number]|number = [1, 1]): T {
    let input4D = input as NDArray as Array4D;
    let reshapedTo4D = false;
    if (input.rank === 3) {
      reshapedTo4D = true;
      input4D = input.as4D(1, input.shape[0], input.shape[1], input.shape[2]);
    }
    util.assert(
        input4D.rank === 4,
        `Error in depthwiseConv2D: input must be rank 4, but got ` +
            `rank ${input4D.rank}.`);
    util.assert(
        filter.rank === 4,
        `Error in depthwiseConv2D: filter must be rank 4, but got rank ` +
            `${filter.rank}.`);
    util.assert(
        input4D.shape[3] === filter.shape[2],
        `Error in depthwiseConv2D: number of input channels ` +
            `(${input4D.shape[3]}) must match the inChannels dimension in ` +
            `filter ${filter.shape[2]}.`);
    rates = rates || [1, 1];
    const [rateHeight, rateWidth] = parseTupleParam(rates);
    util.assert(
        rateHeight === 1 && rateWidth === 1,
        'Error in depthwiseConv2D: rates greater than 1 are not yet ' +
            `supported. Got rates '${rates}'`);

    const convInfo = conv_util.computeConv2DInfo(
        input4D.shape, filter.shape, strides, pad, true /* depthwise */);
    return this.executeOp('depthwiseConv2D', () => {
      const res = this.backendEngine.executeKernel(
          'DepthwiseConv2D', {inputs: {x: input4D, filter}, args: {convInfo}});
      if (reshapedTo4D) {
        return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as NDArray as
            T;
      }
      return res as NDArray as T;
    });
  }

  /**
   * Computes the 2D max pooling of an image.
   *
   * @param input The input ndarray, of rank 4 or rank 3 of shape
   *     [batch, height, width, inChannels]. If rank 3, batch of 1 is assumed.
   * @param filterSize The filter size, a tuple [filterHeight, filterWidth].
   * @param strides The strides of the pooling: [strideHeight, strideWidth].
   * @param pad A string from: 'same', 'valid'. The type of padding algorithm.
   *    - 'same' pad and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *    - 'valid' pad: output will be smaller than input if filter is larger
   *       than 1x1.
   *   - For more info, see this guide:
   *     https://www.tensorflow.org/api_guides/python/nn#Convolution
   */
  maxPool<T extends NDArray>(
      input: T, filterSize: [number, number]|number,
      strides: [number, number]|number, pad: 'valid'|'same'|number): T {
    let input4D = input as NDArray as Array4D;
    let reshapedTo4D = false;
    if (input.rank === 3) {
      reshapedTo4D = true;
      input4D = input.as4D(1, input.shape[0], input.shape[1], input.shape[2]);
    }
    util.assert(
        input4D.rank === 4,
        `Error in maxPool: input must be rank 4 but got rank ${input4D.rank}.`);

    const convInfo =
        conv_util.computePool2DInfo(input4D.shape, filterSize, strides, pad);
    return this.executeOp('maxPool', () => {
      const res = this.backendEngine.executeKernel(
          'MaxPool', {inputs: {x: input4D}, args: {convInfo}});
      if (reshapedTo4D) {
        return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as NDArray as
            T;
      }
      return res as NDArray as T;
    });
  }

  /**
   * Computes the backprop of a max pool.
   *
   * @param dy The dy error, of rank 4 or rank 3 of shape
   *     [batchSize, height, width, channels]. If rank 3, batch of 1 is assumed.
   * @param input The input image, of rank 4 or rank 3 of shape
   *     [batchSize, height, width, channels]. If rank 3, batch of 1 is assumed.
   * @param filterSize The filter size, a tuple [filterHeight, filterWidth].
   * @param strides The strides of the pooling: [strideHeight, strideWidth].
   * @param pad A string from: 'same', 'valid'. The type of padding algorithm
   *     used in the forward prop of the op.
   */
  maxPoolBackprop<T extends NDArray>(
      dy: T, input: T, filterSize: [number, number]|number,
      strides: [number, number]|number, pad: 'valid'|'same'|number): T {
    util.assert(
        input.rank === dy.rank,
        `Rank of input (${input.rank}) does not match rank of dy (${dy.rank})`);

    let input4D = input as NDArray as Array4D;
    let dy4D = dy as NDArray as Array4D;
    let reshapedTo4D = false;
    if (input.rank === 3) {
      reshapedTo4D = true;
      input4D = input.as4D(1, input.shape[0], input.shape[1], input.shape[2]);
      dy4D = dy.as4D(1, dy.shape[0], dy.shape[1], dy.shape[2]);
    }

    util.assert(
        dy4D.rank === 4,
        `Error in maxPoolBackprop: dy must be rank 4 but got rank ` +
            `${dy4D.rank}.`);
    util.assert(
        input4D.rank === 4,
        `Error in maxPoolBackprop: input must be rank 4 but got rank ` +
            `${input4D.rank}.`);

    const convInfo =
        conv_util.computePool2DInfo(input4D.shape, filterSize, strides, pad);
    return this.executeOp('maxPoolBackprop', () => {
      const res = this.backendEngine.executeKernel(
          'MaxPoolBackprop',
          {inputs: {dy: dy4D, x: input4D}, args: {convInfo}});
      if (reshapedTo4D) {
        return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as NDArray as
            T;
      }
      return res as NDArray as T;
    });
  }

  /**
   * Computes the 2D min pooling of an image.
   *
   * @param input The input ndarray, of rank 4 or rank 3 of shape
   *     [batch, height, width, inChannels]. If rank 3, batch of 1 is assumed.
   * @param filterSize The filter size, a tuple [filterHeight, filterWidth].
   * @param strides The strides of the pooling: [strideHeight, strideWidth].
   * @param pad A string from: 'same', 'valid'. The type of padding algorithm.
   *    - 'same' pad and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *    - 'valid' pad: output will be smaller than input if filter is larger
   *       than 1x1.
   *   - For more info, see this guide:
   *     https://www.tensorflow.org/api_guides/python/nn#Convolution
   */
  minPool<T extends NDArray>(
      input: T, filterSize: [number, number]|number,
      strides: [number, number]|number, pad: 'valid'|'same'|number): T {
    let input4D = input as NDArray as Array4D;
    let reshapedTo4D = false;
    if (input.rank === 3) {
      reshapedTo4D = true;
      input4D = input.as4D(1, input.shape[0], input.shape[1], input.shape[2]);
    }
    util.assert(
        input4D.rank === 4,
        `Error in minPool: x must be rank 4 but got rank ${input4D.rank}.`);

    const convInfo =
        conv_util.computePool2DInfo(input4D.shape, filterSize, strides, pad);
    return this.executeOp('minPool', () => {
      const res = this.backendEngine.executeKernel(
          'MinPool', {inputs: {x: input4D}, args: {convInfo}});
      if (reshapedTo4D) {
        return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as NDArray as
            T;
      }
      return res as NDArray as T;
    });
  }

  /**
   * Computes the 2D average pooling of an image.
   *
   * @param input The input ndarray, of rank 4 or rank 3 of shape
   *     [batch, height, width, inChannels]. If rank 3, batch of 1 is assumed.
   * @param filterSize The filter size, a tuple [filterHeight, filterWidth].
   * @param strides The strides of the pooling: [strideHeight, strideWidth].
   * @param pad A string from: 'same', 'valid'. The type of padding algorithm.
   *    - 'same' pad and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *    - 'valid' pad: output will be smaller than input if filter is larger
   *       than 1x1.
   *   - For more info, see this guide:
   *     https://www.tensorflow.org/api_guides/python/nn#Convolution
   */
  avgPool<T extends NDArray>(
      input: T, filterSize: [number, number]|number,
      strides: [number, number]|number, pad: 'valid'|'same'|number): T {
    let input4D = input as NDArray as Array4D;
    let reshapedTo4D = false;
    if (input.rank === 3) {
      reshapedTo4D = true;
      input4D = input.as4D(1, input.shape[0], input.shape[1], input.shape[2]);
    }
    util.assert(
        input4D.rank === 4,
        `Error in avgPool: x must be rank 4 but got rank ${input4D.rank}.`);

    const convInfo =
        conv_util.computePool2DInfo(input4D.shape, filterSize, strides, pad);
    return this.executeOp('avgPool', () => {
      const res = this.backendEngine.executeKernel(
          'AvgPool', {inputs: {x: input4D}, args: {convInfo}});
      if (reshapedTo4D) {
        return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as NDArray as
            T;
      }
      return res as NDArray as T;
    });
  }

  /*
   * Bilinear resize a 3D array per each channel to a new 2D shape.
   * @param x The input Array3D.
   * @param newShape2D The new shape to resize the Array3D to. Each channel is
   * resized individually.
   * @param alignCorners An optional bool. Defaults to False. If true, rescale
   * input by (new_height - 1) / (height - 1), which exactly aligns the 4
   * corners of images and resized images. If false, rescale by new_height /
   * height. Treat similarly the width dimension.
   */
  resizeBilinear3D(
      x: Array3D, newShape2D: [number, number], alignCorners = false): Array3D {
    util.assert(
        x.rank === 3,
        `Error in resizeBilinear3D: x must be rank 3 but got rank ${x.rank}.`);
    util.assert(
        newShape2D.length === 2,
        `Error in resizeBilinear3D: new shape must 2D, but got shape ` +
            `${newShape2D}.`);
    return this.backendEngine.executeKernel(
        'ResizeBilinear3D', {inputs: {x}, args: {newShape2D, alignCorners}});
  }

  /**
   * Batch normalization 2D. Mean, variance, scale, and offset can be of two
   * shapes: 1) The same shape as the input: an Array2D. 2) In the common
   * case, the depth dimension is the last dimension of x, so the values would
   * be an Array1D of shape [depth].
   * @param x The input NDArray.
   * @param mean A mean NDArray.
   * @param variance A variance NDArray.
   * @param varianceEpsilon A small float number to avoid dividing by 0.
   * @param scale A scale NDArray.
   * @param offset An offset NDArray.
   */
  batchNormalization2D(
      x: Array2D, mean: Array2D|Array1D, variance: Array2D|Array1D,
      varianceEpsilon = .001, scale?: Array2D|Array1D,
      offset?: Array2D|Array1D): Array2D {
    util.assert(
        x.rank === 2,
        `Error in batchNormalization3D: x must be rank 3 but got rank ` +
            `${x.rank}.`);
    util.assert(
        mean.rank === 2 || mean.rank === 1,
        `Error in batchNormalization2D: mean must be rank 2 or rank 1 but ` +
            `got rank ${mean.rank}.`);
    util.assert(
        variance.rank === 2 || variance.rank === 1,
        `Error in batchNormalization2D: variance must be rank 2 or rank 1 ` +
            `but got rank ${variance.rank}.`);
    if (scale != null) {
      util.assert(
          scale.rank === 2 || scale.rank === 1,
          `Error in batchNormalization2D: scale must be rank 2 or rank 1 ` +
              `but got rank ${scale.rank}.`);
    }
    if (offset != null) {
      util.assert(
          offset.rank === 2 || offset.rank === 1,
          `Error in batchNormalization2D: offset must be rank 2 or rank 1 ` +
              `but got rank ${offset.rank}.`);
    }

    return this.backendEngine.executeKernel(
        'BatchNorm2D',
        {inputs: {x, mean, variance, scale, offset}, args: {varianceEpsilon}});
  }

  /**
   * Batch normalization 3D. Mean, variance, scale, and offset can be of two
   * shapes: 1) The same shape as the input: an Array3D. 2) In the common
   * case, the depth dimension is the last dimension of x, so the values would
   * be an Array1D of shape [depth].
   * @param x The input NDArray.
   * @param mean A mean NDArray.
   * @param variance A variance NDArray.
   * @param varianceEpsilon A small float number to avoid dividing by 0.
   * @param scale A scale NDArray.
   * @param offset An offset NDArray.
   */
  batchNormalization3D(
      x: Array3D, mean: Array3D|Array1D, variance: Array3D|Array1D,
      varianceEpsilon = .001, scale?: Array3D|Array1D,
      offset?: Array3D|Array1D): Array3D {
    util.assert(
        x.rank === 3,
        `Error in batchNormalization3D: x must be rank 3 but got rank ` +
            `${x.rank}.`);
    util.assert(
        mean.rank === 3 || mean.rank === 1,
        `Error in batchNormalization3D: mean must be rank 3 or rank 1 but ` +
            `got rank ${mean.rank}.`);
    util.assert(
        variance.rank === 3 || variance.rank === 1,
        `Error in batchNormalization3D: variance must be rank 3 or rank 1 ` +
            `but got rank ${variance.rank}.`);
    if (scale != null) {
      util.assert(
          scale.rank === 3 || scale.rank === 1,
          `Error in batchNormalization3D: scale must be rank 3 or rank 1 ` +
              `but got rank ${scale.rank}.`);
    }
    if (offset != null) {
      util.assert(
          offset.rank === 3 || offset.rank === 1,
          `Error in batchNormalization3D: offset must be rank 3 or rank 1 ` +
              `but got rank ${offset.rank}.`);
    }

    return this.backendEngine.executeKernel(
        'BatchNorm3D',
        {inputs: {x, mean, variance, scale, offset}, args: {varianceEpsilon}});
  }

  //////////////
  // LSTM ops //
  //////////////

  /**
   * Computes the next states and outputs of a stack of LSTMCells.
   * Each cell output is used as input to the next cell.
   * This is only the forward mode.
   * Derived from tf.contrib.rn.MultiRNNCell.
   * @param lstmCells Array of LSTMCell functions.
   * @param data The input to the cell.
   * @param c Array of previous cell states.
   * @param h Array of previous cell outputs.
   * @return Tuple [nextCellStates, cellOutputs]
   */
  multiRNNCell(
      lstmCells: LSTMCell[], data: Array2D, c: Array2D[],
      h: Array2D[]): [Array2D[], Array2D[]] {
    const res = this.scope(() => {
      let input = data;
      const newStates = [];
      for (let i = 0; i < lstmCells.length; i++) {
        const output = lstmCells[i](input, c[i], h[i]);
        newStates.push(output[0]);
        newStates.push(output[1]);
        input = output[1];
      }

      return newStates;
    });
    const newC: Array2D[] = [];
    const newH: Array2D[] = [];
    for (let i = 0; i < res.length; i += 2) {
      newC.push(res[i]);
      newH.push(res[i + 1]);
    }
    return [newC, newH];
  }

  /**
   * Computes the next state and output of a BasicLSTMCell.
   * This is only the forward mode.
   * Derived from tf.contrib.rnn.BasicLSTMCell.
   * @param forgetBias Forget bias for the cell.
   * @param lstmKernel The weights for the cell.
   * @param lstmBias The bias for the cell.
   * @param data The input to the cell.
   * @param c Previous cell state.
   * @param h Previous cell output.
   * @return Tuple [nextCellState, cellOutput]
   */
  basicLSTMCell(
      forgetBias: Scalar, lstmKernel: Array2D, lstmBias: Array1D, data: Array2D,
      c: Array2D, h: Array2D): [Array2D, Array2D] {
    const res = this.scope(() => {
      const combined = this.concat2D(data, h, 1);
      const weighted = this.matMul(combined, lstmKernel);
      const res = this.add(weighted, lstmBias) as Array2D;

      // i = input_gate, j = new_input, f = forget_gate, o = output_gate
      const batchSize = res.shape[0];
      const sliceCols = res.shape[1] / 4;
      const sliceSize: [number, number] = [batchSize, sliceCols];
      const i = this.slice2D(res, [0, 0], sliceSize);
      const j = this.slice2D(res, [0, sliceCols], sliceSize);
      const f = this.slice2D(res, [0, sliceCols * 2], sliceSize);
      const o = this.slice2D(res, [0, sliceCols * 3], sliceSize);

      const newC = this.addStrict(
          this.multiplyStrict(
              c, this.sigmoid(this.scalarPlusArray(forgetBias, f))),
          this.multiplyStrict(this.sigmoid(i), this.tanh(j)));
      const newH = this.multiplyStrict(this.tanh(newC), this.sigmoid(o));

      return [newC, newH];
    });
    return [res[0], res[1]];
  }

  /**
   * Draws samples from a multinomial distribution.
   *
   * @param probabilities 1D array with normalized outcome probabilities, or
   *     2D array of shape `[batchSize, numOutcomes]`.
   * @param numSamples Number of samples to draw for each row slice.
   * @param seed Optional. The seed number.
   * @return 1D array of shape `[numSamples]`, or 2D array of shape
   *     `[batchSize, numSamples]`, depending on the rank of the input.
   */
  multinomial(
      probabilities: Array1D|Array2D, numSamples: number,
      seed?: number): Array1D<'int32'>|Array2D<'int32'> {
    const numOutcomes = probabilities.size;
    if (numOutcomes < 2) {
      throw new Error(
          `Error in multinomial: you need at least 2 outcomes, but got ` +
          `${numOutcomes}.`);
    }
    if (probabilities.rank > 2) {
      throw new Error(
          `Rank of probabilities must be 1 or 2, but is ${probabilities.rank}`);
    }
    seed = seed || Math.random();
    const origRank = probabilities.rank;

    if (probabilities.rank === 1) {
      probabilities = probabilities.as2D(1, -1);
    }
    return this.executeOp('multinomial', () => {
      const res = this.backendEngine.executeKernel('Multinomial', {
        inputs: {probs: (probabilities as Array2D)},
        args: {numSamples, seed}
      });
      if (origRank === 1) {
        return res.as1D();
      }
      return res;
    });
  }

  /**
   * Returns a one-hot array. The locations represented by `indices` take
   * value `onValue` (defaults to 1), while all other locations take value
   * `offValue` (defaults to 0).
   *
   * @param indices 1D Array of indices.
   * @param depth The depth of the one hot dimension.
   * @param onValue A number used to fill in output when the index matches the
   *     location.
   * @param offValue A number used to fill in the output when the index does
   *     not match the location.
   */
  oneHot(indices: Array1D, depth: number, onValue = 1, offValue = 0): Array2D {
    if (depth < 2) {
      throw new Error(`Error in oneHot: depth must be >=2, but it is ${depth}`);
    }
    return this.backendEngine.executeKernel(
        'OneHot', {inputs: {indices}, args: {depth, onValue, offValue}});
  }

  /**
   * Calculates the mean and variance of `x`. The mean and variance are
   * calculated by aggregating the contents of `x` across `axes`. If `x` is
   * 1-D and `axes = [0]` this is just the mean and variance of a vector.
   *
   * @param x The input array.
   * @param axis Optional. The dimension(s) along with to compute mean and
   *     variance. By default it reduces all dimensions.
   * @param keepDims If true, the moments have the same dimensionality as the
   *     input.
   * @return An object with two keys: `mean` and `variance`.
   */
  moments(x: NDArray, axis: number|number[] = null, keepDims = false):
      {mean: NDArray<'float32'>, variance: NDArray<'float32'>} {
    const axes = axis_util.parseAxisParam(axis, x.shape);
    const result = this.scope(() => {
      const mean = this.mean(x, axes, keepDims);
      let keepDimsShape = mean.shape;
      if (!keepDims) {
        keepDimsShape = axis_util.expandShapeToKeepDim(mean.shape, axes);
      }
      const devSquared =
          this.square(this.subtract(x, mean.reshape(keepDimsShape)));
      const variance = this.mean(devSquared, axes, keepDims);
      return {mean, variance};
    });
    return result;
  }

  /**
   * Warning: this is not fully implemented yet. Use with caution.
   *
   * Computes a gradient of y with respect to x.
   *
   * @param y The output Scalar. Assumes de/dy = 1.
   *          TODO(nsthorat): Accept non-scalars.
   * @param x The input to compute de/dx over. This can be a single value or an
   * object mapping a string to an NDArray. If using the object mode, this
   * method will return an object of the same shape.
   */
  gradientWrt<T extends NDArray|{[xName: string]: NDArray}>(y: Scalar, x: T):
      T {
    const xIsArray = x instanceof NDArray;

    const xs: NDArray[] = [];
    let xKeys: string[];
    if (xIsArray) {
      xs.push(x as NDArray);
    } else {
      const xMap = x as {[xName: string]: NDArray};
      xKeys = Object.keys(xMap);
      // Flatten the inputs.
      for (let i = 0; i < xKeys.length; i++) {
        xs.push(xMap[xKeys[i]]);
      }
    }

    const gradients = this.backendEngine.gradientWrt(y, xs);

    if (xIsArray) {
      return gradients[0] as T;
    } else {
      // Convert the flat list of gradients back to the object.
      const result: {[xName: string]: NDArray} = {};
      for (let i = 0; i < xKeys.length; i++) {
        result[xKeys[i]] = gradients[i];
      }
      return result as T;
    }
  }

  disposeData(id: number): void {
    this.backend.disposeData(id);
    // TODO(nsthorat): Construct an error and save the stack trace for debugging
    // when in debug mode. Creating a stack trace is too expensive to do
    // unconditionally.
    this.numArrays--;
  }
}

function parseTupleParam(param: number|[number, number]): [number, number] {
  return typeof param === 'number' ? [param, param] : param;
}
