/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as util from '../util';
import * as concat3d_util from './concat3d_util';
import * as copy2d_util from './copy2d_util';

import {Array1D, Array2D, Array3D, Array4D, NDArray, Scalar} from './ndarray';

export type ScopeResult = NDArray[]|NDArray|void;

export interface LSTMCell {
  (data: Array2D, c: Array2D, h: Array2D): [Array2D, Array2D];
}


export abstract class NDArrayMath {
  private ndarrayScopes: NDArray[][] = [];
  private activeScope: NDArray[];

  private ndarraysToKeep: NDArray[][] = [];
  private activeScopeNDArraysToKeep: NDArray[] = [];

  private debugMode = false;

  /**
   * @param safeMode In safe mode, you must use math operations inside
   *     a math.scope() which will automatically clean up intermediate NDArrays.
   */
  constructor(private safeMode: boolean) {}

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
          (keep: <T1 extends NDArray>(ndarray: T1) => T1,
           track: <T2 extends NDArray>(ndarray: T2) => T2) => T) {
    this.startScope();

    const keepFn = <T extends NDArray>(ndarray: T): T => this.keep(ndarray);
    const trackFn = <T extends NDArray>(ndarray: T): T => this.track(ndarray);
    const result = scopeFn(keepFn, trackFn);

    this.endScope(result);

    return result;
  }


  /**
   * In debug mode, the output of every math call will be downloaded to the CPU
   * and checked for NaNs. This significantly impacts performance.
   */
  enableDebugMode() {
    this.debugMode = true;
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

  /**
   * End a scope. Use this with startScope() to achieve the same functionality
   * as scope() without the need for a function closure.
   */
  endScope(result: ScopeResult) {
    let arraysToKeep = this.activeScopeNDArraysToKeep;
    if (result != null) {
      arraysToKeep = arraysToKeep.concat(result as NDArray | NDArray[]);
    }
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
    if (result instanceof NDArray &&
        !this.isNDArrayDataInList(result, this.activeScopeNDArraysToKeep)) {
      this.track(result);
    } else if (Array.isArray(result)) {
      result.forEach(r => {
        if (r instanceof NDArray &&
            !this.isNDArrayDataInList(r, this.activeScopeNDArraysToKeep)) {
          this.track(r);
        }
      });
    }

    this.ndarraysToKeep.pop();
    this.activeScopeNDArraysToKeep = this.ndarraysToKeep.length === 0 ?
        null :
        this.ndarraysToKeep[this.ndarraysToKeep.length - 1];
  }

  private isNDArrayDataInList(ndarray: NDArray, ndarrayList: NDArray[]) {
    for (let i = 0; i < ndarrayList.length; i++) {
      if (ndarrayList[i].getData() === ndarray.getData()) {
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

  private checkForNaN(arr: NDArray): void {
    const vals = arr.getValues();
    for (let i = 0; i < vals.length; i++) {
      if (isNaN(vals[i])) {
        throw Error('The result NDArray of the last math call has NaNs.');
      }
    }
  }

  /**
   * Tracks an NDArray in the current scope to be automatically cleaned up when
   * the current scope ends, and returns the value.
   * @param result The NDArray to track in the current scope.
   */
  track<T extends NDArray>(result: T): T {
    if (this.debugMode) {
      this.checkForNaN(result);
    }
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
            `and ${b.rank}.`);

    util.assert(
        innerShapeA === innerShapeB,
        `Error in matMul: inner shapes (${innerShapeA}) and (` +
            `${innerShapeB}) of NDArrays with shapes ${a.shape} and ` +
            `${b.shape} and orientations ${MatrixOrientation[aOrientation]}` +
            ` and ${MatrixOrientation[bOrientation]} must match.`);

    return this.track(this.matMulInternal(a, b, aOrientation, bOrientation));
  }
  protected abstract matMulInternal(
      a: Array2D, b: Array2D, aOrientation: MatrixOrientation,
      bOrientation: MatrixOrientation): Array2D;

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
        `Error in vectorTimesMatrix: size of first rank 1 input (${v.size}) ` +
            `must match inner dimension of second rank 2 input, but got ` +
            `rank ${matrix.rank}.`);

    return this.matMul(v.as2D(1, v.size), matrix).as1D();
  }

  /**
   * Computes the dot product of a matrix and vector, A * v.
   * @param matrix The matrix in dot product operation.
   * @param v The vector in dot product operation.
   */
  matrixTimesVector(matrix: Array2D, v: Array1D): Array1D {
    util.assert(
        v.rank === 1,
        `Error in vectorTimesMatrix: second input must rank 1, but got ` +
            `rank ${v.rank}.`);
    util.assert(
        matrix.rank === 2,
        `Error in vectorTimesMatrix: first input must be a rank 2, but got ` +
            `rank ${matrix.rank}.`);
    util.assert(
        v.size === matrix.shape[1],
        `Error in vectorTimesMatrix: size of first rank 1 input ${v.size} ` +
            `must match inner dimension of second rank 2 input, but got ` +
            `shape ${matrix.shape}.`);

    return this.matMul(matrix, v.as2D(v.size, 1)).as1D();
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
    return this.matMul(v1.as2D(1, v1.size), v2.as2D(v2.size, 1)).asScalar();
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

    return this.matMul(v1.as2D(v1.size, 1), v2.as2D(1, v2.size));
  }

  ///////////////
  // Shape ops //
  ///////////////

  /**
   * Clones an NDArray of any shape.
   * @param ndarray The NDArray to clone.
   */
  clone<T extends NDArray>(ndarray: T): T {
    return this.track(this.cloneInternal(ndarray));
  }
  protected abstract cloneInternal<T extends NDArray>(ndarray: T): T;

  /**
   * @deprecated Please call reshape() directly on the ndarray object.
   */
  reshape<T1 extends NDArray, T2 extends NDArray>(
      ndarray: T1, newShape: number[]): T2 {
    console.warn(
        'math.reshape() is deprecated. Please call reshape() ' +
        'directly on the ndarray object');
    return ndarray.reshape(newShape);
  }

  /**
   * Extracts a slice from a matrix. The operation extraces a slice from input
   * that starts at coordinates `begin` and is of size `size`.
   * @param input The input matrix to slice from.
   * @param begin The 2D coordinates in the input matrix to start the slice
   * from.
   * @param size The sice of the 2D window to slice.
   */
  slice2D(input: Array2D, begin: [number, number], size: [number, number]):
      Array2D {
    util.assert(
        begin[0] + size[0] <= input.shape[0] &&
            begin[1] + size[1] <= input.shape[1],
        `Error in slice2D: requested start position ${begin} and size ` +
            `${size} would overflow input of shape ${input.shape}.`);
    return this.track(this.slice2DInternal(input, begin, size));
  }
  protected abstract slice2DInternal(
      input: Array2D, begin: [number, number], size: [number, number]): Array2D;

  /**
   * Copies a window from the `source` matrix starting at `sourceBegin` and is
   * of size `sourceSize` to a window in the `dest` matrix starting at
   * `destBegin` and is of size `destSize`/
   * @param source The source matrix to copy from.
   * @param sourceBegin The coordinates to start the copy from.
   * @param sourceSize The size of the copy window.
   * @param dest The destination matrix to copy to.
   * @param destBegin The coordinates in `dest` to copy to.
   * @param destSize The size of the destination window.
   */
  copy2D(
      source: Array2D, sourceBegin: [number, number],
      sourceSize: [number, number], dest: Array2D, destBegin: [number, number],
      destSize: [number, number]) {
    util.assert(
        sourceBegin[0] + sourceSize[0] <= source.shape[0] &&
            sourceBegin[1] + sourceSize[1] <= source.shape[1],
        `Error in copy2D: requested source start position ${sourceBegin} ` +
            `and source size ${sourceSize} would overflow source NDArray` +
            `of shape ${source.shape}.`);
    util.assert(
        destBegin[0] + destSize[0] <= dest.shape[0] &&
            destBegin[1] + destSize[1] <= dest.shape[1],
        `Error in copy2D: requested dest start position ${destBegin} ` +
            `and source size ${destSize} would overflow dest NDArray of` +
            `shape ${dest.shape}.`);
    copy2d_util.validateShapes(sourceSize, destSize);

    return this.copy2DInternal(
        source, sourceBegin, sourceSize, dest, destBegin, destSize);
  }
  protected abstract copy2DInternal(
      source: Array2D, sourceBegin: [number, number],
      sourceSize: [number, number], dest: Array2D, destBegin: [number, number],
      destSize: [number, number]): void;

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
   * @param ndarray1 The first array to concat.
   * @param ndarray2 The second array to conat.
   * @param axis The axis to concate along.
   */
  concat3D(ndarray1: Array3D, ndarray2: Array3D, axis: number): Array3D {
    concat3d_util.assertConcat3DShapesMatch(
        ndarray1.shape, ndarray2.shape, axis, 'Error in concat3d: ');
    return this.track(this.concat3DInternal(ndarray1, ndarray2, axis));
  }
  protected abstract concat3DInternal(
      ndarray1: Array3D, ndarray2: Array3D, axis: number): Array3D;

  ///////////////////
  // Reduction ops //
  ///////////////////

  /**
   * Computes the the log(sum(e ^ x)) for each x in the input ndarray.
   * @param ndarray The input NDArray to compute the logSumExp over.
   */
  logSumExp(ndarray: NDArray): Scalar {
    return this.track(this.logSumExpInternal(ndarray));
  }
  protected abstract logSumExpInternal(ndarray: NDArray): Scalar;

  /**
   * Computes the sum of all the entries in the input NDArray.
   * @param ndarray The input NDArray to compute the sum over.
   */
  sum(ndarray: NDArray): Scalar {
    return this.track(this.sumInternal(ndarray));
  }
  protected abstract sumInternal(ndarray: NDArray): Scalar;

  /**
   * Computes the flattened index of the minimum element in the ndarray.
   * @param ndarray The input NDArray.
   */
  argMin(ndarray: NDArray): Scalar {
    return this.track(this.argMinInternal(ndarray));
  }
  protected abstract argMinInternal(ndarray: NDArray): Scalar;

  /**
   * Computes the flattened index of the maximum element in the ndarray.
   * @param ndarray The input NDArray.
   */
  argMax(ndarray: NDArray): Scalar {
    return this.track(this.argMaxInternal(ndarray));
  }
  protected abstract argMaxInternal(ndarray: NDArray): Scalar;

  /**
   * Returns a 1 if the argMax of x1 and x2 are the same, otherwise 0.
   * @param x1 The first input NDArray.
   * @param x2 The second input NDArray.
   */
  argMaxEquals(x1: NDArray, x2: NDArray): Scalar {
    util.assertShapesMatch(x1.shape, x2.shape, 'Error in argMaxEquals: ');
    return this.track(this.argMaxEqualsInternal(x1, x2));
  }
  protected abstract argMaxEqualsInternal(x1: NDArray, x2: NDArray): Scalar;

  /**
   * Computes the top K values and flattened indices.
   * @param ndarray The input NDArray.
   * @param k How many top values to compute.
   */
  topK(ndarray: NDArray, k: number): {values: Array1D, indices: Array1D} {
    util.assert(
        k <= ndarray.size,
        `Error in topK: k value (${k}) must be less than size of input ` +
            `ndarray, got shape ${ndarray.shape}.`);
    const result = this.topKInternal(ndarray, k);
    this.track(result.values);
    this.track(result.indices);
    return result;
  }
  protected abstract topKInternal(ndarray: NDArray, k: number):
      {values: Array1D, indices: Array1D};

  /**
   * Computes the minimum value from the input.
   * @param ndarray The input NDArray.
   */
  min(ndarray: NDArray): Scalar {
    return this.track(this.minInternal(ndarray));
  }
  protected abstract minInternal(ndarray: NDArray): Scalar;

  /**
   * Computes the maximum value from the input.
   * @param ndarray The input NDArray.
   */
  max(ndarray: NDArray): Scalar {
    return this.track(this.maxInternal(ndarray));
  }
  protected abstract maxInternal(ndarray: NDArray): Scalar;

  /**
   * Computes the softmax normalized vector from the input vector.
   * @param x The input vector.
   */
  softmax(x: Array1D): Array1D {
    return this.scope(() => {
      // Do it in log space for numerical stability.
      // exp(X - logSumExp(X))
      const lse = this.logSumExp(x);
      const logResult = this.arrayMinusScalar(x, lse);
      return this.exp(logResult);
    });
  }

  //////////////////////
  // Element-wise ops //
  //////////////////////

  /**
   * Switches dimensions of the input NDArray.
   * @param a The input NDArray.
   * @param newDim The new indices that define which shapes values to switch.
   */
  switchDim<T extends NDArray>(a: T, newDim: number[]): T {
    util.assert(
        a.rank === newDim.length,
        `Error in switchDim: length of input shape ${a.shape} ` +
            `must match size of newDim array ${newDim}.`);
    return this.track(this.switchDimInternal(a, newDim));
  }
  protected abstract switchDimInternal<T extends NDArray>(
      a: T, newDim: number[]): T;

  /**
   * Computes a scalar plus NDArray, c + A.
   * @param c The scalar c in c + A.
   * @param a The NDArray A in c + A.
   */
  scalarPlusArray<T extends NDArray>(c: Scalar, a: T): T {
    util.assert(
        c.size === 1,
        `Error in scalarPlusArray: first argument must be rank 0, but got ` +
            `rank ${c.rank}.`);
    return this.add(c, a) as T;
  }

  /**
   * Computes a scalar minus NDArray, c - A.
   * @param c The scalar c in c - A.
   * @param a The NDArray A in c - A.
   */
  scalarMinusArray<T extends NDArray>(c: Scalar, a: T): T {
    util.assert(
        c.size === 1,
        `Error in scalarMinusArray: first argument must be rank 0, but got ` +
            `rank ${c.rank}.`);
    return this.sub(c, a) as T;
  }

  /**
   * Computes A - c. A is NDArray, c is Scalar.
   * @param a The NDArray A in A - c.
   * @param c The Scalar c in A - c.
   */
  arrayMinusScalar<T extends NDArray>(a: T, c: Scalar): T {
    util.assert(
        c.size === 1,
        `Error in arrayMinusScalar: second argument must be rank 0, but ` +
            `got rank ${c.rank}.`);
    return this.sub(a, c) as T;
  }

  /**
   * Computes -1 * A element-wise.
   * @param a The input array.
   */
  neg<T extends NDArray>(a: T): T {
    return this.track(this.negInternal(a));
  }
  protected abstract negInternal<T extends NDArray>(a: T): T;

  /**
   * Adds two NDArrays element-wise, A + B. Supports broadcasting.
   * For a stricter version without broadcasting use math.addStrict().
   *
   * @param a The first NDArray to add element-wise.
   * @param b The second NDArray to add element-wise.
   */
  add(a: NDArray, b: NDArray): NDArray {
    util.assertAndGetBroadcastedShape(a.shape, b.shape);
    return this.track(this.addInternal(a, b));
  }
  protected abstract addInternal(a: NDArray, b: NDArray): NDArray;

  /**
   * Adds two NDArrays element-wise, A + B. Inputs must
   * be the same shape. For broadcasting support, use math.add() instead.
   *
   * @param a The first NDArray to multiply element-wise.
   * @param b The second NDArray to multiply element-wise.
   */
  addStrict<T extends NDArray>(a: T, b: T): T {
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
  sub(a: NDArray, b: NDArray): NDArray {
    util.assertAndGetBroadcastedShape(a.shape, b.shape);
    return this.track(this.subInternal(a, b));
  }
  protected abstract subInternal(a: NDArray, b: NDArray): NDArray;

  /**
   * Subtracts two NDArrays element-wise, A - B. Inputs must
   * be the same shape. For broadcasting support, use math.sub() instead.
   *
   * @param a The first NDArray to multiply element-wise.
   * @param b The second NDArray to multiply element-wise.
   */
  subStrict<T extends NDArray>(a: T, b: T): T {
    util.assertShapesMatch(a.shape, b.shape, 'Error in subStrict: ');
    return this.sub(a, b) as T;
  }

  /**
   * Multiplies two NDArrays element-wise, A * B. Supports broadcasting.
   * For a stricter version without broadcasting use math.multiplyStrict().
   *
   * @param a The first NDArray to multiply element-wise.
   * @param b The second NDArray to multiply element-wise.
   */
  multiply(a: NDArray, b: NDArray): NDArray {
    util.assertAndGetBroadcastedShape(a.shape, b.shape);
    return this.track(this.multiplyInternal(a, b));
  }
  protected abstract multiplyInternal<T extends NDArray>(a: T, b: T): T;

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
  divide(a: NDArray, b: NDArray): NDArray {
    util.assertAndGetBroadcastedShape(a.shape, b.shape);
    return this.track(this.divideInternal(a, b));
  }
  protected abstract divideInternal(a: NDArray, b: NDArray): NDArray;

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

  /**
   * Computes a scalar divided by an NDArray, broadcasted over the NDArray, c /
   * A.
   * @param c The scalar value in c / A.
   * @param a The NDArray value in c / A.
   */
  scalarDividedByArray<T extends NDArray>(c: Scalar, a: T): T {
    util.assert(
        c.size === 1,
        `Error in scalarDividedByArray: first argument must be rank 0, but ` +
            `got NDArray of rank ${c.rank}.`);
    return this.divide(c, a) as T;
  }

  /**
   * Computes an NDArray divided by a scalar, broadcasted over the NDArray, A /
   * c.
   * @param a The NDArray value in A / c.
   * @param c The scalar value in A / c.
   */
  arrayDividedByScalar<T extends NDArray>(a: T, c: Scalar): T {
    util.assert(
        c.size === 1,
        `Error in arrayDividedByScalar: second argument must be rank 0, ` +
            `but got NDArray of rank ${c.rank}.`);
    return this.divide(a, c) as T;
  }

  /**
   * Computes exponential of the input NDArray element-wise. y = e ^ x
   * @param ndarray The input NDArray.
   */
  exp<T extends NDArray>(ndarray: T): T {
    return this.track(this.expInternal(ndarray));
  }
  protected abstract expInternal<T extends NDArray>(ndarray: T): T;

  /**
   * Computes natural logarithm of the input NDArray element-wise. y = ln(x)
   * @param ndarray The input NDArray.
   */
  log<T extends NDArray>(ndarray: T): T {
    return this.track(this.logInternal(ndarray));
  }
  protected abstract logInternal<T extends NDArray>(ndarray: T): T;

  /**
   * Computes rectified linear element-wise, max(x, 0).
   * @param ndarray The input NDArray.
   */
  relu<T extends NDArray>(ndarray: T): T {
    return this.track(this.reluInternal(ndarray));
  }
  protected abstract reluInternal<T extends NDArray>(ndarray: T): T;

  /**
   * Computes sigmoid element-wise, y = 1 / (1 + exp(-x)).
   * @param ndarray The input NDArray.
   */
  sigmoid<T extends NDArray>(ndarray: T): T {
    return this.track(this.sigmoidInternal(ndarray));
  }
  protected abstract sigmoidInternal<T extends NDArray>(ndarray: T): T;

  /**
   * Computes hyperbolic tangent of the input NDArray element-wise.
   * @param ndarray The input NDArray.
   */
  tanh<T extends NDArray>(ndarray: T): T {
    return this.track(this.tanhInternal(ndarray));
  }
  protected abstract tanhInternal<T extends NDArray>(ndarray: T): T;

  /**
   * Computes sin of the input NDArray element-wise, y = sin(x).
   * @param ndarray The input NDArray.
   */
  sin<T extends NDArray>(ndarray: T): T {
    return this.track(this.sinInternal(ndarray));
  }
  protected abstract sinInternal<T extends NDArray>(ndarray: T): T;

  /**
   * Computes step of the input NDArray element-wise, y = 1 if x > 0 | 0 if x <=
   * 0
   * @param ndarray The input NDArray.
   */
  step<T extends NDArray>(ndarray: T): T {
    return this.track(this.stepInternal(ndarray));
  }
  protected abstract stepInternal<T extends NDArray>(ndarray: T): T;

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

    return this.track(this.scaledArrayAddInternal(c1, a, c2, b));
  }
  protected abstract scaledArrayAddInternal<T extends NDArray>(
      c1: Scalar, a: T, c2: Scalar, b: T): T;

  /**
   * Computes a scalar times array operation broadcasted over the NDArray, c *
   * A.
   * @param c The scalar in the operation.
   * @param A the NDArray in the operation that will be broadcasted over.
   */
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
   * Computes a 2D convolution over the input x.
   * @param x The input image, must be rank 3, of shape [rows, cols, depth1].
   * @param weights The weights NDArray, must be rank 4, of shape [f, f, depth1,
   * depth2].
   * @param biases Optional biases NDArray, must be rank 1 of shape [depth2].
   * @param stride The stride of the convolution.
   * @param zeroPad The zero padding of each side of the input NDArray. Will pad
   * equally on all sides.
   */
  conv2d(
      x: Array3D, weights: Array4D, biases: Array1D|null, stride: number,
      zeroPad: number): Array3D {
    util.assert(
        x.rank === 3,
        `Error in conv2d: x must be rank 3, but got rank ${x.rank}.`);
    util.assert(
        weights.rank === 4,
        `Error in conv2d: weights must be rank 4, but got rank ` +
            `${weights.rank}.`);
    if (biases != null) {
      util.assert(
          biases.rank === 1,
          `Error in conv2d: biases must be rank 1, but got rank ` +
              `${biases.rank}.`);
    }

    util.assert(
        x.shape[2] === weights.shape[2],
        `Error in conv2d: depth of input (${x.shape[2]}) must match  ` +
            `input depth for weights ${weights.shape[2]}.`);


    return this.track(this.conv2dInternal(x, weights, biases, stride, zeroPad));
  }
  protected abstract conv2dInternal(
      x: Array3D, weights: Array4D, biases: Array1D|null, stride: number,
      zeroPad: number): Array3D;

  /**
   * Computes the backprop of a 2D convolution.
   * @param x The input image, must be rank 3, of shape [xrows, xcols, depth1].
   * @param dy The dy image, must be rank 3, of shape [yrows, ycols, depth2].
   * @param weights The weights NDArray, must be rank 4, of shape [f, f, depth1,
   * depth2].
   * @param stride The stride of the original convolution.
   * @param pad The padding of the original convolution.
   */
  conv2dBackProp(
      x: Array3D, dy: Array3D, weights: Array4D, stride: number,
      pad: number): {dx: Array3D, dw: Array4D, db: Array1D} {
    util.assert(
        x.rank === 3,
        `Error in conv2dBackProp: x must be rank 3, but got shape ` +
            `${x.shape}.`);
    util.assert(
        dy.rank === 3,
        `Error in conv2dBackProp: dy must be rank 3, but got shape ` +
            `${dy.shape}.`);
    util.assert(
        weights.rank === 4,
        `Error in conv2dBackProp: weights must be rank 4, but got shape ` +
            `${weights.shape}.`);
    util.assert(
        x.shape[2] === weights.shape[2],
        `Error in conv2dBackProp: depth of x ${x.shape[2]}) must ` +
            `match input depth for weights (${weights.shape[2]}.`);
    util.assert(
        dy.shape[2] === weights.shape[3],
        `Error in conv2dBackProp: depth of dy (${dy.shape[2]}) must ` +
            `match output depth for weights (${weights.shape[3]}).`);

    const backpropResult =
        this.conv2dBackPropInternal(x, dy, weights, stride, pad);

    this.track(backpropResult.db);
    this.track(backpropResult.dw);
    this.track(backpropResult.dx);

    return backpropResult;
  }
  protected abstract conv2dBackPropInternal(
      x: Array3D, dy: Array3D, weights: Array4D, stride: number,
      pad: number): {dx: Array3D, dw: Array4D, db: Array1D};

  /**
   * Computes the transposed 2D convolution of an image, also known as a
   * deconvolution.
   * @param x The input image, must be rank 3, of shape [xrows, xcols, depth1].
   * @param weights The weights NDArray, must be rank 4, of shape [f, f, depth1,
   * depth2].
   * @param biases Optional biases NDArray, must be rank 1 of shape [depth2].
   * @param stride The stride of the convolution.
   * @param pad The padding of each side of the input NDArray. Will pad equally
   * on all sides.
   */
  conv2dTranspose(
      x: Array3D, weights: Array4D, biases: Array1D|null, stride: number,
      pad: number): Array3D {
    util.assert(
        x.rank === 3,
        `Error in conv2dTranspose: x must be rank 3, but got rank ` +
            `${x.rank}.`);
    util.assert(
        weights.rank === 4,
        `Error in conv2dTranspose: weights must be rank 4, but got ` +
            `rank ${weights.rank}`);
    if (biases != null) {
      util.assert(
          biases.rank === 1,
          `Error in conv2dTranspose: biases must be rank 1, but got ' +
              'rank ${biases.rank}.`);
    }
    util.assert(
        x.shape[2] === weights.shape[3],
        `Error in conv2dTranspose: depth of input (${x.shape[2]}) must ` +
            `match input depth for weights ${weights.shape[3]}.`);

    return this.track(
        this.conv2dTransposeInternal(x, weights, biases, stride, pad));
  }
  protected abstract conv2dTransposeInternal(
      x: Array3D, weights: Array4D, biases: Array1D|null, stride: number,
      pad: number): Array3D;

  /**
   * Computes the 2D max pooling of an image.
   * @param x The input image, must be rank 3.
   * @param fSize The field size of the max pool.
   * @param stride The stride of the max pool.
   * @param pad The padding of each side of the input NDArray. Will pad equally
   * on all sides.
   */
  maxPool(x: Array3D, fSize: number, stride: number, pad: number): Array3D {
    util.assert(
        x.rank === 3,
        'Error in maxPool: x must be rank 3 but got rank ' + x.rank + '.');
    return this.track(this.maxPoolInternal(x, fSize, stride, pad));
  }
  protected abstract maxPoolInternal(
      x: Array3D, fSize: number, stride: number, pad: number): Array3D;

  /**
   * Computes the backprop of a max pool.
   * @param dy The dy error.
   * @param x The input image, must be rank 3.
   * @param fSize The field size of the max pool.
   * @param stride The stride of the max pool.
   * @param pad The padding of each side of the input NDArray. Will pad equally
   * on all sides.
   */
  maxPoolBackprop(
      dy: Array3D, x: Array3D, fSize: number, stride: number,
      pad: number): Array3D {
    util.assert(
        dy.rank === 3,
        `Error in maxPoolBackprop: dy must be rank 3 but got rank ` +
            `${dy.rank}.`);
    util.assert(
        x.rank === 3,
        `Error in maxPoolBackprop: x must be rank 3 but got rank ` +
            `${x.rank}.`);

    return this.track(this.maxPoolBackpropInternal(dy, x, fSize, stride, pad));
  }
  protected abstract maxPoolBackpropInternal(
      dy: Array3D, x: Array3D, fSize: number, stride: number,
      pad: number): Array3D;

  /**
   * Computes the 2D min pooling of an image.
   * @param x The input image, must be rank 3.
   * @param fSize The field size of the max pool.
   * @param stride The stride of the max pool.
   * @param pad The padding of each side of the input NDArray. Will pad equally
   * on all sides.
   */
  minPool(x: Array3D, fSize: number, stride: number, pad: number): Array3D {
    util.assert(
        x.rank === 3,
        `Error in minPool: x must be rank 3 but got rank ${x.rank}.`);
    return this.track(this.minPoolInternal(x, fSize, stride, pad));
  }
  protected abstract minPoolInternal(
      x: Array3D, fSize: number, stride: number, pad: number): Array3D;

  /**
   * Computes the 2D average pooling of an image.
   * @param x The input image, must be rank 3.
   * @param fSize The field size of the max pool.
   * @param stride The stride of the max pool.
   * @param pad The padding of each side of the input NDArray. Will pad equally
   * on all sides.
   */
  avgPool(x: Array3D, fSize: number, stride: number, pad: number): Array3D {
    util.assert(
        x.rank === 3,
        `Error in avgPool: x must be rank 3 but got rank ${x.rank}.`);
    return this.track(this.avgPoolInternal(x, fSize, stride, pad));
  }
  protected abstract avgPoolInternal(
      x: Array3D, fSize: number, stride: number, pad: number): Array3D;

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
    return this.track(
        this.resizeBilinear3DInternal(x, newShape2D, alignCorners));
  }
  protected abstract resizeBilinear3DInternal(
      x: Array3D, newShape2D: [number, number], alignCorners: boolean): Array3D;

  /**
   * Batch normalization 3D. Mean, variance, scale, and offset can be of two
   * shapes: 1) The same shape as the input: an Array3D. 2) In the common case,
   * the depth dimension is the last dimension of x, so the values would be an
   * Array1D of shape [depth].
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

    return this.track(this.batchNormalization3DInternal(
        x, mean, variance, varianceEpsilon, scale, offset));
  }
  protected abstract batchNormalization3DInternal(
      x: Array3D, mean: Array3D|Array1D, variance: Array3D|Array1D,
      varianceEpsilon: number, scale?: Array3D|Array1D,
      offset?: Array3D|Array1D): Array3D;

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
    util.assert(
        data.shape[0] === 1,
        `Error in multiRNNCell: first dimension of data is ${data.shape[0]}, ` +
            `but batch sizes > 1 are not yet supported.`);
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
      newC.push(res[i] as Array2D);
      newH.push(res[i + 1] as Array2D);
    }
    return [newC, newH];
  }

  /**
   * Computes the next state and output of a BasicLSTMCell.
   * This is only the forward mode.
   * Derived from tf.contrib.rnn.BasicLSTMCell.
   * @param forgetBias Forget bias for the cell.
   * @param lstmKernel The weights for the cell.
   * @param lstmBias The biases for the cell.
   * @param data The input to the cell.
   * @param c Previous cell state.
   * @param h Previous cell output.
   * @return Tuple [nextCellState, cellOutput]
   */
  basicLSTMCell(
      forgetBias: Scalar, lstmKernel: Array2D, lstmBias: Array1D, data: Array2D,
      c: Array2D, h: Array2D): [Array2D, Array2D] {
    const res = this.scope(() => {
      util.assert(
          data.shape[0] === 1,
          `Error in multiRNNCell: first dimension of data is ` +
              `${data.shape[0]}, but batch sizes > 1 are not yet supported.`);
      // concat(inputs, h, 1)
      // There is no concat1d, so reshape inputs and h to 3d, concat, then
      // reshape back to 2d.
      const data3D = data.as3D(1, 1, data.shape[1]);
      const h3D = h.as3D(1, 1, h.shape[1]);
      const combined3D = this.concat3D(data3D, h3D, 2);
      const combined2D = combined3D.as2D(1, data.shape[1] + h.shape[1]);

      const weighted = this.matMul(combined2D, lstmKernel);
      const res = this.add(weighted, lstmBias) as Array2D;

      // i = input_gate, j = new_input, f = forget_gate, o = output_gate
      const i = this.slice2D(res, [0, 0], [res.shape[0], res.shape[1] / 4]);
      const j = this.slice2D(
          res, [0, res.shape[1] / 4 * 1], [res.shape[0], res.shape[1] / 4]);
      const f = this.slice2D(
          res, [0, res.shape[1] / 4 * 2], [res.shape[0], res.shape[1] / 4]);
      const o = this.slice2D(
          res, [0, res.shape[1] / 4 * 3], [res.shape[0], res.shape[1] / 4]);

      const newC =
          this.add(
              this.multiplyStrict(
                  c, this.sigmoid(this.scalarPlusArray(forgetBias, f))),
              this.multiplyStrict(this.sigmoid(i), this.tanh(j))) as Array2D;
      const newH =
          this.multiplyStrict(this.tanh(newC), this.sigmoid(o)) as Array2D;

      return [newC, newH];
    });
    return [res[0], res[1]];
  }
}

export enum MatrixOrientation {
  REGULAR,
  TRANSPOSED
}
