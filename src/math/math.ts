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

import {BackendType, ENV} from '../environment';
import * as util from '../util';
import {NamedArrayMap, NamedVariableMap} from '../util';

import * as axis_util from './axis_util';
import {MathBackend} from './backends/backend';
import {BackendEngine} from './backends/backend_engine';
import {TapeNodeInputGradientArrays} from './backends/tape_types';
import {ScopeFn, ScopeResult, ScopeResultImmediate} from './backends/tape_util';
import * as batchnorm from './batchnorm';
import * as binary_ops from './binary_ops';
import * as broadcast_util from './broadcast_util';
import * as compare from './compare';
import * as concat from './concat';
import * as conv from './conv';
import * as matmul from './matmul';
// tslint:disable-next-line:max-line-length
import {Array1D, Array2D, Array3D, Array4D, NDArray, Scalar, Variable} from './ndarray';
import * as norm from './norm';
import * as pool from './pool';
import * as reduction_ops from './reduction_ops';
import * as reverse from './reverse';
import * as slice from './slice';
import * as transpose from './transpose';
import * as types from './types';
import {DataType, DataTypeMap, Rank, RankMap} from './types';
import * as unary_ops from './unary_ops';

export interface LSTMCell {
  (data: Array2D, c: Array2D, h: Array2D): [Array2D, Array2D];
}

export interface NDArrayManager {
  getNumArrays(): number;
  register(a: NDArray): void;
  registerVariable(v: Variable): void;
}

export class NDArrayMath implements NDArrayManager {
  engine: BackendEngine;
  private registeredArrays = new Map<number, number>();
  private backend: MathBackend;
  private customBackend = false;

  // Ops.
  matMul = matmul.Ops.matMul;
  vectorTimesMatrix = matmul.Ops.vectorTimesMatrix;
  outerProduct = matmul.Ops.outerProduct;
  matrixTimesVector = matmul.Ops.matrixTimesVector;
  dotProduct = matmul.Ops.dotProduct;

  slice = slice.Ops.slice;
  slice1D = slice.Ops.slice1D;
  slice2D = slice.Ops.slice2D;
  slice3D = slice.Ops.slice3D;
  slice4D = slice.Ops.slice4D;

  reverse1D = reverse.Ops.reverse1D;
  reverse2D = reverse.Ops.reverse2D;
  reverse3D = reverse.Ops.reverse3D;
  reverse4D = reverse.Ops.reverse4D;

  concat1D = concat.Ops.concat1D;
  concat2D = concat.Ops.concat2D;
  concat3D = concat.Ops.concat3D;
  concat4D = concat.Ops.concat4D;

  batchNormalization2D = batchnorm.Ops.batchNormalization2D;
  batchNormalization3D = batchnorm.Ops.batchNormalization3D;
  batchNormalization4D = batchnorm.Ops.batchNormalization4D;

  avgPool = pool.Ops.avgPool;
  maxPool = pool.Ops.maxPool;
  minPool = pool.Ops.minPool;
  /** @deprecated */
  maxPoolBackprop = pool.Ops.maxPoolBackprop;

  conv1d = conv.Ops.conv1d;
  conv2d = conv.Ops.conv2d;
  conv2dTranspose = conv.Ops.conv2dTranspose;
  depthwiseConv2D = conv.Ops.depthwiseConv2D;
  /** @deprecated */
  conv2dDerBias = conv.Ops.conv2dDerBias;
  /** @deprecated */
  conv2dDerFilter = conv.Ops.conv2dDerFilter;
  /** @deprecated */
  conv2dDerInput = conv.Ops.conv2dDerInput;

  argMax = reduction_ops.Ops.argMax;
  argMaxEquals = reduction_ops.Ops.argMaxEquals;
  argMin = reduction_ops.Ops.argMin;
  logSumExp = reduction_ops.Ops.logSumExp;
  max = reduction_ops.Ops.max;
  mean = reduction_ops.Ops.mean;
  min = reduction_ops.Ops.min;
  sum = reduction_ops.Ops.sum;

  add = binary_ops.Ops.add;
  addStrict = binary_ops.Ops.addStrict;
  /** @deprecated */
  arrayDividedByScalar = binary_ops.Ops.arrayDividedByScalar;
  divide = binary_ops.Ops.divide;
  divideStrict = binary_ops.Ops.divideStrict;
  /** @deprecated */
  elementWiseMul = binary_ops.Ops.elementWiseMul;
  maximum = binary_ops.Ops.maximum;
  minimum = binary_ops.Ops.minimum;
  multiply = binary_ops.Ops.multiply;
  multiplyStrict = binary_ops.Ops.multiplyStrict;
  pow = binary_ops.Ops.pow;
  powStrict = binary_ops.Ops.powStrict;
  /** @deprecated */
  scalarDividedByArray = binary_ops.Ops.scalarDividedByArray;
  /** @deprecated */
  sub = binary_ops.Ops.sub;
  subStrict = binary_ops.Ops.subStrict;
  subtract = binary_ops.Ops.subtract;

  transpose = transpose.Ops.transpose;

  equal = compare.Ops.equal;
  equalStrict = compare.Ops.equalStrict;
  greater = compare.Ops.greater;
  greaterStrict = compare.Ops.greaterStrict;
  greaterEqual = compare.Ops.greaterEqual;
  greaterEqualStrict = compare.Ops.greaterEqualStrict;
  less = compare.Ops.less;
  lessStrict = compare.Ops.lessStrict;
  lessEqual = compare.Ops.lessEqual;
  lessEqualStrict = compare.Ops.lessEqualStrict;
  notEqual = compare.Ops.notEqual;
  notEqualStrict = compare.Ops.notEqualStrict;

  abs = unary_ops.Ops.abs;
  acos = unary_ops.Ops.acos;
  asin = unary_ops.Ops.asin;
  atan = unary_ops.Ops.atan;
  ceil = unary_ops.Ops.ceil;
  clip = unary_ops.Ops.clip;
  cos = unary_ops.Ops.cos;
  cosh = unary_ops.Ops.cosh;
  elu = unary_ops.Ops.elu;
  exp = unary_ops.Ops.exp;
  floor = unary_ops.Ops.floor;
  leakyRelu = unary_ops.Ops.leakyRelu;
  log = unary_ops.Ops.log;
  neg = unary_ops.Ops.neg;
  prelu = unary_ops.Ops.prelu;
  relu = unary_ops.Ops.relu;
  selu = unary_ops.Ops.selu;
  sigmoid = unary_ops.Ops.sigmoid;
  sin = unary_ops.Ops.sin;
  sinh = unary_ops.Ops.sinh;
  sqrt = unary_ops.Ops.sqrt;
  square = unary_ops.Ops.square;
  step = unary_ops.Ops.step;
  tan = unary_ops.Ops.tan;
  tanh = unary_ops.Ops.tanh;

  norm = norm.Ops.norm;

  // Public since optimizers will use it.
  registeredVariables: NamedVariableMap = {};

  time(query: () => NDArray): Promise<number> {
    return this.backend.time(query);
  }

  getNumArrays() {
    return this.registeredArrays.size;
  }

  register(a: NDArray|Variable): void {
    const refCount = this.registeredArrays.has(a.dataId) ?
        this.registeredArrays.get(a.dataId) :
        0;
    if (refCount === 0) {
      this.backend.register(a.dataId, a.shape, a.dtype);
    }
    this.registeredArrays.set(a.dataId, refCount + 1);
    if (!(a instanceof Variable)) {
      this.engine.track(a);
    }
  }

  registerVariable(v: Variable) {
    if (this.registeredVariables[v.name] != null) {
      throw new Error(`Variable with name ${v.name} was already registered`);
    }
    this.registeredVariables[v.name] = v;
  }

  writePixels(
      dataId: number,
      pixels: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      numChannels: number): void {
    this.backend.writePixels(dataId, pixels, numChannels);
  }
  write<D extends DataType>(dataId: number, values: DataTypeMap[D]): void {
    this.backend.write(dataId, values);
  }
  readSync<D extends DataType>(dataId: number): DataTypeMap[D] {
    return this.backend.readSync(dataId);
  }
  read<D extends DataType>(dataId: number): Promise<DataTypeMap[D]> {
    return this.backend.read(dataId);
  }

  /**
   * @param safeMode In safe mode, you must use math operations inside
   *     a math.scope() which will automatically clean up intermediate NDArrays.
   */
  constructor(backend: BackendType|MathBackend, safeMode: boolean) {
    if (typeof backend === 'string') {
      this.backend = ENV.getBackend(backend);
    } else {
      this.customBackend = true;
      this.backend = backend;
    }
    this.engine = new BackendEngine(this.backend, safeMode);
    ENV.setMath(this);
  }

  /**
   * In debug mode, the output of every math call will be downloaded to the CPU
   * and checked for NaNs. This significantly impacts performance.
   */
  enableDebugMode() {
    this.engine.enableDebugMode();
    console.warn(
        'Debugging mode is ON. The output of every math call will ' +
        'be downloaded to CPU and checked for NaNs. ' +
        'This significantly impacts performance.');
  }

  /**
   * Executes the provided function and after it is executed, cleans up all
   * intermediate NDArrays allocated by the function except those returned by
   * the function.
   *
   * When in safe mode, you must enclose all `NDArray` creation and math ops
   * inside a `math.scope()` to prevent memory leaks.
   *
   * @param nameOrScopeFn The name of the scope, or the function to execute.
   *     If a name is provided, the 2nd argument should be the function.
   *     If a name is provided, and debug mode is on, the timing and the memory
   *     usage of the function will be tracked and displayed on the console
   *     using the provided name.
   * @param scopeFn The function to execute.
   * @param gradientsMode If true, enables gradients mode.
   *     See math.gradientsScope for details.
   */
  scope<T extends ScopeResult>(
      nameOrScopeFn: string|ScopeFn<T>, scopeFn?: ScopeFn<T>,
      gradientsMode = false): T {
    if (scopeFn == null) {
      // Called with only 1 argument.
      if (typeof nameOrScopeFn !== 'function') {
        throw new Error('Please provide a function to math.scope()');
      }
      scopeFn = nameOrScopeFn;
      nameOrScopeFn = 'scope';
    } else {
      // Called with 2 arguments.
      if (typeof nameOrScopeFn !== 'string' &&
          !(nameOrScopeFn instanceof String)) {
        throw new Error(
            'When calling with two arguments, the first argument ' +
            'to math.scope() must be a string');
      }
      if (typeof scopeFn !== 'function') {
        throw new Error(
            'When calling with two arguments, the 2nd argument ' +
            'to math.scope() must be a function');
      }
      // TODO(nsthorat,smilkov): Do operation logging and performance profiling.
    }
    return this.engine.scope(nameOrScopeFn as string, scopeFn, gradientsMode);
  }

  /**
   * Create a new gradients scope. Similar to scope, but forces all inner scopes
   * to not clean up so that gradient operations can be used inside of this
   * scope.
   * @param nameOrScopeFn The name of the scope, or the function to execute.
   *     If a name is provided, the 2nd argument should be the function.
   *     If a name is provided, and debug mode is on, the timing and the memory
   *     usage of the function will be tracked and displayed on the console
   *     using the provided name.
   * @param scopeFn The function to execute.
   */
  gradientsScope<T extends ScopeResult>(
      nameOrScopeFn: string|ScopeFn<T>, scopeFn?: ScopeFn<T>): T {
    const gradientsMode = true;
    return this.scope(nameOrScopeFn, scopeFn, gradientsMode);
  }

  /**
   * Start a scope. Use this with endScope() to achieve the same functionality
   * as scope() without the need for a function closure.
   */
  startScope() {
    const gradientsMode = false;
    this.engine.startScope(gradientsMode);
  }

  /**
   * End a scope. Use this with startScope() to achieve the same functionality
   * as scope() without the need for a function closure.
   */
  endScope(result: ScopeResultImmediate) {
    const gradientsMode = false;
    this.engine.endScope(result, gradientsMode);
  }

  /**
   * Keeps an NDArray in the current scope from being disposed automatically.
   * @param result The NDArray to keep from being disposed.
   */
  keep<T extends NDArray>(result: T): T {
    return this.engine.keep(result);
  }

  /** @deprecated This is a no-op. */
  track<T extends NDArray>(result: T): T {
    return result;
  }

  dispose() {
    if (this.customBackend) {
      this.backend.dispose();
    }
  }

  /**
   * Clones an NDArray of any shape.
   * @param x The NDArray to clone.
   */
  clone<T extends NDArray>(x: T): T {
    return this.engine.executeKernel('Clone', {inputs: {x}}) as T;
  }

  /** Reshapes the array. */
  reshape<D extends DataType, R extends Rank, T extends RankMap<D>[R]>(
      x: NDArray<D>, newShape: number[]): T {
    newShape = util.inferFromImplicitShape(newShape, x.size);
    util.assert(
        x.size === util.sizeFromShape(newShape),
        'new shape and old shape must have the same number of elements.');

    const grad = (dy: NDArray<'float32'>, y: NDArray) => {
      return {x: () => dy.reshape(x.shape)};
    };
    return this.engine.executeKernel(
               'Reshape', {inputs: {x}, args: {newShape}}, grad) as T;
  }

  /**
   * Casts a tensor to a new type. If the new type matches the old type,
   * this is a no-op.
   */
  cast<D extends DataType, R extends Rank>(
      x: NDArray<DataType, R>, newDType: D): RankMap<D>[R] {
    const grad = (dy: NDArray<'float32'>, y: NDArray) => {
      return {x: () => dy.reshape(dy.shape)};
    };
    return this.engine.executeKernel(
               'Cast', {inputs: {x}, args: {newDType}}, grad) as RankMap<D>[R];
  }

  /**
   * Returns the truth value of a AND b element-wise. Supports broadcasting.
   *
   * @param a The first input `NDArray<'bool'>`.
   * @param b The second input `NDArray<'bool'>`.
   */
  logicalAnd(a: NDArray<'bool'>, b: NDArray<'bool'>): NDArray<'bool'> {
    util.assert(
        a.dtype === 'bool' && b.dtype === 'bool',
        'Error Array must be of type bool.');
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return this.engine.executeKernel('LogicalAnd', {inputs: {a, b}});
  }

  /**
   * Returns the truth value of a OR b element-wise. Supports broadcasting.
   *
   * @param a The first input `NDArray<'bool'>`.
   * @param b The second input `NDArray<'bool'>`.
   */
  logicalOr(a: NDArray<'bool'>, b: NDArray<'bool'>): NDArray<'bool'> {
    util.assert(
        a.dtype === 'bool' && b.dtype === 'bool',
        'Error Array must be of type bool.');
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return this.engine.executeKernel('LogicalOr', {inputs: {a, b}});
  }

  /**
   * Returns the elements, either `a` or `b` depending on the `condition`.
   *
   * @param condition The input as `NDAray<'bool'>.
   * @param a Input as `NDArray` which may have the same shape as
   *     `condition`. If `condition` is rank 1, `a` may have a higher rank but
   *     its first dimension must match the size of `condition`.
   * @param b Input as `NDArray` with the same shape and type as `a`.
   * @return An `NDArray` with the same type and shape as `a` and `b`.
   */
  where<T extends NDArray>(condition: NDArray<'bool'>, a: T, b: T): T {
    util.assert(
        condition.dtype === 'bool' || a.dtype === 'bool' || b.dtype === 'bool',
        'Error Array must be of type bool.');

    util.assertShapesMatch(a.shape, b.shape, 'Error in where: ');

    if (condition.rank === 1) {
      // If condition rank is 1, then the first dimension must match the size of
      // condition.
      util.assert(
          condition.shape[0] === a.shape[0],
          'The first dimension of `a` must match the size of `condition`.');
    } else {
      // A must have the same shape as condition.
      util.assertShapesMatch(condition.shape, b.shape, 'Error in where: ');
    }

    // Default to highest percision of number:
    const dtype = types.upcastType(a.dtype, b.dtype);
    return this.engine.executeKernel(
               'Where',
               {inputs: {condition, a, b}, args: {dtype: dtype as DataType}}) as
        T;
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
    this.scope('topK', () => {
      values =
          this.engine.executeKernel('TopKValues', {inputs: {x}, args: {k}});
      indices =
          this.engine.executeKernel('TopKIndices', {inputs: {x}, args: {k}});
      return values;
    });
    const result = {values, indices};
    return result;
  }

  /**
   * Computes the softmax normalized vector given the logits.
   * @param logits The logits array.
   * @param dim The dimension softmax would be performed on. Defaults to -1
   *     which indicates the last dimension.
   */
  softmax<D extends DataType, R extends Rank, T extends NDArray<'float32', R>>(
      logits: NDArray<D, R>, dim = -1): RankMap<'float32'>[R] {
    if (dim === -1) {
      dim = logits.rank - 1;
    }
    if (dim !== logits.rank - 1) {
      throw Error(
          'Softmax along a non-last dimension is not yet supported. ' +
          `Logits was rank ${logits.rank} and dim was ${dim}`);
    }

    const gradients = (dy: T, y: T) => {
      return {
        logits: () => {
          const dyTimesY = this.multiply(dy, y);
          const keepDims = true;
          return this.subtract(
                     dyTimesY,
                     this.multiply(this.sum(dyTimesY, [dim], keepDims), y)) as
              NDArray<'float32', R>;
        }
      };
    };

    return this.scope('softmax', () => {
      return this.customGradient(() => {
        // Do it in log space for numerical stability.
        // exp(X - logSumExp(X))
        const keepDims = true;
        const lse = this.logSumExp(logits, [dim], keepDims);
        const logResult = this.subtract(logits.asType('float32'), lse);
        const value = this.exp(logResult);
        return {value, gradients};
      }, {logits}, 'softmax') as RankMap<'float32'>[R];
    });
  }

  /**
   * Computes softmax cross entropy between logits and labels.
   *
   * Measures the probability error in discrete classification tasks in which
   * the classes are mutually exclusive (each entry is in exactly one class).
   * For example, each CIFAR-10 image is labeled with one and only one label: an
   * image can be a dog or a truck, but not both.
   *
   * NOTE: While the classes are mutually exclusive, their probabilities need
   * not be. All that is required is that each row of labels is a valid
   * probability distribution. If they are not, the computation of the gradient
   * will be incorrect.
   *
   * WARNING: This op expects unscaled logits, since it performs a softmax on
   * logits internally for efficiency. Do not call this op with the output of
   * softmax, as it will produce incorrect results.
   *
   * logits and labels must have the same shape, e.g. [batch_size, num_classes]
   * and the same dtype.
   * @param labels The labels array.
   * @param logits The logits array.
   * @param dim The dimension softmax would be performed on. Defaults to -1
   *     which indicates the last dimension.
   */
  softmaxCrossEntropyWithLogits<
      R extends Rank, A extends NDArray<DataType, R>, B extends
          NDArray<DataType, R>, O extends NDArray<'float32'>>(
      labels: A, logits: B, dim = -1): O {
    util.assertShapesMatch(
        labels.shape, logits.shape, 'Error in softmaxCrossEntropyWithLogits: ');
    if (dim === -1) {
      dim = logits.rank - 1;
    }
    if (dim !== logits.rank - 1) {
      throw Error(
          `Softmax cross entropy along a non-last dimension is not yet ` +
          `supported. Labels / logits was rank ${logits.rank} ` +
          `and dim was ${dim}`);
    }

    return this.scope('softmaxCrossEntropyWithLogits', () => {
      // Use a custom gradient for numerical stability.
      return this.customGradient(() => {
        const softmaxLogits = this.softmax(logits, dim);
        const yPlusEps = this.add(Scalar.new(1e-5), softmaxLogits);
        const logOutput = this.log(yPlusEps);
        const tarLogOutput = this.multiply(labels, logOutput);
        const costVector = this.neg(tarLogOutput);
        const value = this.sum(costVector, [dim]) as O;

        const gradients = (dy: O, y: O) => {
          const dyShape = axis_util.expandShapeToKeepDim(dy.shape, [dim]);

          return {
            logits: () => this.multiply(
                dy.reshape(dyShape),
                this.subtract(softmaxLogits, labels.asType('float32'))),
            labels: () => this.multiply(
                dy.reshape(dyShape), this.subtract(labels, softmaxLogits))
          };
        };

        return {value, gradients};
      }, {labels, logits}, 'softmaxCrossEntropyWithLogits') as O;
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
  tile<D extends DataType, T extends NDArray<D>>(x: T, reps: number[]): T {
    util.assert(
        x.rank === reps.length,
        `Error in transpose: rank of input ${x.rank} ` +
            `must match length of reps ${reps}.`);
    return this.engine.executeKernel('Tile', {inputs: {x}, args: {reps}}) as T;
  }

  /**
   * Pads a Array1D.
   *
   * This operation will pad an array according to the `paddings` you specify.
   *
   * This operation currently only implements the `CONSTANT` mode from
   * Tensorflow's `pad` operation.
   *
   * @param x The array to pad.
   * @param paddings A tuple of ints [padLeft, padRight], how much to pad on the
   *     left and right side of the array.
   * @param constantValue The scalar pad value to use. Defaults to 0.
   */
  pad1D(x: Array1D, paddings: [number, number], constantValue = 0): Array1D {
    util.assert(
        paddings.length === 2,
        'Invalid number of paddings. Must be length of 2.');
    return this.engine.executeKernel(
        'Pad1D', {inputs: {x}, args: {paddings, constantValue}});
  }

  /**
   * Pads a Array2D.
   *
   * This operation will pad an array according to the `paddings` you specify.
   *
   * This operation currently only implements the `CONSTANT` mode from
   * Tensorflow's `pad` operation.
   *
   * @param x The array to pad.
   * @param paddings A pair of tuple ints
   *     [[padTop, padBottom], [padLeft, padRight]], how much to pad on the
   *     array.
   * @param constantValue The scalar pad value to use. Defaults to 0.
   */
  pad2D(
      x: Array2D, paddings: [[number, number], [number, number]],
      constantValue = 0): Array2D {
    util.assert(
        paddings.length === 2 && paddings[0].length === 2 &&
            paddings[1].length === 2,
        'Invalid number of paddings. Must be length of 2 each.');
    return this.engine.executeKernel(
        'Pad2D', {inputs: {x}, args: {paddings, constantValue}});
  }

  /**
   * Gather slices from array `x`'s axis `axis` according to `indices`
   *
   * @param x The array to transpose.
   * @param indices The indices of the values to extract.
   * @param axis Optional. The axis over which to select values. Defaults to 0.
   */
  gather<D extends DataType, T extends NDArray<D>>(
      x: T, indices: Array1D<'int32'>, axis = 0): T {
    return this.engine.executeKernel(
               'Gather', {inputs: {x, indices}, args: {axis}}) as T;
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

    return this.scope('scaledArrayAdd', () => {
      // TODO(nsthorat): Add an SGEMM kernel and then update this.
      return this.add(this.multiply(c1, a), this.multiply(c2, b)) as T;
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
    return this.engine.executeKernel(
        'ResizeBilinear3D', {inputs: {x}, args: {newShape2D, alignCorners}});
  }

  /**
   * Normalizes the activation of a local neighborhood across or within
   * channels.
   * @param x The input NDArray.
   * @param radius The number of adjacent channels or spatial locations of the
   *     1D normalization window. In Tensorflow this param is called
   *     'depth_radius' because only 'acrossChannels' mode is supported.
   * @param bias A constant bias term for the basis.
   * @param alpha A scale factor, usually positive.
   * @param beta An exponent.
   * @param normRegion A string from: ['acrossChannels', 'withinChannel'].
   *     Default is 'acrossChannels'.
   */
  localResponseNormalization3D(
      x: Array3D, radius = 5, bias = 1, alpha = 1, beta = 0.5,
      normRegion: 'acrossChannels'|
      'withinChannel' = 'acrossChannels'): Array3D {
    util.assert(
        x.rank === 3,
        `Error in localResponseNormalization3D: x must be rank 3 but got
         rank ${x.rank}.`);
    util.assert(
        util.isInt(radius),
        `Error in localResponseNormalization3D: radius must be an integer
         but got radius ${radius}.`);

    const input4D = x.as4D(1, x.shape[0], x.shape[1], x.shape[2]);
    const res = this.localResponseNormalization4D(
        input4D, radius, bias, alpha, beta, normRegion);
    return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
  }

  /**
   * Normalizes the activation of a local neighborhood across or within
   * channels.
   * @param x The input NDArray. The 4-D input tensor is treated as a 3-D array
   *     of 1D vectors (along the last dimension), and each vector is
   * normalized independently.
   * @param radius The number of adjacent channels or spatial locations of the
   *     1D normalization window. In Tensorflow this param is called
   *     'depth_radius' because only 'acrossChannels' mode is supported.
   * @param bias A constant bias term for the basis.
   * @param alpha A scale factor, usually positive.
   * @param beta An exponent.
   * @param normRegion A string from: ['acrossChannels', 'withinChannel'].
   *     Default is 'acrossChannels'.
   */
  localResponseNormalization4D(
      x: Array4D, radius = 5, bias = 1, alpha = 1, beta = 0.5,
      normRegion: 'acrossChannels'|
      'withinChannel' = 'acrossChannels'): Array4D {
    util.assert(
        x.rank === 4,
        `Error in localResponseNormalization4D: x must be rank 4 but got
         rank ${x.rank}.`);
    util.assert(
        util.isInt(radius),
        `Error in localResponseNormalization3D: radius must be an integer
         but got radius ${radius}.`);

    return this.engine.executeKernel(
        'LRN4D', {inputs: {x}, args: {radius, bias, alpha, beta, normRegion}});
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
    const res = this.scope('multiRNNCell', () => {
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
    const res = this.scope('basicLSTMCell', () => {
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
          this.multiplyStrict(c, this.sigmoid(this.add(forgetBias, f))),
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
    return this.scope('multinomial', () => {
      const res = this.engine.executeKernel('Multinomial', {
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
    return this.engine.executeKernel(
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
    const result = this.scope('moments', () => {
      const mean = this.mean(x, axes, keepDims);
      let keepDimsShape = mean.shape;
      if (!keepDims) {
        keepDimsShape = axis_util.expandShapeToKeepDim(mean.shape, axes);
      }
      const devSquared = this.square(
          this.subtract(x.asType('float32'), mean.reshape(keepDimsShape)));
      const variance = this.mean(devSquared, axes, keepDims);
      return {mean, variance};
    });
    return result;
  }

  /**
   * Warning: this is not fully implemented yet. Use with caution.
   *
   * Computes and returns the vector jacobian product of f(x) with respect to x.
   * This method allows you to provide a non-scalar dy to backprop from.
   *
   * @param f The function to execute. f() should return an NDArray of the same
   * shape and dtype as dy.
   * @param x The input to compute dy/dx over. This can be a single value or
   * an object mapping a string to an NDArray. If using the object mode, this
   * method will return an object of the same shape.
   */
  vjp<T extends NDArray|NamedArrayMap, R extends Rank>(
      f: () => NDArray<DataType, R>, x: T, dy: NDArray<'float32', R>): T {
    const keys = x instanceof NDArray ? null : Object.keys(x);
    const xs = util.flattenNameArrayMap(x, keys);

    const vjp = this.engine.vjp(f, xs, dy) as NDArray[];

    if (x instanceof NDArray) {
      return vjp[0] as T;
    } else {
      return util.unflattenToNameArrayMap(keys, vjp) as T;
    }
  }

  /**
   * Warning: this is not fully implemented yet. Use with caution.
   *
   * Computes and returns the gradient of f(x) with respect to x.
   *
   * @param f The function to execute. f() should return a scalar.
   *          TODO(nsthorat): Accept non-scalars.
   * @param x The input to compute de/dx over. This can be a single value or
   * an object mapping a string to an NDArray. If using the object mode, this
   * method will return an object of the same shape.
   */
  gradients<T extends NDArray|NamedArrayMap, D extends DataType>(
      f: () => Scalar<D>, x: T): T {
    const keys = x instanceof NDArray ? null : Object.keys(x);
    const xs = util.flattenNameArrayMap(x, keys);

    const returnValue = false;
    const gradients = this.engine.gradients(f, xs, returnValue) as NDArray[];

    if (x instanceof NDArray) {
      return gradients[0] as T;
    } else {
      return util.unflattenToNameArrayMap(keys, gradients) as T;
    }
  }

  /**
   * Computes and returns the gradient of f(x) with respect to the list of
   * trainable variables provided by `varList`. If no list is provided, it
   * defaults to all trainable variables.
   * @param f The function to execute. f() should return a scalar.
   * @param varList An optional list of variables to provide gradients with
   * respect to. Defaults to all trainable variables.
   */
  variableGradients<D extends DataType>(
      f: () => Scalar<D>,
      varList?: Variable[]): {value: Scalar<D>, gradients: NamedArrayMap} {
    if (varList == null) {
      // Get all of the trainable variables.
      varList = [];
      const varNames = Object.keys(this.registeredVariables);
      for (let i = 0; i < varNames.length; i++) {
        const variable = this.registeredVariables[varNames[i]];
        if (variable.trainable) {
          varList.push(variable);
        }
      }
    } else {
      // Prune non-trainable variables.
      varList = varList.filter(variable => variable.trainable);
    }

    return this.engine.variableGradientsAndValue(f, varList);
  }

  /**
   * Warning: this is not fully implemented yet. Use with caution.
   *
   * Computes and returns the gradient of f(x) with respect to x. Returns
   * both f(x) and f'(x).
   *
   * @param f The function to execute. f() should return a scalar.
   *          TODO(nsthorat): Accept non-scalars.
   * @param x The input to compute de/dx over. This can be a single value or
   * an object mapping a string to an NDArray. If using the object mode,
   * this method will return an object of the same shape.
   */
  valueAndGradients<T extends NDArray|NamedArrayMap, D extends DataType>(
      f: () => Scalar<D>, x: T): {value: Scalar, gradients: T} {
    const keys = x instanceof NDArray ? null : Object.keys(x);
    const xs = util.flattenNameArrayMap(x, keys);

    const returnValue = true;
    const valueAndGradients = this.engine.gradients(f, xs, returnValue) as
        {value: Scalar, gradients: NDArray[]};

    let gradients: T;
    if (x instanceof NDArray) {
      gradients = valueAndGradients.gradients[0] as T;
    } else {
      gradients =
          util.unflattenToNameArrayMap(keys, valueAndGradients.gradients) as T;
    }
    return {value: valueAndGradients.value, gradients};
  }

  /**
   * Evaluates a function f() with a custom gradient function f'() to use during
   * backpropagation.
   * @param f The function to evaluate in forward mode. Returns a value NDArray
   * and a gradient function closure.
   * @param inputs The inputs to compute the gradient with respect to. These
   * NDArrays should be used in f().
   * @param name An optional name for the customGradient method. Used for
   * debugging.
   */
  customGradient<D extends DataType, R extends Rank>(
      f: () => {
        value: NDArray<D, R>,
        gradients: (dy: NDArray<'float32', R>, y: NDArray<D, R>) =>
            TapeNodeInputGradientArrays
      },
      inputs: NamedArrayMap, name?: string): NDArray<D, R> {
    return this.engine.customGradient(f, inputs, name == null ? '' : name);
  }

  disposeData(dataId: number): void {
    if (!this.registeredArrays.has(dataId)) {
      return;
    }
    const refCount = this.registeredArrays.get(dataId);
    if (refCount <= 1) {
      this.registeredArrays.delete(dataId);
      this.backend.disposeData(dataId);
    } else {
      this.registeredArrays.set(dataId, refCount - 1);
    }
    // TODO(nsthorat): Construct an error and save the stack trace for
    // debugging when in debug mode. Creating a stack trace is too expensive
    // to do unconditionally.
  }
}
