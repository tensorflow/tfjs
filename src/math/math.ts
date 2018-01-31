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

import * as array_ops from './array_ops';
import {MathBackend} from './backends/backend';
import {BackendEngine} from './backends/backend_engine';
import {TapeNodeInputGradientArrays} from './backends/tape_types';
import {ScopeFn, ScopeResult, ScopeResultImmediate} from './backends/tape_util';
import * as batchnorm from './batchnorm';
import * as binary_ops from './binary_ops';
import * as compare from './compare';
import * as concat from './concat';
import * as conv from './conv';
import * as image_ops from './image_ops';
import * as logical from './logical_ops';
import * as lstm_ops from './lstm';
import * as matmul from './matmul';
// tslint:disable-next-line:max-line-length
import {Array1D, Array3D, Array4D, NDArray, Scalar, Variable} from './ndarray';
import * as norm from './norm';
import * as ops from './ops';
import * as pool from './pool';
import * as reduction_ops from './reduction_ops';
import * as reverse from './reverse';
import * as slice from './slice';
import * as softmax_ops from './softmax';
import * as transpose from './transpose';
import {NamedArrayMap, NamedVariableMap} from './types';
import {Rank, TypedArray} from './types';
import * as unary_ops from './unary_ops';

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

  reverse = reverse.Ops.reverse;
  reverse1D = reverse.Ops.reverse1D;
  reverse2D = reverse.Ops.reverse2D;
  reverse3D = reverse.Ops.reverse3D;
  reverse4D = reverse.Ops.reverse4D;

  concat = concat.Ops.concat;
  concat1D = concat.Ops.concat1D;
  concat2D = concat.Ops.concat2D;
  concat3D = concat.Ops.concat3D;
  concat4D = concat.Ops.concat4D;

  batchNormalization = batchnorm.Ops.batchNormalization;
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
  moments = reduction_ops.Ops.moments;
  sum = reduction_ops.Ops.sum;

  add = binary_ops.Ops.add;
  addStrict = binary_ops.Ops.addStrict;
  /** @deprecated */
  arrayDividedByScalar = binary_ops.Ops.arrayDividedByScalar;
  div = binary_ops.Ops.div;
  divide = this.div;  // Alias.
  divStrict = binary_ops.Ops.divStrict;
  divideStrict = this.divStrict;  // Alias.
  /** @deprecated */
  elementWiseMul = binary_ops.Ops.elementWiseMul;
  maximum = binary_ops.Ops.maximum;
  maximumStrict = binary_ops.Ops.maximumStrict;
  minimum = binary_ops.Ops.minimum;
  minimumStrict = binary_ops.Ops.minimumStrict;
  mul = binary_ops.Ops.mul;
  multiply = this.mul;  // Alias.
  mulStrict = binary_ops.Ops.mulStrict;
  multiplyStrict = this.mulStrict;  // Alias.
  pow = binary_ops.Ops.pow;
  powStrict = binary_ops.Ops.powStrict;
  /** @deprecated */
  scalarDividedByArray = binary_ops.Ops.scalarDividedByArray;
  sub = binary_ops.Ops.sub;
  subtract = this.sub;  // Alias.
  subStrict = binary_ops.Ops.subStrict;

  logicalAnd = logical.Ops.logicalAnd;
  logicalOr = logical.Ops.logicalOr;
  where = logical.Ops.where;

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

  basicLSTMCell = lstm_ops.Ops.basicLSTMCell;
  multiRNNCell = lstm_ops.Ops.multiRNNCell;

  softmax = softmax_ops.Ops.softmax;
  softmaxCrossEntropy = softmax_ops.Ops.softmaxCrossEntropy;

  cast = array_ops.Ops.cast;
  clone = array_ops.Ops.clone;
  gather = array_ops.Ops.gather;
  reshape = array_ops.Ops.reshape;
  tile = array_ops.Ops.tile;
  oneHot = array_ops.Ops.oneHot;
  multinomial = array_ops.Ops.multinomial;
  pad1D = array_ops.Ops.pad1D;
  pad2D = array_ops.Ops.pad2D;

  /** @deprecated Use dl.image.resizeBilinear() */
  resizeBilinear3D = image_ops.Ops.resizeBilinear;

  // Public since optimizers will use it.
  registeredVariables: NamedVariableMap = {};

  time(f: () => void): Promise<number> {
    return this.backend.time(f);
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

  fromPixels(
      pixels: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      numChannels: number): Array3D {
    return this.backend.fromPixels(pixels, numChannels);
  }
  write(dataId: number, values: TypedArray): void {
    this.backend.write(dataId, values);
  }
  readSync(dataId: number): TypedArray {
    return this.backend.readSync(dataId);
  }
  read(dataId: number): Promise<TypedArray> {
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
    ENV.set('DEBUG', true);

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
   * Computes the top K values and flattened indices.
   * @param x The input NDArray.
   * @param k How many top values to compute.
   */
  topK(x: NDArray, k: number): {values: Array1D, indices: Array1D} {
    util.assert(
        k <= x.size,
        `Error in topK: k value (${k}) must be less than size of input ` +
            `ndarray, got shape ${x.shape}.`);
    let values: Array1D;
    let indices: Array1D;
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

  /** @deprecated Use math.transpose() instead. */
  switchDim<R extends Rank>(x: NDArray<R>, perm?: number[]): NDArray<R> {
    return ops.transpose<R>(x, perm);
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
      f: () => NDArray<R>, x: T, dy: NDArray<R>): T {
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
  gradients<T extends NDArray|NamedArrayMap>(f: () => Scalar, x: T): T {
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
  variableGradients(f: () => Scalar, varList?: Variable[]):
      {value: Scalar, gradients: NamedArrayMap} {
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
  valueAndGradients<T extends NDArray|NamedArrayMap>(f: () => Scalar, x: T):
      {value: Scalar, gradients: T} {
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
  customGradient<R extends Rank, T extends NDArray<R>>(
      name: string, f: () => {
        value: T,
        gradients: (dy: T, y: T) => TapeNodeInputGradientArrays
      },
      inputs: NamedArrayMap): T {
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
