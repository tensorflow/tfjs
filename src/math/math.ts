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
// tslint:disable-next-line:max-line-length
import {customGradient, gradients, valueAndGradients, variableGradients, vjp} from './backends/gradients';
import {ScopeResult} from './backends/tape_util';
import {keep, tidy} from './backends/tracking';
import * as batchnorm from './batchnorm';
import * as binary_ops from './binary_ops';
import * as compare from './compare';
import * as concat from './concat';
import * as conv from './conv';
import * as image_ops from './image_ops';
import * as logical from './logical_ops';
import * as lstm_ops from './lstm';
import * as matmul from './matmul';
import {Array1D, Array3D, Array4D, NDArray, Scalar} from './ndarray';
import * as norm from './norm';
import * as ops from './ops';
import * as pool from './pool';
import * as reduction_ops from './reduction_ops';
import * as reverse from './reverse';
import * as slice from './slice';
import * as softmax_ops from './softmax';
import * as transpose from './transpose';
import {Rank} from './types';
import * as unary_ops from './unary_ops';

export class NDArrayMath {
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

  logicalNot = logical.Ops.logicalNot;
  logicalAnd = logical.Ops.logicalAnd;
  logicalOr = logical.Ops.logicalOr;
  logicalXor = logical.Ops.logicalXor;
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

  // Tracking methods.
  keep = keep;

  // Gradient methods.
  customGradient = customGradient;
  gradients = gradients;
  valueAndGradients = valueAndGradients;
  variableGradients = variableGradients;
  vjp = vjp;

  register: typeof ENV.engine.register;
  engine: typeof ENV.engine;
  getNumArrays: typeof ENV.engine.getNumArrays;
  dispose: typeof ENV.engine.dispose;
  registeredVariables: typeof ENV.engine.registeredVariables;
  write: typeof ENV.engine.write;
  read: typeof ENV.engine.read;
  readSync: typeof ENV.engine.readSync;
  disposeData: typeof ENV.engine.disposeData;
  registerVariable: typeof ENV.engine.registerVariable;
  startScope: typeof ENV.engine.startScope;
  endScope: typeof ENV.engine.endScope;

  /**
   * @param safeMode In safe mode, you must use math operations inside
   *     a dl.tidy() which will automatically clean up intermediate NDArrays.
   */
  constructor(backend: BackendType|MathBackend, safeMode: boolean) {
    ENV.setMath(this, backend, safeMode);
    this.register = ENV.engine.register.bind(ENV.engine);
    this.engine = ENV.engine;
    this.getNumArrays = ENV.engine.getNumArrays.bind(ENV.engine);
    this.dispose = ENV.engine.dispose.bind(ENV.engine);
    this.registeredVariables = ENV.engine.registeredVariables;
    this.write = ENV.engine.write.bind(ENV.engine);
    this.read = ENV.engine.read.bind(ENV.engine);
    this.readSync = ENV.engine.readSync.bind(ENV.engine);
    this.disposeData = ENV.engine.disposeData.bind(ENV.engine);
    this.registerVariable = ENV.engine.registerVariable.bind(ENV.engine);
    this.startScope = ENV.engine.startScope.bind(ENV.engine);
    this.endScope = ENV.engine.endScope.bind(ENV.engine);
  }

  /** @deprecated Use dl.tidy() */
  scope<T extends ScopeResult>(scopeFn?: ScopeFn<T>): T {
    const keepFn = <T extends NDArray>(ndarray: T): T => keep(ndarray);
    const trackFn = <T extends NDArray>(ndarray: T): T => ndarray;
    return tidy(() => scopeFn(keepFn, trackFn));
  }

  /** @deprecated This is a no-op. */
  track<T extends NDArray>(result: T): T {
    return result;
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
    tidy('topK', () => {
      values = ENV.engine.executeKernel('TopKValues', {inputs: {x}, args: {k}});
      indices =
          ENV.engine.executeKernel('TopKIndices', {inputs: {x}, args: {k}});
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

    return tidy('scaledArrayAdd', () => {
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

    return ENV.engine.executeKernel(
        'LRN4D', {inputs: {x}, args: {radius, bias, alpha, beta, normRegion}});
  }
}

export type ScopeFn<T extends ScopeResult> =
    (keep: <T1 extends NDArray>(ndarray: T1) => T1,
     track: <T2 extends NDArray>(ndarray: T2) => T2) => T;
