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

import {BackendType, ENV} from './environment';
import {KernelBackend} from './kernels/backend';
import * as array_ops from './ops/array_ops';
import * as batchnorm from './ops/batchnorm';
import * as binary_ops from './ops/binary_ops';
import * as compare from './ops/compare';
import * as concat from './ops/concat';
import * as conv from './ops/conv';
import * as image_ops from './ops/image_ops';
import * as logical from './ops/logical_ops';
import * as lrn_ops from './ops/lrn';
import * as lstm_ops from './ops/lstm';
import * as matmul from './ops/matmul';
import * as norm from './ops/norm';
import * as ops from './ops/ops';
import * as pool from './ops/pool';
import * as reduction_ops from './ops/reduction_ops';
import * as reverse from './ops/reverse';
import * as slice from './ops/slice';
import * as softmax_ops from './ops/softmax';
import * as transpose from './ops/transpose';
import * as unary_ops from './ops/unary_ops';
import {ScopeResult} from './tape_util';
import {Scalar, Tensor, Tensor1D} from './tensor';
import {Tracking} from './tracking';
import {Rank} from './types';
import * as util from './util';

const tidy = Tracking.tidy;
const keep = Tracking.keep;

export class NDArrayMath {
  // Ops.
  matMul = matmul.Ops.matMul;
  vectorTimesMatrix = matmul.Ops.vectorTimesMatrix;
  outerProduct = matmul.Ops.outerProduct;
  matrixTimesVector = matmul.Ops.matrixTimesVector;
  dotProduct = matmul.Ops.dotProduct;

  slice = slice.Ops.slice;
  slice1D = slice.Ops.slice1d;
  slice2D = slice.Ops.slice2d;
  slice3D = slice.Ops.slice3d;
  slice4D = slice.Ops.slice4d;

  reverse = reverse.Ops.reverse;
  reverse1D = reverse.Ops.reverse1d;
  reverse2D = reverse.Ops.reverse2d;
  reverse3D = reverse.Ops.reverse3d;
  reverse4D = reverse.Ops.reverse4d;

  concat = concat.Ops.concat;
  concat1D = concat.Ops.concat1d;
  concat2D = concat.Ops.concat2d;
  concat3D = concat.Ops.concat3d;
  concat4D = concat.Ops.concat4d;

  batchNormalization = batchnorm.Ops.batchNormalization;
  batchNormalization2D = batchnorm.Ops.batchNormalization2d;
  batchNormalization3D = batchnorm.Ops.batchNormalization3d;
  batchNormalization4D = batchnorm.Ops.batchNormalization4d;

  avgPool = pool.Ops.avgPool;
  maxPool = pool.Ops.maxPool;
  minPool = pool.Ops.minPool;
  /** @deprecated */
  maxPoolBackprop = pool.Ops.maxPoolBackprop;

  conv1d = conv.Ops.conv1d;
  conv2d = conv.Ops.conv2d;
  conv2dTranspose = conv.Ops.conv2dTranspose;
  depthwiseConv2D = conv.Ops.depthwiseConv2d;
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
  div = binary_ops.Ops.div;
  divide = this.div;  // Alias.
  divStrict = binary_ops.Ops.divStrict;
  divideStrict = this.divStrict;  // Alias.
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
  clip = unary_ops.Ops.clipByValue;
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
  pad1D = array_ops.Ops.pad1d;
  pad2D = array_ops.Ops.pad2d;

  /** @deprecated Use dl.image.resizeBilinear() */
  resizeBilinear3D = image_ops.Ops.resizeBilinear;

  localResponseNormalization3D = lrn_ops.LRN.localResponseNormalization;
  localResponseNormalization4D = lrn_ops.LRN.localResponseNormalization;

  // Tracking methods.
  keep = Tracking.keep;

  engine: typeof ENV.engine;
  dispose: typeof ENV.engine.dispose;
  registeredVariables: typeof ENV.engine.registeredVariables;
  startScope: typeof ENV.engine.startScope;
  endScope: typeof ENV.engine.endScope;

  /** @deprecated */
  constructor(backend: BackendType|KernelBackend, safeMode: boolean) {
    ENV.setMath(this, backend, safeMode);
    this.engine = ENV.engine;
    this.dispose = ENV.engine.dispose.bind(ENV.engine);
    this.registeredVariables = ENV.engine.registeredVariables;
    this.startScope = ENV.engine.startScope.bind(ENV.engine);
    this.endScope = ENV.engine.endScope.bind(ENV.engine);
  }

  /** @deprecated Use dl.tidy() */
  scope<T extends ScopeResult>(scopeFn?: ScopeFn<T>): T {
    const keepFn = <T extends Tensor>(tensor: T): T => keep(tensor);
    const trackFn = <T extends Tensor>(tensor: T): T => tensor;
    return tidy(() => scopeFn(keepFn, trackFn));
  }

  /** @deprecated This is a no-op. */
  track<T extends Tensor>(result: T): T {
    return result;
  }

  /**
   * Computes the top K values and flattened indices.
   * @param x The input Tensor.
   * @param k How many top values to compute.
   */
  topK(x: Tensor, k: number): {values: Tensor1D, indices: Tensor1D} {
    util.assert(
        k <= x.size,
        `Error in topK: k value (${k}) must be less than size of input ` +
            `tensor, got shape ${x.shape}.`);
    let values: Tensor1D;
    let indices: Tensor1D;
    tidy('topK', () => {
      values = ENV.engine.executeKernel('TopKValues', {inputs: {x}, args: {k}});
      indices =
          ENV.engine.executeKernel('TopKIndices', {inputs: {x}, args: {k}});
      return values;
    });
    const result = {values, indices};
    return result;
  }

  /** @deprecated Use mulStrict() instead. */
  elementWiseMul<T extends Tensor>(a: T, b: T): T {
    return a.mulStrict(b);
  }

  /** @deprecated Use div() instead. */
  scalarDividedByArray<T extends Tensor>(c: Scalar, a: T): T {
    util.assert(
        c.size === 1,
        `Error in scalarDividedByArray: first argument must be rank 0, but ` +
            `got Tensor of rank ${c.rank}.`);
    return c.div(a) as T;
  }

  /** @deprecated Use div(A, c) instead. */
  arrayDividedByScalar<T extends Tensor>(a: T, c: Scalar): T {
    util.assert(
        c.size === 1,
        `Error in arrayDividedByScalar: second argument must be rank 0, ` +
            `but got Tensor of rank ${c.rank}.`);
    return a.div(c) as T;
  }

  /** @deprecated Use dl.transpose() instead. */
  switchDim<R extends Rank>(x: Tensor<R>, perm?: number[]): Tensor<R> {
    return ops.transpose<R>(x, perm);
  }

  /** @deprecated Use dl.add(c, A) instead. */
  scalarPlusArray<T extends Tensor>(c: Scalar, a: T): T {
    util.assert(
        c.size === 1,
        `Error in scalarPlusArray: first argument must be rank 0, but got ` +
            `rank ${c.rank}.`);
    return this.add(c, a) as T;
  }

  /** @deprecated Use dl.sub(c, A) instead. */
  scalarMinusArray<T extends Tensor>(c: Scalar, a: T): T {
    util.assert(
        c.size === 1,
        `Error in scalarMinusArray: first argument must be rank 0, but got ` +
            `rank ${c.rank}.`);
    return this.subtract(c, a) as T;
  }

  /** @deprecated Use dl.sub(A, c) instead. */
  arrayMinusScalar<T extends Tensor>(a: T, c: Scalar): T {
    util.assert(
        c.size === 1,
        `Error in arrayMinusScalar: second argument must be rank 0, but ` +
            `got rank ${c.rank}.`);
    return this.subtract(a, c) as T;
  }

  /** @deprecated */
  scaledArrayAdd<T extends Tensor>(c1: Scalar, a: T, c2: Scalar, b: T): T {
    util.assert(
        c1.size === 1,
        `Error in scaledArrayAdd: first argument must rank 0, but got ` +
            ` rank ${c1.rank}.`);
    util.assert(
        c2.size === 1,
        `Error in scaledArrayAdd: third argument must be rank 0, but got ` +
            `Tensor of rank ${c2.rank}.`);
    util.assertShapesMatch(a.shape, b.shape, 'Error in scaledArrayAdd: ');

    return tidy('scaledArrayAdd', () => {
      // TODO(nsthorat): Add an SGEMM kernel and then update this.
      return this.add(this.multiply(c1, a), this.multiply(c2, b)) as T;
    });
  }

  /** @deprecated Use dl.multiply(c, A) instead. */
  scalarTimesArray<T extends Tensor>(c: Scalar, a: T): T {
    util.assert(
        c.size === 1,
        `Error in arrayDividedByScalar: first argument must be rank 0, but ` +
            `got rank ${c.rank}.`);
    return this.multiply(c, a) as T;
  }
}

export type ScopeFn<T extends ScopeResult> =
    (keep: <T1 extends Tensor>(tensor: T1) => T1,
     track: <T2 extends Tensor>(tensor: T2) => T2) => T;
