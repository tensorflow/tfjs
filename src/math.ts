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
import {ArrayOps} from './ops/array_ops';
import {BatchNormOps} from './ops/batchnorm';
import {BinaryOps} from './ops/binary_ops';
import {CompareOps} from './ops/compare';
import {ConvOps} from './ops/conv';
import {ImageOps} from './ops/image_ops';
import {LogicalOps} from './ops/logical_ops';
import {LRNOps} from './ops/lrn';
import {LSTMOps} from './ops/lstm';
import {MatmulOps} from './ops/matmul';
import {NormOps} from './ops/norm';
import * as ops from './ops/ops';
import {PoolOps} from './ops/pool';
import {ReductionOps} from './ops/reduction_ops';
import {ReverseOps} from './ops/reverse';
import {SliceOps} from './ops/slice';
import {SoftmaxOps} from './ops/softmax';
import {TransposeOps} from './ops/transpose';
import {UnaryOps} from './ops/unary_ops';
import {ScopeResult} from './tape';
import {Scalar, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D} from './tensor';
import {Tracking} from './tracking';
import {Rank} from './types';
import * as util from './util';

const tidy = Tracking.tidy;
const keep = Tracking.keep;

export class NDArrayMath {
  // Ops.
  matMul = MatmulOps.matMul;
  vectorTimesMatrix = MatmulOps.vectorTimesMatrix;
  outerProduct = MatmulOps.outerProduct;
  matrixTimesVector = MatmulOps.matrixTimesVector;
  dotProduct = MatmulOps.dotProduct;

  slice = SliceOps.slice;
  slice1D = SliceOps.slice1d;
  slice2D = SliceOps.slice2d;
  slice3D = SliceOps.slice3d;
  slice4D = SliceOps.slice4d;

  reverse = ReverseOps.reverse;
  reverse1D = ReverseOps.reverse1d;
  reverse2D = ReverseOps.reverse2d;
  reverse3D = ReverseOps.reverse3d;
  reverse4D = ReverseOps.reverse4d;

  batchNormalization = BatchNormOps.batchNormalization;
  batchNormalization2D = BatchNormOps.batchNormalization2d;
  batchNormalization3D = BatchNormOps.batchNormalization3d;
  batchNormalization4D = BatchNormOps.batchNormalization4d;

  avgPool = PoolOps.avgPool;
  maxPool = PoolOps.maxPool;
  minPool = PoolOps.minPool;
  /** @deprecated */
  maxPoolBackprop = PoolOps.maxPoolBackprop;

  conv2dTranspose = ConvOps.conv2dTranspose;
  depthwiseConv2D = ConvOps.depthwiseConv2d;
  /** @deprecated */
  conv2dDerFilter = ConvOps.conv2dDerFilter;
  /** @deprecated */
  conv2dDerInput = ConvOps.conv2dDerInput;

  argMax = ReductionOps.argMax;
  argMin = ReductionOps.argMin;
  logSumExp = ReductionOps.logSumExp;
  max = ReductionOps.max;
  mean = ReductionOps.mean;
  min = ReductionOps.min;
  moments = ReductionOps.moments;
  sum = ReductionOps.sum;

  add = BinaryOps.add;
  addStrict = BinaryOps.addStrict;
  div = BinaryOps.div;
  divide = this.div;  // Alias.
  divStrict = BinaryOps.divStrict;
  divideStrict = this.divStrict;  // Alias.
  maximum = BinaryOps.maximum;
  maximumStrict = BinaryOps.maximumStrict;
  minimum = BinaryOps.minimum;
  minimumStrict = BinaryOps.minimumStrict;
  mul = BinaryOps.mul;
  multiply = this.mul;  // Alias.
  mulStrict = BinaryOps.mulStrict;
  multiplyStrict = this.mulStrict;  // Alias.
  pow = BinaryOps.pow;
  powStrict = BinaryOps.powStrict;
  sub = BinaryOps.sub;
  subtract = this.sub;  // Alias.
  subStrict = BinaryOps.subStrict;

  logicalNot = LogicalOps.logicalNot;
  logicalAnd = LogicalOps.logicalAnd;
  logicalOr = LogicalOps.logicalOr;
  logicalXor = LogicalOps.logicalXor;
  where = LogicalOps.where;

  transpose = TransposeOps.transpose;

  equal = CompareOps.equal;
  equalStrict = CompareOps.equalStrict;
  greater = CompareOps.greater;
  greaterStrict = CompareOps.greaterStrict;
  greaterEqual = CompareOps.greaterEqual;
  greaterEqualStrict = CompareOps.greaterEqualStrict;
  less = CompareOps.less;
  lessStrict = CompareOps.lessStrict;
  lessEqual = CompareOps.lessEqual;
  lessEqualStrict = CompareOps.lessEqualStrict;
  notEqual = CompareOps.notEqual;
  notEqualStrict = CompareOps.notEqualStrict;

  abs = UnaryOps.abs;
  acos = UnaryOps.acos;
  asin = UnaryOps.asin;
  atan = UnaryOps.atan;
  ceil = UnaryOps.ceil;
  clip = UnaryOps.clipByValue;
  cos = UnaryOps.cos;
  cosh = UnaryOps.cosh;
  elu = UnaryOps.elu;
  exp = UnaryOps.exp;
  floor = UnaryOps.floor;
  leakyRelu = UnaryOps.leakyRelu;
  log = UnaryOps.log;
  neg = UnaryOps.neg;
  prelu = UnaryOps.prelu;
  relu = UnaryOps.relu;
  selu = UnaryOps.selu;
  sigmoid = UnaryOps.sigmoid;
  sin = UnaryOps.sin;
  sinh = UnaryOps.sinh;
  sqrt = UnaryOps.sqrt;
  square = UnaryOps.square;
  step = UnaryOps.step;
  tan = UnaryOps.tan;
  tanh = UnaryOps.tanh;

  norm = NormOps.norm;

  basicLSTMCell = LSTMOps.basicLSTMCell;
  multiRNNCell = LSTMOps.multiRNNCell;

  softmax = SoftmaxOps.softmax;
  softmaxCrossEntropy = SoftmaxOps.softmaxCrossEntropy;

  cast = ArrayOps.cast;
  clone = ArrayOps.clone;
  gather = ArrayOps.gather;
  reshape = ArrayOps.reshape;
  tile = ArrayOps.tile;
  oneHot = ArrayOps.oneHot;
  multinomial = ArrayOps.multinomial;
  pad1D = ArrayOps.pad1d;
  pad2D = ArrayOps.pad2d;

  /** @deprecated Use dl.image.resizeBilinear() */
  resizeBilinear3D = ImageOps.resizeBilinear;

  localResponseNormalization3D = LRNOps.localResponseNormalization;
  localResponseNormalization4D = LRNOps.localResponseNormalization;

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
      values = ENV.engine.runKernel(backend => backend.topKValues(x, k));
      indices = ENV.engine.runKernel(backend => backend.topKIndices(x, k));
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
    return ops.transpose(x, perm);
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

  /** @deprecated */
  concat<T extends Tensor>(a: T, b: T, axis: number): T {
    return ops.concat([a, b], axis);
  }

  /** @deprecated */
  concat1D(a: Tensor1D, b: Tensor1D): Tensor1D {
    return ops.concat1d([a, b]);
  }

  /** @deprecated */
  concat2D(a: Tensor2D, b: Tensor2D, axis: number): Tensor2D {
    return ops.concat2d([a, b], axis);
  }

  /** @deprecated */
  concat3D(a: Tensor3D, b: Tensor3D, axis: number): Tensor3D {
    return ops.concat3d([a, b], axis);
  }

  /** @deprecated */
  concat4D(a: Tensor4D, b: Tensor4D, axis: number): Tensor4D {
    return ops.concat4d([a, b], axis);
  }

  /** @deprecated */
  conv1d<T extends Tensor2D|Tensor3D>(
      input: T, filter: Tensor3D, bias: Tensor1D|null, stride: number,
      pad: 'valid'|'same'|number, dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    if (bias != null) {
      util.assert(
          bias.rank === 1,
          `Error in conv1d: bias must be rank 1, but got rank ` +
              `${bias.rank}.`);
    }
    const res = ops.conv1d(input, filter, stride, pad, dimRoundingMode);
    return res.add(bias) as T;
  }

  /** @deprecated */
  conv2d<T extends Tensor3D|Tensor4D>(
      x: T, filter: Tensor4D, bias: Tensor1D|null,
      strides: [number, number]|number, pad: 'valid'|'same'|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    if (bias != null) {
      util.assert(
          bias.rank === 1,
          `Error in conv2d: bias must be rank 1, but got rank ` +
              `${bias.rank}.`);
    }
    const res = ops.conv2d(x, filter, strides, pad, dimRoundingMode);
    return res.add(bias) as T;
  }

  /** @deprecated */
  argMaxEquals(x1: Tensor, x2: Tensor): Scalar {
    util.assertShapesMatch(x1.shape, x2.shape, 'Error in argMaxEquals: ');
    return x1.argMax().equal(x2.argMax());
  }
}

export type ScopeFn<T extends ScopeResult> =
    (keep: <T1 extends Tensor>(tensor: T1) => T1,
     track: <T2 extends Tensor>(tensor: T2) => T2) => T;
