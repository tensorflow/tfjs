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

import * as array_ops from './array_ops';
import * as batchnorm_ops from './batchnorm';
import * as binary_ops from './binary_ops';
import * as compare_ops from './compare';
import * as concat_ops from './concat';
import * as conv_ops from './conv';
import * as image_ops from './image_ops';
import * as logical_ops from './logical_ops';
import * as lstm_ops from './lstm';
import * as matmul_ops from './matmul';
import * as norm_ops from './norm';
import * as pool_ops from './pool';
import * as reduction_ops from './reduction_ops';
import * as reverse_ops from './reverse';
import * as slice_ops from './slice';
import * as softmax_ops from './softmax';
import * as transpose_ops from './transpose';
import * as unary_ops from './unary_ops';

export const batchNormalization = batchnorm_ops.Ops.batchNormalization;
export const batchNormalization2D = batchnorm_ops.Ops.batchNormalization2D;
export const batchNormalization3D = batchnorm_ops.Ops.batchNormalization3D;
export const batchNormalization4D = batchnorm_ops.Ops.batchNormalization4D;

export const concat = concat_ops.Ops.concat;
export const concat1D = concat_ops.Ops.concat1D;
export const concat2D = concat_ops.Ops.concat2D;
export const concat3D = concat_ops.Ops.concat3D;
export const concat4D = concat_ops.Ops.concat4D;

export const conv1d = conv_ops.Ops.conv1d;
export const conv2d = conv_ops.Ops.conv2d;
export const conv2dTranspose = conv_ops.Ops.conv2dTranspose;
export const depthwiseConv2D = conv_ops.Ops.depthwiseConv2D;

export const dotProduct = matmul_ops.Ops.dotProduct;
export const matMul = matmul_ops.Ops.matMul;
export const matrixTimesVector = matmul_ops.Ops.matrixTimesVector;
export const outerProduct = matmul_ops.Ops.outerProduct;
export const vectorTimesMatrix = matmul_ops.Ops.vectorTimesMatrix;

export const avgPool = pool_ops.Ops.avgPool;
export const maxPool = pool_ops.Ops.maxPool;
export const minPool = pool_ops.Ops.minPool;

export const transpose = transpose_ops.Ops.transpose;

export const reverse = reverse_ops.Ops.reverse;
export const reverse1D = reverse_ops.Ops.reverse1D;
export const reverse2D = reverse_ops.Ops.reverse2D;
export const reverse3D = reverse_ops.Ops.reverse3D;
export const reverse4D = reverse_ops.Ops.reverse4D;

export const slice = slice_ops.Ops.slice;
export const slice1D = slice_ops.Ops.slice1D;
export const slice2D = slice_ops.Ops.slice2D;
export const slice3D = slice_ops.Ops.slice3D;
export const slice4D = slice_ops.Ops.slice4D;

export const argMax = reduction_ops.Ops.argMax;
export const argMaxEquals = reduction_ops.Ops.argMaxEquals;
export const argMin = reduction_ops.Ops.argMin;
export const logSumExp = reduction_ops.Ops.logSumExp;
export const max = reduction_ops.Ops.max;
export const mean = reduction_ops.Ops.mean;
export const min = reduction_ops.Ops.min;
export const moments = reduction_ops.Ops.moments;
export const sum = reduction_ops.Ops.sum;

export const equal = compare_ops.Ops.equal;
export const equalStrict = compare_ops.Ops.equalStrict;
export const greater = compare_ops.Ops.greater;
export const greaterStrict = compare_ops.Ops.greaterStrict;
export const greaterEqual = compare_ops.Ops.greaterEqual;
export const greaterEqualStrict = compare_ops.Ops.greaterEqualStrict;
export const less = compare_ops.Ops.less;
export const lessStrict = compare_ops.Ops.lessStrict;
export const lessEqual = compare_ops.Ops.lessEqual;
export const lessEqualStrict = compare_ops.Ops.lessEqualStrict;
export const notEqual = compare_ops.Ops.notEqual;
export const notEqualStrict = compare_ops.Ops.notEqualStrict;

export const logicalNot = logical_ops.Ops.logicalNot;
export const logicalAnd = logical_ops.Ops.logicalAnd;
export const logicalOr = logical_ops.Ops.logicalOr;
export const logicalXor = logical_ops.Ops.logicalXor;
export const where = logical_ops.Ops.where;

export const abs = unary_ops.Ops.abs;
export const acos = unary_ops.Ops.acos;
export const asin = unary_ops.Ops.asin;
export const atan = unary_ops.Ops.atan;
export const ceil = unary_ops.Ops.ceil;
export const clip = unary_ops.Ops.clip;
export const cos = unary_ops.Ops.cos;
export const cosh = unary_ops.Ops.cosh;
export const elu = unary_ops.Ops.elu;
export const exp = unary_ops.Ops.exp;
export const floor = unary_ops.Ops.floor;
export const leakyRelu = unary_ops.Ops.leakyRelu;
export const log = unary_ops.Ops.log;
export const neg = unary_ops.Ops.neg;
export const prelu = unary_ops.Ops.prelu;
export const relu = unary_ops.Ops.relu;
export const selu = unary_ops.Ops.selu;
export const sigmoid = unary_ops.Ops.sigmoid;
export const sin = unary_ops.Ops.sin;
export const sinh = unary_ops.Ops.sinh;
export const sqrt = unary_ops.Ops.sqrt;
export const square = unary_ops.Ops.square;
export const step = unary_ops.Ops.step;
export const tan = unary_ops.Ops.tan;
export const tanh = unary_ops.Ops.tanh;

export const add = binary_ops.Ops.add;
export const addStrict = binary_ops.Ops.addStrict;
export const div = binary_ops.Ops.div;
export const divStrict = binary_ops.Ops.divStrict;
export const maximum = binary_ops.Ops.maximum;
export const maximumStrict = binary_ops.Ops.maximumStrict;
export const minimum = binary_ops.Ops.minimum;
export const minimumStrict = binary_ops.Ops.minimumStrict;
export const mul = binary_ops.Ops.mul;
export const mulStrict = binary_ops.Ops.mulStrict;
export const pow = binary_ops.Ops.pow;
export const powStrict = binary_ops.Ops.powStrict;
export const sub = binary_ops.Ops.sub;
export const subStrict = binary_ops.Ops.subStrict;

export const norm = norm_ops.Ops.norm;

export const cast = array_ops.Ops.cast;
export const clone = array_ops.Ops.clone;
export const fromPixels = array_ops.Ops.fromPixels;
export const ones = array_ops.Ops.ones;
export const onesLike = array_ops.Ops.onesLike;
export const zeros = array_ops.Ops.zeros;
export const zerosLike = array_ops.Ops.zerosLike;
export const rand = array_ops.Ops.rand;
export const randomNormal = array_ops.Ops.randomNormal;
export const truncatedNormal = array_ops.Ops.truncatedNormal;
export const randomUniform = array_ops.Ops.randomUniform;
export const reshape = array_ops.Ops.reshape;
export const tile = array_ops.Ops.tile;
export const gather = array_ops.Ops.gather;
export const multinomial = array_ops.Ops.multinomial;
export const oneHot = array_ops.Ops.oneHot;
export const pad1D = array_ops.Ops.pad1D;
export const pad2D = array_ops.Ops.pad2D;
export const range = array_ops.Ops.range;
export const buffer = array_ops.Ops.buffer;
export const fill = array_ops.Ops.fill;

export const basicLSTMCell = lstm_ops.Ops.basicLSTMCell;
export const multiRNNCell = lstm_ops.Ops.multiRNNCell;

export const softmax = softmax_ops.Ops.softmax;

// So typings can propagate.
import {Tensor, Tensor3D, Tensor4D} from './tensor';
import {Rank} from './types';
// tslint:disable-next-line:no-unused-expression
[Tensor, Rank, Tensor3D, Tensor4D];

export const losses = {
  softmaxCrossEntropy: softmax_ops.Ops.softmaxCrossEntropy
};

export const image = {
  resizeBilinear: image_ops.Ops.resizeBilinear
};
