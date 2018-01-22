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

import * as xhr_dataset from './data/xhr-dataset';
import * as environment from './environment';
import * as gpgpu_util from './math/backends/webgl/gpgpu_util';
// tslint:disable-next-line:max-line-length
import * as render_ndarray_gpu_util from './math/backends/webgl/render_ndarray_gpu_util';
import * as webgl_util from './math/backends/webgl/webgl_util';
import * as batchnorm from './math/batchnorm';
import * as compare from './math/compare';
import * as concat from './math/concat';
import * as conv from './math/conv';
import * as conv_util from './math/conv_util';
// tslint:disable-next-line:max-line-length
import * as matmul from './math/matmul';
import * as minmax from './math/minmax';
import * as pool from './math/pool';
import * as reverse from './math/reverse';
import * as slice from './math/slice';
import {Ops as TranposeOps} from './math/transpose';
import * as test_util from './test_util';
import * as util from './util';
import {version} from './version';

export {CheckpointLoader} from './data/checkpoint_loader';
export {DataStats, InMemoryDataset} from './data/dataset';
// tslint:disable-next-line:max-line-length
export {InCPUMemoryShuffledInputProviderBuilder, InGPUMemoryShuffledInputProviderBuilder, InputProvider} from './data/input_provider';
export {XhrDataset, XhrDatasetConfig, XhrModelConfig} from './data/xhr-dataset';
export {ENV, Environment, Features} from './environment';
export {Graph, Tensor} from './graph/graph';
// tslint:disable-next-line:max-line-length
export {GraphRunner, GraphRunnerEventObserver, MetricReduction} from './graph/graph_runner';
export {AdadeltaOptimizer} from './graph/optimizers/adadelta_optimizer';
export {AdagradOptimizer} from './graph/optimizers/adagrad_optimizer';
export {AdamOptimizer} from './graph/optimizers/adam_optimizer';
export {AdamaxOptimizer} from './graph/optimizers/adamax_optimizer';
export {MomentumOptimizer} from './graph/optimizers/momentum_optimizer';
export {RMSPropOptimizer} from './graph/optimizers/rmsprop_optimizer';
export {CostReduction, FeedEntry, Session} from './graph/session';
// tslint:disable-next-line:max-line-length
export {ConstantInitializer, Initializer, NDArrayInitializer, OnesInitializer, RandomNormalInitializer, RandomTruncatedNormalInitializer, RandomUniformInitializer, VarianceScalingInitializer, ZerosInitializer} from './initializers';
export {MathBackendCPU, NDArrayMathCPU} from './math/backends/backend_cpu';
export {MathBackendWebGL, NDArrayMathGPU} from './math/backends/backend_webgl';
export {MatrixOrientation} from './math/backends/types/matmul';
export {GPGPUContext} from './math/backends/webgl/gpgpu_context';
export {LSTMCell, NDArrayMath} from './math/math';
// tslint:disable-next-line:max-line-length
export {Array1D, Array2D, Array3D, Array4D, NDArray, Scalar, variable, Variable} from './math/ndarray';
export {Optimizer} from './math/optimizers/optimizer';
export {SGDOptimizer} from './math/optimizers/sgd_optimizer';
export {Model} from './model';
export {version};
// Second level exports.
export {
  conv_util,
  environment,
  gpgpu_util,
  render_ndarray_gpu_util,
  test_util,
  util,
  webgl_util,
  xhr_dataset
};

// Math methods.
export const batchNormalization2D = batchnorm.Ops.batchNormalization2D;
export const batchNormalization3D = batchnorm.Ops.batchNormalization3D;
export const batchNormalization4D = batchnorm.Ops.batchNormalization4D;

export const concat1D = concat.Ops.concat1D;
export const concat2D = concat.Ops.concat2D;
export const concat3D = concat.Ops.concat3D;
export const concat4D = concat.Ops.concat4D;

export const conv1d = conv.Ops.conv1d;
export const conv2d = conv.Ops.conv2d;
export const conv2dTranspose = conv.Ops.conv2dTranspose;
export const depthwiseConv2D = conv.Ops.depthwiseConv2D;

export const dotProduct = matmul.Ops.dotProduct;
export const matMul = matmul.Ops.matMul;
export const matrixTimesVector = matmul.Ops.matrixTimesVector;
export const outerProduct = matmul.Ops.outerProduct;
export const vectorTimesMatrix = matmul.Ops.vectorTimesMatrix;

export const avgPool = pool.Ops.avgPool;
export const maxPool = pool.Ops.maxPool;
export const minPool = pool.Ops.minPool;

export const transpose = TranposeOps.transpose;

export const reverse1D = reverse.Ops.reverse1D;
export const reverse2D = reverse.Ops.reverse2D;
export const reverse3D = reverse.Ops.reverse3D;
export const reverse4D = reverse.Ops.reverse4D;

export const slice1D = slice.Ops.slice1D;
export const slice2D = slice.Ops.slice2D;
export const slice3D = slice.Ops.slice3D;
export const slice4D = slice.Ops.slice4D;

export const min = minmax.Ops.min;
export const max = minmax.Ops.max;
export const minimum = minmax.Ops.minimum;
export const maximum = minmax.Ops.maximum;
export const argMax = minmax.Ops.argMax;
export const argMin = minmax.Ops.argMin;
export const argMaxEquals = minmax.Ops.argMaxEquals;

export const equal = compare.Ops.equal;
export const equalStrict = compare.Ops.equalStrict;
export const greater = compare.Ops.greater;
export const greaterEqual = compare.Ops.greaterEqual;
export const less = compare.Ops.less;
export const lessEqual = compare.Ops.lessEqual;
export const notEqual = compare.Ops.notEqual;
export const notEqualStrict = compare.Ops.notEqualStrict;
