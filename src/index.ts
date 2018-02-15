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

import {BrowserUtil} from './browser_util';
import * as contrib from './contrib';
import * as xhr_dataset from './data/xhr-dataset';
import * as environment from './environment';
import {Environment} from './environment';
import * as gpgpu_util from './kernels/webgl/gpgpu_util';
import * as webgl_util from './kernels/webgl/webgl_util';
import * as conv_util from './ops/conv_util';
import * as test_util from './test_util';
import * as util from './util';
import {version} from './version';

export {CheckpointLoader} from './data/checkpoint_loader';
export {DataStats, InMemoryDataset} from './data/dataset';
// tslint:disable-next-line:max-line-length
export {InCPUMemoryShuffledInputProviderBuilder, InGPUMemoryShuffledInputProviderBuilder, InputProvider} from './data/input_provider';
export {XhrDataset, XhrDatasetConfig, XhrModelConfig} from './data/xhr-dataset';
export {ENV, Environment, Features} from './environment';
export {Graph, SymbolicTensor} from './graph/graph';
// tslint:disable-next-line:max-line-length
export {GraphRunner, GraphRunnerEventObserver, MetricReduction} from './graph/graph_runner';
// tslint:disable-next-line:max-line-length
export {ConstantInitializer, Initializer, OnesInitializer, RandomNormalInitializer, RandomTruncatedNormalInitializer, RandomUniformInitializer, TensorInitializer, VarianceScalingInitializer, ZerosInitializer} from './graph/initializers';
export {CostReduction, FeedEntry, Session} from './graph/session';
export {MathBackendCPU, NDArrayMathCPU} from './kernels/backend_cpu';
// tslint:disable-next-line:max-line-length
export {MathBackendWebGL, NDArrayMathGPU, WebGLTimingInfo} from './kernels/backend_webgl';
export {MatrixOrientation} from './kernels/types/matmul';
export {GPGPUContext} from './kernels/webgl/gpgpu_context';
export {NDArrayMath} from './math';
export {Model} from './model';
export {LSTMCell} from './ops/lstm';
export {AdadeltaOptimizer} from './optimizers/adadelta_optimizer';
export {AdagradOptimizer} from './optimizers/adagrad_optimizer';
export {AdamOptimizer} from './optimizers/adam_optimizer';
export {AdamaxOptimizer} from './optimizers/adamax_optimizer';
export {MomentumOptimizer} from './optimizers/momentum_optimizer';
export {Optimizer} from './optimizers/optimizer';
export {RMSPropOptimizer} from './optimizers/rmsprop_optimizer';
export {SGDOptimizer} from './optimizers/sgd_optimizer';
// tslint:disable-next-line:max-line-length
export {Array1D, Array2D, Array3D, Array4D, NDArray, Scalar, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D, variable, Variable} from './tensor';
export {Rank} from './types';
export {version};
// Second level exports.
export {
  contrib,
  conv_util,
  environment,
  gpgpu_util,
  test_util,
  util,
  webgl_util,
  xhr_dataset
};

export * from './ops/ops';
export * from './train';
export * from './globals';

export const setBackend = Environment.setBackend;
export const getBackend = Environment.getBackend;
export const memory = Environment.memory;

export const nextFrame = BrowserUtil.nextFrame;
