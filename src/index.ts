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

// backend_cpu.ts and backend_webgl.ts are standalone files and should be
// explicitly included here.
import './kernels/backend_webgl';
import './kernels/backend_cpu';

import {nextFrame} from './browser_util';
import * as environment from './environment';
import {deprecationWarn, disableDeprecationWarnings, enableDebugMode, enableProdMode, Environment} from './environment';
// Serialization.
import * as io from './io/io';
import * as math from './math';
import * as browser from './ops/browser';
import * as serialization from './serialization';
import {setOpHandler} from './tensor';
import * as tensor_util from './tensor_util';
import * as test_util from './test_util';
import * as util from './util';
import {version} from './version';
import * as webgl from './webgl';

export {InferenceModel, ModelPredictConfig} from './model_types';
// Optimizers.
export {AdadeltaOptimizer} from './optimizers/adadelta_optimizer';
export {AdagradOptimizer} from './optimizers/adagrad_optimizer';
export {AdamOptimizer} from './optimizers/adam_optimizer';
export {AdamaxOptimizer} from './optimizers/adamax_optimizer';
export {MomentumOptimizer} from './optimizers/momentum_optimizer';
export {Optimizer} from './optimizers/optimizer';
export {RMSPropOptimizer} from './optimizers/rmsprop_optimizer';
export {SGDOptimizer} from './optimizers/sgd_optimizer';
export {Scalar, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D, TensorBuffer, variable, Variable} from './tensor';
export {GradSaveFunc, NamedTensorMap, TensorContainer, TensorContainerArray, TensorContainerObject} from './tensor_types';
export {DataType, DataTypeMap, DataValues, Rank, ShapeMap, TensorLike} from './types';

export * from './ops/ops';
export {LSTMCellFunc} from './ops/lstm';
export {Reduction} from './ops/loss_ops';

export * from './train';
export * from './globals';

export {Features} from './environment_util';
export {TimingInfo} from './engine';
export {ENV, Environment} from './environment';

export const setBackend = Environment.setBackend;
export const getBackend = Environment.getBackend;
export const disposeVariables = Environment.disposeVariables;
export const memory = Environment.memory;
export {version as version_core};

// Top-level method exports.
export {
  nextFrame,
  enableProdMode,
  enableDebugMode,
  disableDeprecationWarnings,
  deprecationWarn
};

// Second level exports.
export {
  browser,
  environment,
  io,
  math,
  serialization,
  test_util,
  util,
  webgl,
  tensor_util
};

// Backend specific.
export {KernelBackend, BackendTimingInfo, DataMover, DataStorage} from './kernels/backend';

import * as ops from './ops/ops';
setOpHandler(ops);
