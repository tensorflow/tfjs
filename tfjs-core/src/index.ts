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

/**
 * @fileoverview
 * @suppress {partialAlias} Optimization disabled due to passing the module
 * object into a function below:
 *
 *   import * as ops from './ops/ops';
 *   setOpHandler(ops);
 */

// Engine is the global singleton that needs to be initialized before the rest
// of the app.
import './engine';
// Register backend-agnostic flags.
import './flags';

// backend_cpu.ts and backend_webgl.ts are standalone files and should be
// explicitly included here.
import './backends/webgl/backend_webgl';
import './backends/cpu/backend_cpu';
// Import all kernels from cpu.
import './backends/cpu/all_kernels';
// Import all kernels from webgl.
import './backends/webgl/all_kernels';

import './platforms/platform_browser';
import './platforms/platform_node';

import * as backend_util from './backends/backend_util';
// Serialization.
import * as io from './io/io';
import * as math from './math';
import * as browser from './ops/browser';
import * as slice_util from './ops/slice_util';
import * as serialization from './serialization';
import {setOpHandler} from './tensor';
import * as tensor_util from './tensor_util';
import * as test_util from './test_util';
import * as util from './util';
import {version} from './version';
import * as webgl from './webgl';

export {InferenceModel, MetaGraph, MetaGraphInfo, ModelPredictConfig, ModelTensorInfo, SavedModelTensorInfo, SignatureDef, SignatureDefInfo} from './model_types';
// Optimizers.
export {AdadeltaOptimizer} from './optimizers/adadelta_optimizer';
export {AdagradOptimizer} from './optimizers/adagrad_optimizer';
export {AdamOptimizer} from './optimizers/adam_optimizer';
export {AdamaxOptimizer} from './optimizers/adamax_optimizer';
export {MomentumOptimizer} from './optimizers/momentum_optimizer';
export {Optimizer} from './optimizers/optimizer';
export {RMSPropOptimizer} from './optimizers/rmsprop_optimizer';
export {SGDOptimizer} from './optimizers/sgd_optimizer';
export {Scalar, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D, Tensor5D, TensorBuffer, Variable} from './tensor';
export {GradSaveFunc, NamedTensorMap, TensorContainer, TensorContainerArray, TensorContainerObject} from './tensor_types';
export {DataType, DataTypeMap, DataValues, Rank, RecursiveArray, ShapeMap, TensorLike} from './types';

export * from './ops/ops';
export {LSTMCellFunc} from './ops/lstm';
export {Reduction} from './ops/loss_ops';

export * from './train';
export * from './globals';
export * from './kernel_registry';
export {customGrad, grad, grads, valueAndGrad, valueAndGrads, variableGrads} from './gradients';

export {TimingInfo, MemoryInfo} from './engine';
export {Environment, env, ENV} from './environment';
export {Platform} from './platforms/platform';

export {version as version_core};

// Top-level method exports.
export {nextFrame} from './browser_util';

// Second level exports.
export {
  browser,
  io,
  math,
  serialization,
  test_util,
  util,
  backend_util,
  webgl,
  tensor_util,
  slice_util
};

// Backend specific.
export {KernelBackend, BackendTimingInfo, DataMover, DataStorage} from './backends/backend';

import * as ops from './ops/ops';
setOpHandler(ops);
