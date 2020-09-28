/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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

// base.ts is tfjs-core without auto registration of things like flags,
// gradients, chained ops or the opHandler. See base_side_effects.ts for parts
// tfjs core that are required side effects.

/**
 * @fileoverview
 * @suppress {partialAlias} Optimization disabled due to passing the module
 * object into a function below:
 *
 *   import * as ops from './ops/ops';
 *   setOpHandler(ops);
 */

// Serialization.
import * as io from './io/io';
import * as math from './math';
import * as browser from './ops/browser';
import * as gather_util from './ops/gather_nd_util';
import * as scatter_util from './ops/scatter_nd_util';
import * as slice_util from './ops/slice_util';
import * as serialization from './serialization';
import * as tensor_util from './tensor_util';
import * as test_util from './test_util';
import * as util from './util';
import {version} from './version';

export {InferenceModel, MetaGraph, MetaGraphInfo, ModelPredictConfig, ModelTensorInfo, SavedModelTensorInfo, SignatureDef, SignatureDefEntry, SignatureDefInfo} from './model_types';
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
export {BackendValues, DataType, DataTypeMap, DataValues, NumericDataType, PixelData, Rank, RecursiveArray, ScalarLike, ShapeMap, sumOutType, TensorLike, TypedArray, upcastType} from './types';

export * from './ops/ops';
export {Reduction} from './ops/loss_ops_utils';

export * from './train';
export * from './globals';
export * from './kernel_registry';
export {customGrad, grad, grads, valueAndGrad, valueAndGrads, variableGrads} from './gradients';

export {TimingInfo, MemoryInfo, ForwardFunc} from './engine';
export {Environment, env, ENV} from './environment';
export {Platform} from './platforms/platform';

export {version as version_core};

// Top-level method exports.
export {nextFrame} from './browser_util';

// Second level exports.
import * as backend_util from './backends/backend_util';
import * as device_util from './device_util';
export {
  browser,
  io,
  math,
  serialization,
  test_util,
  util,
  backend_util,
  tensor_util,
  slice_util,
  gather_util,
  scatter_util,
  device_util
};

import * as kernel_impls from './backends/kernel_impls';
export {kernel_impls};
// Backend specific.
export {KernelBackend, BackendTimingInfo, DataMover, DataStorage} from './backends/backend';

// Export all kernel names / info.
export * from './kernel_names';
