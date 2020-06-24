/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

// So typings can propagate.
import {AdadeltaOptimizer} from './optimizers/adadelta_optimizer';
import {AdagradOptimizer} from './optimizers/adagrad_optimizer';
import {AdamOptimizer} from './optimizers/adam_optimizer';
import {AdamaxOptimizer} from './optimizers/adamax_optimizer';
import {MomentumOptimizer} from './optimizers/momentum_optimizer';
import {OptimizerConstructors} from './optimizers/optimizer_constructors';
import {RMSPropOptimizer} from './optimizers/rmsprop_optimizer';
import {SGDOptimizer} from './optimizers/sgd_optimizer';

// tslint:disable-next-line:no-unused-expression
[MomentumOptimizer, SGDOptimizer, AdadeltaOptimizer, AdagradOptimizer,
 RMSPropOptimizer, AdamaxOptimizer, AdamOptimizer];

export const train = {
  sgd: OptimizerConstructors.sgd,
  momentum: OptimizerConstructors.momentum,
  adadelta: OptimizerConstructors.adadelta,
  adagrad: OptimizerConstructors.adagrad,
  rmsprop: OptimizerConstructors.rmsprop,
  adamax: OptimizerConstructors.adamax,
  adam: OptimizerConstructors.adam
};
