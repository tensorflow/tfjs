/**
 * @license
 * Copyright 2022 Google LLC.
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

import {AdadeltaOptimizer} from './adadelta_optimizer';
import {AdagradOptimizer} from './adagrad_optimizer';
import {AdamOptimizer} from './adam_optimizer';
import {AdamaxOptimizer} from './adamax_optimizer';
import {MomentumOptimizer} from './momentum_optimizer';
import {RMSPropOptimizer} from './rmsprop_optimizer';
import {SGDOptimizer} from './sgd_optimizer';
import {registerClass} from '../serialization';

const OPTIMIZERS = [
  AdadeltaOptimizer,
  AdagradOptimizer,
  AdamOptimizer,
  AdamaxOptimizer,
  MomentumOptimizer,
  RMSPropOptimizer,
  SGDOptimizer,
];

export function registerOptimizers() {
  for (const optimizer of OPTIMIZERS) {
    registerClass(optimizer);
  }
}
