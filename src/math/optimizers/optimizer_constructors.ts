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
import {doc} from '../decorators';

import {MomentumOptimizer} from './momentum_optimizer';
import {SGDOptimizer} from './sgd_optimizer';

export class OptimizerConstructors {
  /**
   * Constructs a `dl.train.SGDOptimizer` that uses stochastic gradient descent.
   *
   * @param learningRate The learning rate to use for the SGD algorithm.
   */
  @doc({heading: 'Training', subheading: 'Optimizers', namespace: 'train'})
  static sgd(learningRate: number): SGDOptimizer {
    return new SGDOptimizer(learningRate);
  }

  /**
   * Constructs a `dl.train.MomentumOptimizer` that uses momentum gradient
   * descent.
   *
   * @param learningRate The learning rate to use for the momentum gradient
   * descent algorithm.
   * @param momentum The momentum to use for the momentum gradient descent
   * algorithm.
   */
  @doc({heading: 'Training', subheading: 'Optimizers', namespace: 'train'})
  static momentum(learningRate: number, momentum: number): MomentumOptimizer {
    return new MomentumOptimizer(learningRate, momentum);
  }
}
