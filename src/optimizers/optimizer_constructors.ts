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

import {doc} from '../doc';
import {AdadeltaOptimizer} from './adadelta_optimizer';
import {AdagradOptimizer} from './adagrad_optimizer';
import {MomentumOptimizer} from './momentum_optimizer';
import {RMSPropOptimizer} from './rmsprop_optimizer';
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

  /**
   * Constructs a `dl.train.RMSPropOptimizer` that uses RMSProp gradient
   * descent. This implementation uses plain momentum and is not centered
   * version of RMSProp.
   *
   * See
   * http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
   *
   * @param learningRate The learning rate to use for the RMSProp gradient
   * descent algorithm.
   * @param decay The discounting factor for the history/coming gradient
   * @param momentum The momentum to use for the RMSProp gradient descent
   * algorithm.
   * @param epsilon Small value to avoid zero denominator.
   */
  @doc({heading: 'Training', subheading: 'Optimizers', namespace: 'train'})
  static rmsprop(
      learningRate: number, decay = .9, momentum = 0.0, epsilon = 1e-8):
      RMSPropOptimizer {
    return new RMSPropOptimizer(
        learningRate, decay, momentum,
        undefined /** @deprecated specifiedVariableList */, epsilon);
  }

  /**
   * Constructs a `dl.train.AdadeltaOptimizer` that uses the Adadelta algorithm.
   * See https://arxiv.org/abs/1212.5701
   *
   * @param learningRate
   * @param rho
   * @param epsilon
   */
  @doc({heading: 'Training', subheading: 'Optimizers', namespace: 'train'})
  static adadelta(learningRate = .001, rho = .95, epsilon = 1e-8):
      AdadeltaOptimizer {
    return new AdadeltaOptimizer(
        learningRate, rho, undefined /** @deprecated specifiedVariableList */,
        epsilon);
  }

  /**
   * Constructs a `dl.train.AdagradOptimizer` that uses the Adagrad algorithm.
   * See http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf or
   * http://ruder.io/optimizing-gradient-descent/index.html#adagrad
   *
   * @param learningRate
   * @param initialAccumulatorValue
   */
  @doc({heading: 'Training', subheading: 'Optimizers', namespace: 'train'})
  static adagrad(learningRate: number, initialAccumulatorValue = 0.1):
      AdagradOptimizer {
    return new AdagradOptimizer(
        learningRate, undefined /** @deprecated specifiedVariableList */,
        initialAccumulatorValue);
  }
}
