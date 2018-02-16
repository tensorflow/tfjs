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
import {AdamOptimizer} from './adam_optimizer';
import {AdamaxOptimizer} from './adamax_optimizer';
import {MomentumOptimizer} from './momentum_optimizer';
import {RMSPropOptimizer} from './rmsprop_optimizer';
import {SGDOptimizer} from './sgd_optimizer';

export class OptimizerConstructors {
  /**
   * Constructs a `SGDOptimizer` that uses stochastic gradient descent.
   *
   * ```js
   * // Fit a quadratic function by learning the coefficients a, b, c.
   * const xs = dl.tensor1d([0, 1, 2, 3]);
   * const ys = dl.tensor1d([1.1, 5.9, 16.8, 33.9]);
   *
   * const a = dl.scalar(Math.random()).variable();
   * const b = dl.scalar(Math.random()).variable();
   * const c = dl.scalar(Math.random()).variable();
   *
   * // y = a * x^2 + b * x + c.
   * const f = x => a.mul(x.square()).add(b.mul(x)).add(c);
   * const loss = (pred, label) => pred.sub(label).square().mean();
   *
   * const learningRate = 0.01;
   * const optimizer = dl.train.sgd(learningRate);
   *
   * // Train the model.
   * for (let i = 0; i < 10; i++) {
   *   optimizer.minimize(() => loss(f(xs), ys));
   * }
   *
   * // Make predictions.
   * console.log(
   *     `a: ${a.dataSync()}, b: ${b.dataSync()}, c: ${c.dataSync()}`);
   * const preds = f(xs).dataSync();
   * preds.forEach((pred, i) => {
   *   console.log(`x: ${i}, pred: ${pred}`);
   * });
   * ```
   *
   * @param learningRate The learning rate to use for the SGD algorithm.
   */
  @doc({heading: 'Training', subheading: 'Optimizers', namespace: 'train'})
  static sgd(learningRate: number): SGDOptimizer {
    return new SGDOptimizer(learningRate);
  }

  /**
   * Constructs a `MomentumOptimizer` that uses momentum gradient
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
   * Constructs a `RMSPropOptimizer` that uses RMSProp gradient
   * descent. This implementation uses plain momentum and is not centered
   * version of RMSProp.
   *
   * See
   * [http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf](
   * http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
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
   * Constructs a `AdamOptimizer` that uses the Adam algorithm.
   * See [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
   *
   * @param learningRate
   * @param beta1
   * @param beta2
   */
  @doc({heading: 'Training', subheading: 'Optimizers', namespace: 'train'})
  static adam(learningRate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
      AdamOptimizer {
    return new AdamOptimizer(
        learningRate, beta1, beta2, epsilon,
        undefined /** @deprecated specifiedVariableList */);
  }

  /**
   * Constructs a `AdadeltaOptimizer` that uses the Adadelta algorithm.
   * See [https://arxiv.org/abs/1212.5701](https://arxiv.org/abs/1212.5701)
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
   * Constructs a `AdamaxOptimizer` that uses the Adam algorithm.
   * See [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
   *
   * @param learningRate
   * @param beta1
   * @param beta2
   * @param decay
   */
  @doc({heading: 'Training', subheading: 'Optimizers', namespace: 'train'})
  static adamax(
      learningRate = 0.002, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8,
      decay = 0.0): AdamaxOptimizer {
    return new AdamaxOptimizer(
        learningRate, beta1, beta2, epsilon, decay,
        undefined /** @deprecated specifiedVariableList */);
  }

  /**
   * Constructs a `dl.train.AdagradOptimizer` that uses the Adagrad algorithm.
   * Constructs a `AdagradOptimizer` that uses the Adagrad algorithm.
   * See
   * [http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf](
   * http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
   * or
   * [http://ruder.io/optimizing-gradient-descent/index.html#adagrad](
   * http://ruder.io/optimizing-gradient-descent/index.html#adagrad)
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
