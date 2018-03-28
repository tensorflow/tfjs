/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * Optimizers.
 *
 * These optimizers are wrappers around the core optimizers, with
 * layers-specific support for constraint-compatible variable and
 * (de)serialization.
 *
 * TODO(cais, nsthorat): maybe these additional features should be pushed down
 *   to core optimizers, so there is no need for these wrappers.
 */

// tslint:disable:max-line-length
import {AdagradOptimizer, AdamOptimizer, Optimizer as CoreOptimizer, RMSPropOptimizer, Scalar, SGDOptimizer, train, Variable} from '@tensorflow/tfjs-core';

// tslint:enable:max-line-length

import * as K from './backend/tfjs_backend';
import {NotImplementedError, ValueError} from './errors';
import {ConfigDict, LayerVariable} from './types';
import {ClassNameMap, Constructor} from './utils/generic_utils';

/**
 * Base configuration for optimizers.
 */
export interface OptimizerConfig {
  /**
   * float >= 0. Gradients will be clipped when their L2 norm exceeds this
   * value.
   */
  clipnorm?: number;
  /**
   * float >= 0. Gradients will be clipped when their absolute value exceeds
   * this value.
   */
  clipvalue?: number;
}

/**
 * Abstract optimizer base class.
 *
 * Note this is the parent class of all optimizers, not an actual
 * optimizer that can be used for training models.
 */
export abstract class LayersOptimizer {
  clipnorm: number;
  clipvalue: number;

  protected optimizer: CoreOptimizer;
  private readonly createdFromCoreOptimizer: boolean;

  constructor(config: OptimizerConfig|CoreOptimizer) {
    if (config instanceof CoreOptimizer) {
      this.createdFromCoreOptimizer = true;
      this.constructFromCoreOptimizer(config);
    } else {
      this.createdFromCoreOptimizer = false;
      this.clipnorm = config.clipnorm;
      this.clipvalue = config.clipvalue;
      this.constructFromConfig(config);
    }
  }

  protected abstract constructFromCoreOptimizer(optimizer: CoreOptimizer): void;

  protected abstract constructFromConfig(config: OptimizerConfig): void;

  // TODO(michaelterry): Add get/setWeights, if needed.
  getConfig(): ConfigDict {
    if (this.createdFromCoreOptimizer) {
      throw new NotImplementedError(
          'getConfig() for a LayersOptimizer constructed from a core ' +
          'Optimizer is not supported yet.');
      // TODO(cais): Once the hyperparameters of CoreOptimizers are public or
      //   have getters available, use them to implement the logic here.
    }

    const config: ConfigDict = {};
    if (this.clipnorm != null) {
      config['clipnorm'] = this.clipnorm;
    }
    if (this.clipvalue != null) {
      config['clipvalue'] = this.clipvalue;
    }
    return config;
  }

  /**
   * Calculates the gradients based on the loss and params (variables) passed
   * in and updates the params.
   *
   * Porting note: This is getUpdates() in PyKeras. However
   * TensorFlow.js Layers assumes an eager-execution backend,
   * making it unnecessary to create update ops.
   *
   * @param lossFn A function to calculate the loss.
   * @param params The variables to optimize/update.
   *
   * @return Loss value as a `Scalar`.
   */
  updateVariables(lossFn: () => Scalar, params: LayerVariable[]): Scalar {
    const variables = params.map(param => param.read() as Variable);
    return this.optimizer.minimize(lossFn, true, variables);
  }

  static fromConfig<T>(cls: Constructor<T>, config: ConfigDict): T {
    return new cls(config);
  }
}

export interface SGDConfig extends OptimizerConfig {
  /** float >= 0. Learning rate. Defaults to 0.01 if not specified. */
  lr?: number;
  /**
   * float >=0. Parameter that accelerates SGD in the relevant direction and
   * dampens oscillations. Defaults to 0.0 if not specified.
   */
  momentum?: number;
  /**
   * float >= 0. Learning rate decay over each update. Defaults to 0.0 if not
   * specified.
   */
  decay?: number;
  /**
   * Whether to apply Nesterov momentum. Defaults to `false` if not specified.
   */
  nesterov?: boolean;
}

/**
 * Stochastic gradient descent optimizer.
 *
 * Includes support for momentum, learning rate decay, and Nesterov momentum.
 */
export class SGD extends LayersOptimizer {
  lr: number;
  momentum: number;
  decay: number;
  nesterov: boolean;

  constructor(config: SGDConfig|SGDOptimizer) {
    super(config);
  }

  /**
   * Tracks momentum over time. These are paired with specific sets of
   * variables the first time updateVariables() is called with a given set of
   * variables.
   */
  constructFromConfig(config: SGDConfig) {
    this.lr = (config.lr == null) ? 0.01 : config.lr;
    if (this.lr < 0) {
      throw new ValueError(
          `Invalid lr (${this.lr}). Must be >= 0 or undefined.`);
    }

    this.momentum = (config.momentum == null) ? 0.0 : config.momentum;
    if (this.momentum < 0) {
      throw new ValueError(
          `Invalid momentum (${this.momentum}). Must be >= 0 or undefined.`);
    }
    if (this.momentum !== 0) {
      throw new NotImplementedError('SGD momentum is not implemented yet.');
    }

    this.decay = (config.decay == null) ? 0.0 : config.decay;
    if (this.decay < 0) {
      throw new ValueError(
          `Invalid decay (${this.decay}). Must be >= 0 or undefined.`);
    }
    if (this.decay !== 0) {
      throw new NotImplementedError('SGD decay is not implemented yet');
    }

    this.nesterov = (config.nesterov == null) ? false : config.nesterov;
    if (this.nesterov !== false) {
      throw new NotImplementedError('SGD nesterov is not implemented yet');
    }

    this.optimizer = train.sgd(this.lr);
  }

  protected constructFromCoreOptimizer(optimizer: CoreOptimizer) {
    if (!(optimizer instanceof SGDOptimizer)) {
      throw new ValueError(
          'Cannot construct SGD from a non-SGD core optimizer');
    }
    this.optimizer = optimizer;
  }

  getConfig(): ConfigDict {
    const config: ConfigDict = {
      lr: this.lr,
      momentum: this.momentum,
      decay: this.decay,
      nestorv: this.nesterov,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
ClassNameMap.register('SGD', SGD);

export interface AdamConfig extends OptimizerConfig {
  /** float >= 0. Learning rate. Defaults to 0.001 if not specified. */
  lr?: number;
  /**
   * float, 0 < beta_1 < 1. Generally close to 1. Default: 0.9.
   */
  beta_1?: number;
  /**
   * float, 0 < beta_2 < 1. Generally close to 1. Default: 0.999.
   */
  beta_2?: number;
  /**
   * float >= 0. Fuzz factor. If `null` or `undefined`, defaults to
   *   `K.epsilon()`.
   */
  epsilon?: number;
  /**
   * float >= 0. Learning rate decay over each update. Default: 0.
   */
  decay?: number;
  /**
   * Whether to apply AMSGrad variant of this algorithm from the paper
   * "On the Convergence of Adam and Beyond".
   *
   * References:
   *   - [On the Convergence of Adam and
   * Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
   */
  amsgrad?: boolean;
}

/**
 * Adam optimizer.
 *
 * Default parameters follow those provided in the original paper.
 *
 * References
 *   - [Adam - A Method for Stochastic
 * Optimization](http://arxiv.org/abs/1412.6980v8)
 *   - [On the Convergence of Adam and
 * Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
 */
export class Adam extends LayersOptimizer {
  lr: number;
  beta1: number;
  beta2: number;
  decay: number;
  epsilon: number;
  amsgrad: boolean;

  constructor(config: AdamConfig|AdamOptimizer) {
    super(config);
  }

  constructFromConfig(config: AdamConfig) {
    this.lr = config.lr == null ? 0.001 : config.lr;

    this.beta1 = config.beta_1 == null ? 0.9 : config.beta_1;
    this.beta2 = config.beta_2 == null ? 0.999 : config.beta_2;
    this.epsilon = config.epsilon == null ? K.epsilon() : config.epsilon;

    this.decay = config.decay == null ? 0 : config.decay;
    if (this.decay !== 0.0) {
      throw new NotImplementedError('Adam decay is not implemented yet');
    }

    this.amsgrad = config.amsgrad == null ? false : config.amsgrad;
    if (this.amsgrad !== false) {
      throw new NotImplementedError('Adam amsgrad is not implemented yet');
    }

    this.optimizer = train.adam(this.lr, this.beta1, this.beta2, this.epsilon);
  }

  protected constructFromCoreOptimizer(optimizer: CoreOptimizer) {
    if (!(optimizer instanceof AdamOptimizer)) {
      throw new ValueError(
          'Cannot construct Adam from a non-Adam core optimizer');
    }
    this.optimizer = optimizer;
  }

  getConfig(): ConfigDict {
    const config: ConfigDict = {
      lr: this.lr,
      beta1: this.beta1,
      beta2: this.beta2,
      decay: this.decay,
      epsilon: this.epsilon,
      amsgrad: this.amsgrad
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
ClassNameMap.register('Adam', Adam);

export interface RMSPropConfig extends OptimizerConfig {
  /** float >= 0. Learning rate. Defaults to 0.001 if not specified. */
  lr?: number;

  /** float >=0.  Defaults to 0.9 */
  rho?: number;

  /** Float >= 0.  Fuzz factor, defaults to K.epsilon(). */
  epsilon?: number;

  /** Float. >=0.  Learning rate decay after each update.  Default 0. */
  decay?: number;
}

/**
 * RMSProp optimizer.
 *
 * It is recommended to leave the parameters of this optimizer at their
 * default values (except the learning rate, which can be freely tuned).
 *
 * This optimizer is a usually a good choice for recurrent neural networks.
 *
 * References
 *   - [rmsprop: Divide the gradient by a running average of its
 *      recent magnitude]
 * (http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
 */
export class RMSProp extends LayersOptimizer {
  lr: number;
  rho: number;
  decay: number;
  iterations: number;
  epsilon: number;

  constructor(config: RMSPropConfig|RMSPropOptimizer) {
    super(config);
  }

  constructFromConfig(config: RMSPropConfig) {
    this.lr = config.lr == null ? 0.001 : config.lr;
    this.rho = config.rho == null ? 0.9 : config.rho;
    this.epsilon = config.epsilon == null ? K.epsilon() : config.epsilon;

    if (config.decay != null) {
      throw new NotImplementedError('RMSProp decay is not implemented yet');
    }

    this.optimizer = train.rmsprop(this.lr, this.rho, null, this.epsilon);
  }

  protected constructFromCoreOptimizer(optimizer: CoreOptimizer) {
    if (!(optimizer instanceof RMSPropOptimizer)) {
      throw new ValueError(
          'Cannot construct RMSProp from a non-RMSProp core optimizer');
    }
    this.optimizer = optimizer;
  }

  getConfig(): ConfigDict {
    const config: ConfigDict = {
      lr: this.lr,
      rho: this.rho,
      decay: this.decay,
      epsilon: this.epsilon,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
ClassNameMap.register('RMSProp', RMSProp);

export interface AdagradConfig extends OptimizerConfig {
  /** Learning rate. Defaults to 0.01. */
  lr?: number;


  /** Float >= 0.  Fuzz factor, defaults to K.epsilon(). */
  epsilon?: number;

  /** Float >= 0. Learning rate decay over each update. Default: 0. */
  decay?: number;
}

export class Adagrad extends LayersOptimizer {
  private lr: number;
  private epsilon: number;
  private decay: number;

  constructor(config: AdagradConfig|AdagradOptimizer) {
    super(config);
  }

  constructFromConfig(config: AdagradConfig) {
    this.lr = config.lr == null ? 0.01 : config.lr;
    this.epsilon = config.epsilon == null ? K.epsilon() : config.epsilon;

    this.decay = config.decay == null ? 0 : config.decay;
    if (this.decay !== 0) {
      throw new NotImplementedError('Adagrad decay is not implemented yet');
    }

    this.optimizer = train.adagrad(this.lr);
  }

  constructFromCoreOptimizer(optimizer: CoreOptimizer) {
    if (!(optimizer instanceof AdagradOptimizer)) {
      throw new ValueError(
          'Cannot construct Adagrad from a non-Adagrad core optimizer');
    }
    this.optimizer = optimizer;
  }

  getConfig(): ConfigDict {
    const config: ConfigDict = {
      lr: this.lr,
      decay: this.decay,
      epsilon: this.epsilon,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
ClassNameMap.register('Adagrad', Adagrad);

export const adagrad = Adagrad;
export const adam = Adam;
export const rmsprop = RMSProp;
export const sgd = SGD;

// Add (de)serialize()

// Porting note: This diverges from the PyKeras implementation and may need to
// change based on (de)serialization requirements.
export function get(identifier: string|
                    CoreOptimizer): Constructor<LayersOptimizer> {
  const coreOptimizerToConstructorMap:
      {[coreOptimizerTypeName: string]: Constructor<LayersOptimizer>} = {
        'AdagradOptimizer': Adagrad,
        'AdamOptimizer': Adam,
        'RMSPropOptimizer': RMSProp,
        'SGDOptimizer': SGD
      };

  const optimizerMap: {[optimizerName: string]: Constructor<LayersOptimizer>} =
      {Adagrad, Adam, RMSProp, SGD, adagrad, adam, rmsprop, sgd};

  if (typeof identifier === 'string') {
    if (identifier in optimizerMap) {
      return optimizerMap[identifier];
    }
    throw new ValueError(`Unknown Optimizer ${identifier}`);
  } else {
    const coreOptimizerTypeName = identifier.constructor.name;
    if (coreOptimizerTypeName in coreOptimizerToConstructorMap) {
      return coreOptimizerToConstructorMap[coreOptimizerTypeName];
    }
    throw new ValueError(
        `Unsupported core optimizer type: ${coreOptimizerTypeName}`);
  }
}
