/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/* Original source: keras/optimizers.py */

import {Scalar, scalar, Tensor} from 'deeplearn';

import * as K from './backend/deeplearnjs_backend';
import {NotImplementedError, ValueError} from './errors';
import {DType} from './types';
import {ConfigDict, LayerVariable} from './types';
import * as generic_utils from './utils/generic_utils';

function clipNorm(gradient: Tensor, clipnorm: number, norm: number): Tensor {
  if (clipnorm <= 0) {
    return gradient;
  }
  if (norm >= clipnorm) {
    return K.scalarTimesArray(K.getScalar(clipnorm / norm), gradient);
  }
  return gradient;
}

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
export abstract class Optimizer {
  clipnorm: number;
  clipvalue: number;
  weights: LayerVariable[];

  constructor(config: OptimizerConfig) {
    this.clipnorm = config.clipnorm;
    this.clipvalue = config.clipvalue;
  }

  /**
   * Calculates the gradients based on the loss and params (variables) passed in
   * and updates the params.
   *
   * Porting note: This is getUpdates() in PyKeras. However
   * TensorFlow.js Layers assumes an eager-execution backend,
   * making it unnecessary to create update ops.
   *
   * @param lossFn A function to calculate the loss.
   * @param params The variables to optimize/update.
   */
  abstract updateVariables(lossFn: () => Scalar, params: LayerVariable[]): void;

  getGradients(lossFn: () => Scalar, params: LayerVariable[]): Tensor[] {
    let grads = K.gradients(lossFn, params);
    if (this.clipnorm != null && this.clipnorm > 0) {
      const sumOfSquaredGrads = grads.map(g => K.sum(K.square(g)));
      // TODO(cais): Remove dataSync() to benefit speed.
      const sumAcrossAllGrads = sumOfSquaredGrads.reduce(
          (prevValue, curValue) => prevValue + curValue.dataSync()[0], 0);
      const norm = Math.sqrt(sumAcrossAllGrads);
      grads = grads.map(g => clipNorm(g, this.clipnorm, norm));
    }
    if (this.clipvalue != null && this.clipvalue > 0) {
      grads = grads.map(g => K.clip(g, -this.clipvalue, this.clipvalue));
    }
    return grads;
  }
  // TODO(michaelterry): Add get/setWeights, if needed.
  getConfig(): ConfigDict {
    const config: ConfigDict = {};
    if (this.clipnorm != null) {
      config['clipnorm'] = this.clipnorm;
    }
    if (this.clipvalue != null) {
      config['clipvalue'] = this.clipvalue;
    }
    return config;
  }

  static fromConfig<T>(cls: generic_utils.Constructor<T>, config: ConfigDict):
      T {
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
export class SGD extends Optimizer {
  /** Number of iterations performed. */
  iterations: number;
  // TODO(cais): This needs to be turned into a Variable for serialization.
  /** Learning rate. */
  lr: number;
  momentum: number;
  momentumScalar: Scalar;
  decay: number;
  nesterov: boolean;
  /**
   * Tracks momentum over time. These are paired with specific sets of
   * variables the first time updateVariables() is called with a given set of
   * variables.
   */
  private momentsMap: {[variableIDs: string]: LayerVariable[]} = {};
  constructor(config: SGDConfig) {
    super(config);
    this.iterations = 0;
    this.lr = (config.lr == null) ? 0.01 : config.lr;
    this.momentum = (config.momentum == null) ? 0.0 : config.momentum;
    this.momentumScalar = K.getScalar(this.momentum);
    this.decay = (config.decay == null) ? 0.0 : config.decay;
    this.nesterov = (config.nesterov == null) ? false : config.nesterov;

    if (this.lr < 0) {
      throw new ValueError(
          `Invalid lr (${this.lr}). Must be >= 0 or undefined.`);
    }
    if (this.momentum < 0) {
      throw new ValueError(
          `Invalid momentum (${this.momentum}). Must be >= 0 or undefined.`);
    }
    if (this.decay < 0) {
      throw new ValueError(
          `Invalid decay (${this.decay}). Must be >= 0 or undefined.`);
    }
  }

  /**
   * Calculates the gradients based on the loss and params (variables) passed in
   * and updates the params.
   *
   * @param lossFn A function to calculate the loss.
   * @param params The variables to optimize/update.
   */
  updateVariables(lossFn: () => Scalar, params: LayerVariable[]): void {
    const variablesKey = params.map(x => x.id).join(':');
    if (!(variablesKey in this.momentsMap)) {
      // First time updateVariables() is called with these variables. Initialize
      // moments for these variables.
      const shapes = params.map(p => K.intShape(p));
      this.momentsMap[variablesKey] =
          shapes.map(shape => K.zerosVariable(shape));
    }

    const moments = this.momentsMap[variablesKey];

    // TODO(cais): Populate this.weights.

    const grads = this.getGradients(lossFn, params);
    this.iterations++;

    let lr = this.lr;
    if (this.decay > 0) {
      lr *= 1 / (1 + this.decay * this.iterations);
      // TODO(cais): This should be done with Tensors/Scalars for speed.
    }
    const lrScalar = K.getScalar(lr);
    for (let i = 0; i < params.length; i++) {
      const param = params[i];
      const gradient = grads[i];
      const moment = moments[i];
      const negLRXgradient = K.neg(K.scalarTimesArray(lrScalar, gradient));
      const velocity = K.add(
          K.scalarTimesArray(this.momentumScalar, moment.read()),
          negLRXgradient);
      K.update(moment, velocity);
      let newParamValues: Tensor;
      if (this.nesterov) {
        newParamValues = K.add(
            param.read(),
            K.add(
                K.scalarTimesArray(this.momentumScalar, velocity),
                negLRXgradient));
      } else {
        newParamValues = K.add(param.read(), velocity);
      }
      // TODO(michaelterry): Apply any constraints
      K.update(param, newParamValues);
    }
  }

  // TODO(michaelterry): Add getConfig()
}

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
export class Adam extends Optimizer {
  iterations: LayerVariable;
  // TODO(cais): This probably needs to be turned into a variable for
  //   serialization / deserialization.
  lr: LayerVariable;
  beta1: LayerVariable;
  beta2: LayerVariable;
  decay: LayerVariable;
  epsilon: Scalar;
  initialDecay: number;
  amsgrad: boolean;
  private oneFloat: Tensor;
  private oneInt: Tensor;
  private ms: LayerVariable[];
  private vs: LayerVariable[];

  constructor(config: AdamConfig) {
    super(config);
    K.nameScope(this.constructor.name, () => {
      // TODO(cais): Use int64 type once supported.
      this.iterations =
          new LayerVariable(K.getScalar(0), DType.int32, null, false);
      this.lr = new LayerVariable(
          K.getScalar(config.lr == null ? 0.001 : config.lr), null, 'lr',
          false);
      this.beta1 = new LayerVariable(
          K.getScalar(config.beta_1 == null ? 0.9 : config.beta_1), null,
          'beta_1', false);
      this.beta2 = new LayerVariable(
          K.getScalar(config.beta_2 == null ? 0.999 : config.beta_2), null,
          'beta_2', false);
      this.decay = new LayerVariable(
          K.getScalar(config.decay == null ? 0 : config.decay), null, 'decay',
          false);
    });
    this.epsilon =
        scalar(config.epsilon == null ? K.epsilon() : config.epsilon);
    this.initialDecay = config.decay == null ? 0 : config.decay;
    this.amsgrad = config.amsgrad;
    this.oneFloat = K.getScalar(1);
    this.oneInt = scalar(1, DType.int32);
  }

  /**
   * Calculates the gradients based on the loss and params (variables) passed in
   * and updates the params.
   *
   * @param lossFn A function to calculate the loss.
   * @param params The variables to optimize/update.
   */
  updateVariables(lossFn: () => Scalar, params: LayerVariable[]): void {
    const grads = this.getGradients(lossFn, params);

    const updates: Array<() => void> = [];
    updates.push(() => {
      K.update(this.iterations, K.add(this.iterations.read(), this.oneInt));
    });

    const lr = this.lr;
    const iterationsFloat = K.cast(this.iterations.read(), DType.float32);
    if (this.initialDecay > 0) {
      const lrMultiplier = K.divide(
          this.oneFloat,
          K.add(this.oneFloat, K.multiply(this.decay.read(), iterationsFloat)));
      K.update(lr, K.multiply(lr.read(), lrMultiplier));
    }
    const t = K.add(this.iterations.read(), this.oneInt);
    const oneMinusBeta2Pow =
        K.subtract(this.oneFloat, K.pow(this.beta2.read(), t));
    const oneMinusBeta1Pow =
        K.subtract(this.oneFloat, K.pow(this.beta1.read(), t));
    const lrT = K.multiply(
        this.lr.read(), K.divide(K.sqrt(oneMinusBeta2Pow), oneMinusBeta1Pow));

    // Porting Note: Due to the imperative nature of tfjs-layers' backend, `ms`
    // and `vs` need to be member variables of this class and initialized the
    // first time the optimizer's `updateVariables` method is called..
    if (this.ms == null) {
      this.ms = params.map(p => K.zerosVariable(p.shape, p.dtype));
    }
    if (this.vs == null) {
      this.vs = params.map(p => K.zerosVariable(p.shape, p.dtype));
    }
    if (this.amsgrad) {
      throw new NotImplementedError(
          'The support for amsgrad in Adam optimizer is not implemented yet');
    }
    // TODO(cais): Add vhats this.weights.
    this.weights = [this.iterations].concat(this.ms).concat(this.vs);

    for (let i = 0; i < params.length; ++i) {
      const p = params[i];
      const g = grads[i];
      const m = this.ms[i];
      const v = this.vs[i];
      const mT = K.add(
          K.multiply(this.beta1.read(), m.read()),
          K.multiply(K.subtract(this.oneFloat, this.beta1.read()), g));
      const vT = K.add(
          K.multiply(this.beta2.read(), v.read()),
          K.multiply(
              K.subtract(this.oneFloat, this.beta2.read()), K.square(g)));
      let pT: Tensor;
      if (this.amsgrad) {
        throw new NotImplementedError(
            'The support for amsgrad in Adam optimizer is not implemented yet');
      } else {
        pT = K.subtract(
            p.read(),
            K.divide(K.multiply(lrT, mT), K.add(K.sqrt(vT), this.epsilon)));
      }
      updates.push(() => {
        K.update(m, mT);
        K.update(v, vT);
      });
      const newP = pT;
      if (p.constraint != null) {
        throw new NotImplementedError(
            'Adam optimizer does not support variable constraints yet.');
      }
      updates.push(() => {
        K.update(p, newP);
      });
    }

    // Apply all the updates.
    for (const update of updates) {
      update();
    }
  }

  getConfig(): ConfigDict {
    const config: ConfigDict = {
      lr: this.lr.read().get(),
      beta1: this.beta1.read().get(),
      beta2: this.beta2.read().get(),
      decay: this.decay.read().get(),
      epsilon: this.epsilon.get(),
      amsgrad: this.amsgrad
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}

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
export class RMSProp extends Optimizer {
  lr: LayerVariable;
  rho: LayerVariable;
  decay: LayerVariable;
  iterations: LayerVariable;
  epsilon: Scalar;
  initialDecay: number;

  weights: LayerVariable[];
  updates: Tensor[];

  constructor(config: RMSPropConfig) {
    super(config);
    this.initialDecay = config.decay == null ? 0 : config.decay;
    K.nameScope(this.constructor.name, () => {
      // TODO(cais): Use int64 type once supported.
      this.iterations = K.variable(K.getScalar(0, DType.int32), DType.int32);
      this.lr =
          K.variable(scalar(config.lr == null ? 0.001 : config.lr), null, 'lr');
      this.rho = K.variable(
          scalar(config.rho == null ? 0.9 : config.rho), null, 'rho');
      this.decay = K.variable(scalar(this.initialDecay), null, 'decay');
    });
    this.epsilon =
        scalar(config.epsilon == null ? K.epsilon() : config.epsilon);
    this.weights = null;
  }

  updateVariables(lossFn: () => Scalar, params: LayerVariable[]): void {
    if (this.weights === null) {
      this.weights = [];
      for (const p of params) {
        this.weights.push(
            K.variable(K.zeros(K.intShape(p.read()), K.dtype(p.read()))));
      }
    } else if (this.weights.length !== params.length) {
      throw new ValueError('Number of params changed mid-training');
    }
    const grads = this.getGradients(lossFn, params);
    const updates: Array<() => void> = [];
    updates.push(() => {
      K.update(
          this.iterations,
          K.add(this.iterations.read(), K.getScalar(1, DType.int32)));
    });

    const lr = this.lr;
    const iterationsFloat = K.cast(this.iterations.read(), DType.float32);

    if (this.initialDecay > 0) {
      const lrMultiplier = K.divide(
          K.getScalar(1),
          K.add(
              K.getScalar(1), K.multiply(this.decay.read(), iterationsFloat)));
      K.update(lr, K.multiply(lr.read(), lrMultiplier));
    }
    for (const index in params) {
      const p = params[index];
      const g = grads[index];
      const a = this.weights[index];
      const newA = K.add(
          K.multiply(this.rho.read(), a.read()),
          K.multiply(K.subtract(K.getScalar(1), this.rho.read()), K.square(g)));
      updates.push(() => {
        K.update(a, newA);
      });
      const newP = K.subtract(
          p.read(),
          K.multiply(
              this.lr.read(), K.divide(g, K.add(K.sqrt(newA), this.epsilon))));
      if (p['constraint'] != null) {
        throw new NotImplementedError(
            'RMSProp optimizer does not support variable constraints yet.');
      }
      updates.push(() => {
        K.update(p, newP);
      });
    }
    // Apply all the updates.
    for (const update of updates) {
      update();
    }
  }
  getConfig(): ConfigDict {
    const config: ConfigDict = {
      lr: this.lr.read().get(),
      rho: this.rho.read().get(),
      decay: this.decay.read().get(),
      epsilon: this.epsilon.get(),
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
export const sgd = SGD;
export const adam = Adam;
export const rmsprop = RMSProp;

// Add (de)serialize()

// Porting note: This diverges from the PyKeras implementation and may need to
// change based on (de)serialization requirements.
export function get(identifier: string): generic_utils.Constructor<Optimizer> {
  const optimizerMap:
      {[optimizerName: string]: generic_utils.Constructor<Optimizer>} =
          {Adam, SGD, adam, sgd, RMSProp, rmsprop};
  if (identifier in optimizerMap) {
    return optimizerMap[identifier];
  }
  throw new ValueError(`Unknown Optimizer ${identifier}`);
}
