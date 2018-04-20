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
 * Normalization layers.
 */

import {Tensor, util} from '@tensorflow/tfjs-core';

// tslint:disable:max-line-length
import * as K from '../backend/tfjs_backend';
import {Constraint, ConstraintIdentifier, getConstraint, serializeConstraint} from '../constraints';
import {InputSpec, Layer, LayerConfig} from '../engine/topology';
import {NotImplementedError, ValueError} from '../errors';
import {getInitializer, Initializer, InitializerIdentifier, serializeInitializer} from '../initializers';
import {getRegularizer, Regularizer, RegularizerIdentifier, serializeRegularizer} from '../regularizers';
import {Shape} from '../types';
import {ConfigDict, LayerVariable} from '../types';
import * as generic_utils from '../utils/generic_utils';
import {range} from '../utils/math_utils';
// tslint:enable:max-line-length

export interface BatchNormalizationLayerConfig extends LayerConfig {
  /**
   * The integer axis that should be normalized (typically the features axis).
   * Defaults to -1.
   *
   * For instance, after a `Conv2D` layer with `data_format="channels_first"`,
   * set `axis=1` in `batchNormalization`.
   */
  axis?: number;

  /**
   * Momentum of the moving average. Defaults to 0.99.
   */
  momentum?: number;

  /**
   * Small float added to the variance to avoid dividing by zero. Defaults to
   * 1e-3.
   */
  epsilon?: number;

  /**
   * If `true`, add offset of `beta` to normalized tensor.
   * If `false`, `beta` is ignored.
   * Defaults to `true`.
   */
  center?: boolean;

  /**
   * If `true`, multiply by `gamma`.
   * If `false`, `gamma` is not used.
   * When the next layer is linear (also e.g. `nn.relu`),
   * this can be disabled since the scaling will be done by the next layer.
   * Defaults to `true`.
   */
  scale?: boolean;

  /**
   * Initializer for the beta weight.
   *  Defaults to 'zeros'.
   */
  betaInitializer?: InitializerIdentifier|Initializer;

  /**
   * Initializer for the gamma weight.
   *  Defaults to `ones`.
   */
  gammaInitializer?: InitializerIdentifier|Initializer;

  /**
   * Initializer for the moving mean.
   * Defaults to `zeros`
   */
  movingMeanInitializer?: InitializerIdentifier|Initializer;

  /**
   * Initializer for the moving variance.
   *  Defaults to 'Ones'.
   */
  movingVarianceInitializer?: InitializerIdentifier|Initializer;

  /**
   * Constraint for the beta weight.
   */
  betaConstraint?: ConstraintIdentifier|Constraint;

  /**
   * Constraint for gamma weight.
   */
  gammaConstraint?: ConstraintIdentifier|Constraint;

  /**
   * Regularizer for the beta weight.
   */
  betaRegularizer?: RegularizerIdentifier|Regularizer;

  /**
   * Regularizer for the gamma weight.
   */
  gammaRegularizer?: RegularizerIdentifier|Regularizer;
}


/**
 * Batch normalization layer (Ioffe and Szegedy, 2014).
 *
 * Normalize the activations of the previous layer at each batch,
 * i.e. applies a transformation that maintains the mean activation
 * close to 0 and the activation standard deviation close to 1.
 *
 * Input shape:
 *   Arbitrary. Use the keyword argument `inputShape` (Array of integers, does
 *   not include the sample axis) when calling the constructor of this class,
 *   if this layer is used as a first layer in a model.
 *
 * Output shape:
 *   Same shape as input.
 *
 * References:
 *   - [Batch Normalization: Accelerating Deep Network Training by Reducing
 * Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
 */
export class BatchNormalization extends Layer {
  private readonly axis: number;
  private readonly momentum: number;
  private readonly epsilon: number;
  private readonly center: boolean;
  private readonly scale: boolean;
  private readonly betaInitializer: Initializer;
  private readonly gammaInitializer: Initializer;
  private readonly movingMeanInitializer: Initializer;
  private readonly movingVarianceInitializer: Initializer;
  private readonly betaConstraint: Constraint;
  private readonly gammaConstraint: Constraint;
  private readonly betaRegularizer: Regularizer;
  private readonly gammaRegularizer: Regularizer;
  private gamma: LayerVariable;
  private beta: LayerVariable;
  private movingMean: LayerVariable;
  private movingVariance: LayerVariable;

  constructor(config: BatchNormalizationLayerConfig) {
    super(config);
    this.supportsMasking = true;
    this.axis = config.axis == null ? -1 : config.axis;
    this.momentum = config.momentum == null ? 0.99 : config.momentum;
    this.epsilon = config.epsilon == null ? 1e-3 : config.epsilon;
    this.center = config.center == null ? true : config.center;
    this.scale = config.scale == null ? true : config.scale;
    this.betaInitializer = getInitializer(config.betaInitializer || 'zeros');
    this.gammaInitializer = getInitializer(config.gammaInitializer || 'ones');
    this.movingMeanInitializer =
        getInitializer(config.movingMeanInitializer || 'zeros');
    this.movingVarianceInitializer =
        getInitializer(config.movingVarianceInitializer || 'ones');
    this.betaConstraint = getConstraint(config.betaConstraint);
    this.gammaConstraint = getConstraint(config.gammaConstraint);
    this.betaRegularizer = getRegularizer(config.betaRegularizer);
    this.gammaRegularizer = getRegularizer(config.gammaRegularizer);
  }

  public build(inputShape: Shape|Shape[]): void {
    inputShape = generic_utils.getExactlyOneShape(inputShape);
    const axis = this.axis >= 0 ? this.axis : (this.axis + inputShape.length);
    const dim = inputShape[axis];
    if (dim == null) {
      throw new ValueError(
          `Axis ${axis} of input tensor should have a defined dimension but ` +
          `the layer received an input with shape ` +
          `${JSON.stringify(inputShape)}.`);
    }
    this.inputSpec =
        [new InputSpec({ndim: inputShape.length, axes: {[axis]: dim}})];
    const shape = [dim];
    if (this.scale) {
      this.gamma = this.addWeight(
          'gamma', shape, null, this.gammaInitializer, this.gammaRegularizer,
          true, this.gammaConstraint);
    }
    if (this.center) {
      this.beta = this.addWeight(
          'beta', shape, null, this.betaInitializer, this.betaRegularizer, true,
          this.betaConstraint);
    }
    this.movingMean = this.addWeight(
        'moving_mean', shape, null, this.movingMeanInitializer, null, false);
    this.movingVariance = this.addWeight(
        'moving_variance', shape, null, this.movingVarianceInitializer, null,
        false);
    this.built = true;
  }

  // tslint:disable-next-line:no-any
  call(inputs: Tensor|Tensor[], kwargs: any): Tensor|Tensor[] {
    const training = kwargs['training'] == null ? false : kwargs['training'];
    const input = generic_utils.getExactlyOneTensor(inputs);
    const inputShape = K.shape(input);
    const ndim = inputShape.length;
    const reductionAxes = range(0, ndim);
    const axis = this.axis >= 0 ? this.axis : (this.axis + ndim);
    reductionAxes.splice(axis, 1);
    const broadcastShape = generic_utils.pyListRepeat(1, ndim);
    broadcastShape[axis] = inputShape[axis];

    const sortedReductionAxes = reductionAxes.slice();
    sortedReductionAxes.sort();
    const needsBroadcasting = !util.arraysEqual(
        sortedReductionAxes, range(0, ndim).slice(0, ndim - 1));

    const normalizeInference: () => Tensor = () => {
      if (needsBroadcasting) {
        const broadcastMovingMean =
            K.reshape(this.movingMean.read(), broadcastShape);
        const broadcastMovingVariance =
            K.reshape(this.movingVariance.read(), broadcastShape);
        const broadcastBeta =
            this.center ? K.reshape(this.beta.read(), broadcastShape) : null;
        const broadcastGamma =
            this.scale ? K.reshape(this.gamma.read(), broadcastShape) : null;
        return K.batchNormalization(
            input, broadcastMovingMean, broadcastMovingVariance, broadcastBeta,
            broadcastGamma, this.epsilon);
      } else {
        return K.batchNormalization(
            input, this.movingMean.read(), this.movingVariance.read(),
            this.beta == null ? null : this.beta.read(),
            this.gamma == null ? null : this.gamma.read(), this.epsilon);
      }
    };

    if (!training) {
      return normalizeInference();
    }

    throw new NotImplementedError(
        'BatchNormalization.call() has not been implemented for training ' +
        'mode yet.');
  }

  getClassName(): string {
    return 'BatchNormalization';
  }

  getConfig(): ConfigDict {
    const config: ConfigDict = {
      axis: this.axis,
      momentum: this.momentum,
      epsilon: this.epsilon,
      center: this.center,
      scale: this.scale,
      betaInitializer: serializeInitializer(this.betaInitializer),
      gammaInitializer: serializeInitializer(this.gammaInitializer),
      movingMeanInitializer: serializeInitializer(this.movingMeanInitializer),
      movingVarianceInitializer:
          serializeInitializer(this.movingVarianceInitializer),
      betaRegularizer: serializeRegularizer(this.betaRegularizer),
      gammaRegularizer: serializeRegularizer(this.gammaRegularizer),
      betaConstraint: serializeConstraint(this.betaConstraint),
      gammaConstraint: serializeConstraint(this.gammaConstraint)
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
generic_utils.ClassNameMap.register('BatchNormalization', BatchNormalization);
