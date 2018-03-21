/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/* Original source: keras/contraints.py */

// tslint:disable:max-line-length
import {doc, Tensor} from '@tensorflow/tfjs-core';

import * as K from './backend/deeplearnjs_backend';
import {ConfigDict, ConfigDictValue} from './types';
import {ClassNameMap, deserializeKerasObject, serializeKerasObject} from './utils/generic_utils';
// tslint:enable:max-line-length

/**
 * Helper function used by many of the Constraints to find the L2Norms.
 */
function calcL2Norms(w: Tensor, axis: number): Tensor {
  return K.sqrt(K.sum(K.square(w), axis, true));
}

/**
 * Base class for functions that impose constraints on weight values
 */
@doc({heading: 'Constraints', subheading: 'Classes', namespace: 'constraints'})
export abstract class Constraint {
  /* Porting note: was __call__, apply chosen to match other similar choices */
  abstract apply(w: Tensor): Tensor;
  getConfig(): ConfigDict {
    return {};
  }
}

export interface MaxNormConfig {
  /**
   * Maximum norm for incoming weights
   */
  maxValue?: number;
  /**
   * Axis along which to calculate norms.
   *
   *  For instance, in a `Dense` layer the weight matrix
   *  has shape `[inputDim, outputDim]`,
   *  set `axis` to `0` to constrain each weight vector
   *  of length `[inputDim,]`.
   *  In a `Conv2D` layer with `dataFormat="channels_last"`,
   *  the weight tensor has shape
   *  `[rows, cols, inputDepth, outputDepth]`,
   *  set `axis` to `[0, 1, 2]`
   *  to constrain the weights of each filter tensor of size
   *  `[rows, cols, inputDepth]`.
   */
  axis?: number;
}

/**
 * MaxNorm weight constraint.
 *
 * Constrains the weights incident to each hidden unit
 * to have a norm less than or equal to a desired value.
 *
 * References
 *       - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting
 * Srivastava, Hinton, et al.
 * 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
 * @docalias Constraint
 */
export class MaxNorm extends Constraint {
  private maxValue: number;
  private axis: number;
  private readonly defaultMaxValue = 2;
  private readonly defaultAxis = 0;

  constructor(config: MaxNormConfig) {
    super();
    this.maxValue =
        config.maxValue != null ? config.maxValue : this.defaultMaxValue;
    this.axis = config.axis != null ? config.axis : this.defaultAxis;
  }

  apply(w: Tensor): Tensor {
    const norms = calcL2Norms(w, this.axis);
    const desired = K.clip(norms, 0, this.maxValue);
    return K.multiply(
        w,
        K.divide(desired, K.scalarPlusArray(K.getScalar(K.epsilon()), norms)));
  }

  getConfig(): ConfigDict {
    return {maxValue: this.maxValue, axis: this.axis};
  }
}
ClassNameMap.register('MaxNorm', MaxNorm);

export interface UnitNormConfig {
  /**
   * Axis along which to calculate norms.
   *
   * For instance, in a `Dense` layer the weight matrix
   * has shape `[inputDim, outputDim]`,
   * set `axis` to `0` to constrain each weight vector
   * of length `[inputDim,]`.
   * In a `Conv2D` layer with `dataFormat="channels_last"`,
   * the weight tensor has shape
   * [rows, cols, inputDepth, outputDepth]`,
   * set `axis` to `[0, 1, 2]`
   * to constrain the weights of each filter tensor of size
   * `[rows, cols, inputDepth]`.
   */
  axis?: number;
}

/**
 * Constrains the weights incident to each hidden unit to have unit norm.
 * @docalias Constraint
 */
export class UnitNorm extends Constraint {
  private axis: number;
  private readonly defaultAxis = 0;

  constructor(config: UnitNormConfig) {
    super();
    this.axis = config.axis != null ? config.axis : this.defaultAxis;
  }

  apply(w: Tensor): Tensor {
    return K.divide(
        w,
        K.scalarPlusArray(K.getScalar(K.epsilon()), calcL2Norms(w, this.axis)));
  }

  getConfig(): ConfigDict {
    return {axis: this.axis};
  }
}
ClassNameMap.register('UnitNorm', UnitNorm);

/**
 * Constains the weight to be non-negative.
 * @docalias Constraint
 */
export class NonNeg extends Constraint {
  apply(w: Tensor): Tensor {
    return K.relu(w);
  }
}
ClassNameMap.register('NonNeg', NonNeg);

export interface MinMaxNormConfig {
  /**
   * Minimum norm for incoming weights
   */
  minValue?: number;
  /**
   * Maximum norm for incoming weights
   */
  maxValue?: number;
  /**
   * Axis along which to calculate norms.
   * For instance, in a `Dense` layer the weight matrix
   * has shape `[inputDim, outputDim]`,
   * set `axis` to `0` to constrain each weight vector
   * of length `[inputDim,]`.
   * In a `Conv2D` layer with `dataFormat="channels_last"`,
   * the weight tensor has shape
   * `[rows, cols, inputDepth, outputDepth]`,
   * set `axis` to `[0, 1, 2]`
   * to constrain the weights of each filter tensor of size
   * `[rows, cols, inputDepth]`.
   */
  axis?: number;
  /**
   * Rate for enforcing the constraint: weights will be rescaled to yield:
   * `(1 - rate) * norm + rate * norm.clip(minValue, maxValue)`.
   * Effectively, this means that rate=1.0 stands for strict
   * enforcement of the constraint, while rate<1.0 means that
   * weights will be rescaled at each step to slowly move
   * towards a value inside the desired interval.
   */
  rate?: number;
}

/** @docalias Constraint */
export class MinMaxNorm extends Constraint {
  private minValue: number;
  private maxValue: number;
  private rate: number;
  private axis: number;
  private readonly defaultMinValue = 0.0;
  private readonly defaultMaxValue = 1.0;
  private readonly defaultRate = 1.0;
  private readonly defaultAxis = 0;

  constructor(config: MinMaxNormConfig) {
    super();
    this.minValue =
        config.minValue != null ? config.minValue : this.defaultMinValue;
    this.maxValue =
        config.maxValue != null ? config.maxValue : this.defaultMaxValue;
    this.rate = config.rate != null ? config.rate : this.defaultRate;
    this.axis = config.axis != null ? config.axis : this.defaultAxis;
  }

  apply(w: Tensor): Tensor {
    const norms = calcL2Norms(w, this.axis);
    const desired = K.add(
        K.scalarTimesArray(
            K.getScalar(this.rate),
            K.clip(norms, this.minValue, this.maxValue)),
        K.scalarTimesArray(K.getScalar(1.0 - this.rate), norms));
    return K.multiply(
        w,
        K.divide(desired, K.scalarPlusArray(K.getScalar(K.epsilon()), norms)));
  }

  getConfig(): ConfigDict {
    return {
      minValue: this.minValue,
      maxValue: this.maxValue,
      rate: this.rate,
      axis: this.axis
    };
  }
}
ClassNameMap.register('MinMaxNorm', MinMaxNorm);

/** @docinline */
export type ConstraintIdentifier =
    'maxNorm'|'minMaxNorm'|'nonNeg'|'unitNorm'|string;

// Maps the JavaScript-like identifier keys to the corresponding registry
// symbols.
export const CONSTRAINT_IDENTIFIER_REGISTRY_SYMBOL_MAP:
    {[identifier in ConstraintIdentifier]: string} = {
      'maxNorm': 'MaxNorm',
      'minMaxNorm': 'MinMaxNorm',
      'nonNeg': 'NonNeg',
      'unitNorm': 'UnitNorm'
    };

export function serializeConstraint(constraint: Constraint): ConfigDictValue {
  return serializeKerasObject(constraint);
}

export function deserializeConstraint(
    config: ConfigDict, customObjects: ConfigDict = {}): Constraint {
  return deserializeKerasObject(
      config, ClassNameMap.getMap().pythonClassNameMap, customObjects,
      'constraint');
}

export function getConstraint(identifier: ConstraintIdentifier|ConfigDict|
                              Constraint): Constraint {
  if (identifier == null) {
    return null;
  }
  if (typeof identifier === 'string') {
    const className = identifier in CONSTRAINT_IDENTIFIER_REGISTRY_SYMBOL_MAP ?
        CONSTRAINT_IDENTIFIER_REGISTRY_SYMBOL_MAP[identifier] :
        identifier;
    const config = {className, config: {}};
    return deserializeConstraint(config);
  } else if (identifier instanceof Constraint) {
    return identifier;
  } else {
    return deserializeConstraint(identifier);
  }
}
