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

import * as tfc from '@tensorflow/tfjs-core';
import {serialization, Tensor, tidy} from '@tensorflow/tfjs-core';
import {epsilon} from './backend/common';
import {deserializeKerasObject, serializeKerasObject} from './utils/generic_utils';

/**
 * Helper function used by many of the Constraints to find the L2Norms.
 */
function calcL2Norms(w: Tensor, axis: number): Tensor {
  return tidy(() => tfc.sqrt(tfc.sum(tfc.mulStrict(w, w), axis, true)));
}

/**
 * Base class for functions that impose constraints on weight values
 */
/**
 * @doc {
 *   heading: 'Constraints',
 *   subheading: 'Classes',
 *   namespace: 'constraints'
 * }
 */
export abstract class Constraint extends serialization.Serializable {
  /* Porting note: was __call__, apply chosen to match other similar choices */
  abstract apply(w: Tensor): Tensor;
  getConfig(): serialization.ConfigDict {
    return {};
  }
}

export interface MaxNormArgs {
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

export class MaxNorm extends Constraint {
  /** @nocollapse */
  static readonly className = 'MaxNorm';
  private maxValue: number;
  private axis: number;
  private readonly defaultMaxValue = 2;
  private readonly defaultAxis = 0;

  constructor(args: MaxNormArgs) {
    super();
    this.maxValue =
        args.maxValue != null ? args.maxValue : this.defaultMaxValue;
    this.axis = args.axis != null ? args.axis : this.defaultAxis;
  }

  apply(w: Tensor): Tensor {
    return tidy(() => {
      const norms = calcL2Norms(w, this.axis);
      const desired = tfc.clipByValue(norms, 0, this.maxValue);
      return tfc.mul(w, tfc.div(desired, tfc.add(epsilon(), norms)));
    });
  }

  getConfig(): serialization.ConfigDict {
    return {maxValue: this.maxValue, axis: this.axis};
  }
}
serialization.registerClass(MaxNorm);

export interface UnitNormArgs {
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

export class UnitNorm extends Constraint {
  /** @nocollapse */
  static readonly className = 'UnitNorm';
  private axis: number;
  private readonly defaultAxis = 0;
  constructor(args: UnitNormArgs) {
    super();
    this.axis = args.axis != null ? args.axis : this.defaultAxis;
  }

  apply(w: Tensor): Tensor {
    return tidy(
        () => tfc.div(w, tfc.add(epsilon(), calcL2Norms(w, this.axis))));
  }

  getConfig(): serialization.ConfigDict {
    return {axis: this.axis};
  }
}
serialization.registerClass(UnitNorm);

export class NonNeg extends Constraint {
  /** @nocollapse */
  static readonly className = 'NonNeg';

  apply(w: Tensor): Tensor {
    return tfc.relu(w);
  }
}
serialization.registerClass(NonNeg);

export interface MinMaxNormArgs {
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

export class MinMaxNorm extends Constraint {
  /** @nocollapse */
  static readonly className = 'MinMaxNorm';
  private minValue: number;
  private maxValue: number;
  private rate: number;
  private axis: number;
  private readonly defaultMinValue = 0.0;
  private readonly defaultMaxValue = 1.0;
  private readonly defaultRate = 1.0;
  private readonly defaultAxis = 0;

  constructor(args: MinMaxNormArgs) {
    super();
    this.minValue =
        args.minValue != null ? args.minValue : this.defaultMinValue;
    this.maxValue =
        args.maxValue != null ? args.maxValue : this.defaultMaxValue;
    this.rate = args.rate != null ? args.rate : this.defaultRate;
    this.axis = args.axis != null ? args.axis : this.defaultAxis;
  }

  apply(w: Tensor): Tensor {
    return tidy(() => {
      const norms = calcL2Norms(w, this.axis);
      const desired = tfc.add(
          tfc.mul(
              this.rate, tfc.clipByValue(norms, this.minValue, this.maxValue)),
          tfc.mul(1.0 - this.rate, norms));
      return tfc.mul(w, tfc.div(desired, tfc.add(epsilon(), norms)));
    });
  }

  getConfig(): serialization.ConfigDict {
    return {
      minValue: this.minValue,
      maxValue: this.maxValue,
      rate: this.rate,
      axis: this.axis
    };
  }
}
serialization.registerClass(MinMaxNorm);

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

export function serializeConstraint(constraint: Constraint):
    serialization.ConfigDictValue {
  return serializeKerasObject(constraint);
}

export function deserializeConstraint(
    config: serialization.ConfigDict,
    customObjects: serialization.ConfigDict = {}): Constraint {
  return deserializeKerasObject(
      config, serialization.SerializationMap.getMap().classNameMap,
      customObjects, 'constraint');
}

export function getConstraint(identifier: ConstraintIdentifier|
                              serialization.ConfigDict|Constraint): Constraint {
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
