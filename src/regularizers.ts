/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/* original source: keras/regularizers.py */

// tslint:disable:max-line-length
import {doc, Scalar, Tensor, zeros} from '@tensorflow/tfjs-core';

import * as K from './backend/tfjs_backend';
import {ConfigDict, ConfigDictValue, Constructor, Serializable} from './types';
import {ClassNameMap, deserializeKerasObject, serializeKerasObject} from './utils/generic_utils';
// tslint:enable:max-line-length

/**
 * Regularizer base class.
 */
export abstract class Regularizer extends Serializable {
  abstract apply(x: Tensor): Scalar;
}

export interface L1L2Config {
  /** L1 regularization rate. Defaults to 0.01. */
  l1?: number;
  /** L2 regularization rate. Defaults to 0.01. */
  l2?: number;
}

export interface L1Config {
  /** L1 regularization rate. Defaults to 0.01. */
  l1: number;
}

export interface L2Config {
  /** L2 regularization rate. Defaults to 0.01. */
  l2: number;
}

/**
 * Regularizer for L1 and L2 regularization.
 *
 * Adds a term to the loss to penalize large weights:
 * loss += sum(l1 * abs(x)) + sum(l2 * x^2)
 */
@doc({heading: 'Regularizers', namespace: 'regularizers'})
export class L1L2 extends Regularizer {
  static className = 'L1L2';

  private readonly l1: Scalar;
  private readonly l2: Scalar;
  private readonly hasL1: boolean;
  private readonly hasL2: boolean;
  constructor(config?: L1L2Config) {
    super();

    const l1 = config == null || config.l1 == null ? 0.01 : config.l1;
    const l2 = config == null || config.l2 == null ? 0.01 : config.l2;
    this.hasL1 = l1 !== 0;
    this.hasL2 = l2 !== 0;

    this.l1 = K.getScalar(l1);
    this.l2 = K.getScalar(l2);
  }

  /**
   * Porting note: Renamed from __call__.
   * @param x Variable of which to calculate the regularization score.
   */
  apply(x: Tensor): Scalar {
    let regularization: Tensor = zeros([1]);
    if (this.hasL1) {
      regularization =
          K.add(regularization, K.sum(K.scalarTimesArray(this.l1, K.abs(x))));
    }
    if (this.hasL2) {
      regularization = K.add(
          regularization, K.sum(K.scalarTimesArray(this.l2, K.square(x))));
    }
    return regularization.asScalar();
  }

  getConfig(): ConfigDict {
    return {'l1': this.l1.dataSync()[0], 'l2': this.l2.dataSync()[0]};
  }

  static fromConfig<T extends Serializable>(
      cls: Constructor<T>, config: ConfigDict): T {
    return new cls({l1: config.l1 as number, l2: config.l2 as number});
  }
}
ClassNameMap.register(L1L2);

/**
 * Regularizer for L1 regularization.
 *
 * Adds a term to the loss to penalize large weights:
 * loss += sum(l1 * abs(x))
 * @param config l1 config.
 */
export function l1(config?: L1Config) {
  return new L1L2({l1: config != null ? config.l1 : null, l2: 0});
}

/**
 * Regularizer for L2 regularization.
 *
 * Adds a term to the loss to penalize large weights:
 * loss += sum(l2 * x^2)
 * @param config l2 config.
 */
export function l2(config: L2Config) {
  return new L1L2({l2: config != null ? config.l2 : null, l1: 0});
}

/** @docinline */
export type RegularizerIdentifier = 'l1l2'|string;

// Maps the JavaScript-like identifier keys to the corresponding keras symbols.
export const REGULARIZER_IDENTIFIER_REGISTRY_SYMBOL_MAP:
    {[identifier in RegularizerIdentifier]: string} = {
      'l1l2': 'L1L2'
    };

export function serializeRegularizer(constraint: Regularizer): ConfigDictValue {
  return serializeKerasObject(constraint);
}

export function deserializeRegularizer(
    config: ConfigDict, customObjects: ConfigDict = {}): Regularizer {
  return deserializeKerasObject(
      config, ClassNameMap.getMap().pythonClassNameMap, customObjects,
      'regularizer');
}

export function getRegularizer(identifier: RegularizerIdentifier|ConfigDict|
                               Regularizer): Regularizer {
  if (identifier == null) {
    return null;
  }
  if (typeof identifier === 'string') {
    const className = identifier in REGULARIZER_IDENTIFIER_REGISTRY_SYMBOL_MAP ?
        REGULARIZER_IDENTIFIER_REGISTRY_SYMBOL_MAP[identifier] :
        identifier;
    const config = {className, config: {}};
    return deserializeRegularizer(config);
  } else if (identifier instanceof Regularizer) {
    return identifier;
  } else {
    return deserializeRegularizer(identifier);
  }
}
