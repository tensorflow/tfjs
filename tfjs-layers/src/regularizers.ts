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

import * as tfc from '@tensorflow/tfjs-core';
import {abs, add, Scalar, serialization, sum, Tensor, tidy, zeros} from '@tensorflow/tfjs-core';
import * as K from './backend/tfjs_backend';
import {deserializeKerasObject, serializeKerasObject} from './utils/generic_utils';

/**
 * Regularizer base class.
 */
export abstract class Regularizer extends serialization.Serializable {
  abstract apply(x: Tensor): Scalar;
}

export interface L1L2Args {
  /** L1 regularization rate. Defaults to 0.01. */
  l1?: number;
  /** L2 regularization rate. Defaults to 0.01. */
  l2?: number;
}

export interface L1Args {
  /** L1 regularization rate. Defaults to 0.01. */
  l1: number;
}

export interface L2Args {
  /** L2 regularization rate. Defaults to 0.01. */
  l2: number;
}

export class L1L2 extends Regularizer {
  /** @nocollapse */
  static className = 'L1L2';

  private readonly l1: number;
  private readonly l2: number;
  private readonly hasL1: boolean;
  private readonly hasL2: boolean;
  constructor(args?: L1L2Args) {
    super();

    this.l1 = args == null || args.l1 == null ? 0.01 : args.l1;
    this.l2 = args == null || args.l2 == null ? 0.01 : args.l2;
    this.hasL1 = this.l1 !== 0;
    this.hasL2 = this.l2 !== 0;
  }

  /**
   * Porting note: Renamed from __call__.
   * @param x Variable of which to calculate the regularization score.
   */
  apply(x: Tensor): Scalar {
    return tidy(() => {
      let regularization: Tensor = zeros([1]);
      if (this.hasL1) {
        regularization = add(regularization, sum(tfc.mul(this.l1, abs(x))));
      }
      if (this.hasL2) {
        regularization =
            add(regularization, sum(tfc.mul(this.l2, K.square(x))));
      }
      return regularization.asScalar();
    });
  }

  getConfig(): serialization.ConfigDict {
    return {'l1': this.l1, 'l2': this.l2};
  }

  /** @nocollapse */
  static fromConfig<T extends serialization.Serializable>(
      cls: serialization.SerializableConstructor<T>,
      config: serialization.ConfigDict): T {
    return new cls({l1: config['l1'] as number, l2: config['l2'] as number});
  }
}
serialization.registerClass(L1L2);

export function l1(args?: L1Args) {
  return new L1L2({l1: args != null ? args.l1 : null, l2: 0});
}

export function l2(args: L2Args) {
  return new L1L2({l2: args != null ? args.l2 : null, l1: 0});
}

/** @docinline */
export type RegularizerIdentifier = 'l1l2'|string;

// Maps the JavaScript-like identifier keys to the corresponding keras symbols.
export const REGULARIZER_IDENTIFIER_REGISTRY_SYMBOL_MAP:
    {[identifier in RegularizerIdentifier]: string} = {
      'l1l2': 'L1L2'
    };

export function serializeRegularizer(constraint: Regularizer):
    serialization.ConfigDictValue {
  return serializeKerasObject(constraint);
}

export function deserializeRegularizer(
    config: serialization.ConfigDict,
    customObjects: serialization.ConfigDict = {}): Regularizer {
  return deserializeKerasObject(
      config, serialization.SerializationMap.getMap().classNameMap,
      customObjects, 'regularizer');
}

export function getRegularizer(identifier: RegularizerIdentifier|
                               serialization.ConfigDict|
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
