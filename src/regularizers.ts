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
import {Scalar, Tensor, zeros} from 'deeplearn';

import * as K from './backend/deeplearnjs_backend';
import {LayerVariable} from './types';
import {ConfigDict, ConfigDictValue} from './types';
import * as generic_utils from './utils/generic_utils';
import {ClassNameMap, deserializeKerasObject, serializeKerasObject} from './utils/generic_utils';
// tslint:enable:max-line-length

/**
 * Regularizer base class.
 */
export abstract class Regularizer { abstract apply(x: LayerVariable): Tensor; }

/**
 * Regularizer for L1 and L2 regularization.
 */
export class L1L2 extends Regularizer {
  private readonly l1: Scalar;
  private readonly l2: Scalar;
  private readonly hasL1: boolean;
  private readonly hasL2: boolean;
  constructor(l1 = 0.0, l2 = 0.0) {
    super();
    this.hasL1 = l1 !== 0;
    this.hasL2 = l2 !== 0;
    this.l1 = K.getScalar(l1);
    this.l2 = K.getScalar(l2);
  }
  /**
   * Porting note: Renamed from __call__.
   * @param x Variable of which to calculate the regularization score.
   */
  apply(x: LayerVariable): Tensor {
    let regularization: Tensor = zeros([1]);
    if (this.hasL1) {
      regularization = K.add(
          regularization, K.sum(K.scalarTimesArray(this.l1, K.abs(x.read()))));
    }
    if (this.hasL2) {
      regularization = K.add(
          regularization,
          K.sum(K.scalarTimesArray(this.l2, K.square(x.read()))));
    }
    return regularization;
  }
  getConfig(): ConfigDict {
    return {'l1': this.l1.dataSync()[0], 'l2': this.l2.dataSync()[0]};
  }
  static fromConfig(cls: generic_utils.Constructor<L1L2>, config: ConfigDict):
      L1L2 {
    return new L1L2(config.l1 as number, config.l2 as number);
  }
}
ClassNameMap.register('L1L2', L1L2);

export function l1(l = 0.01): Regularizer {
  return new L1L2(l);
}

export function l2(l = 0.01): Regularizer {
  return new L1L2(0, l);
}

export function l1_l2(l1 = 0.01, l2 = 0.01): Regularizer {
  return new L1L2(l1, l2);
}

export function serializeRegularizer(constraint: Regularizer): ConfigDictValue {
  return serializeKerasObject(constraint);
}

export function deserializeRegularizer(
    config: ConfigDict, customObjects: ConfigDict = {}): Regularizer {
  return deserializeKerasObject(
      config, ClassNameMap.getMap().pythonClassNameMap, customObjects,
      'regularizer');
}

export function getRegularizer(identifier: string|ConfigDict|
                               Regularizer): Regularizer {
  if (identifier == null) {
    return null;
  }
  if (typeof identifier === 'string') {
    const config = {className: identifier, config: {}};
    return deserializeRegularizer(config);
  } else if (identifier instanceof Regularizer) {
    return identifier;
  } else {
    return deserializeRegularizer(identifier);
  }
}
