/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
import * as regularizers from './regularizers';
// tslint:disable-next-line:max-line-length
import {L1Args, L1L2, L1L2Args, L2Args, Regularizer} from './regularizers';

/**
 * Regularizer for L1 and L2 regularization.
 *
 * Adds a term to the loss to penalize large weights:
 * loss += sum(l1 * abs(x)) + sum(l2 * x^2)
 *
 * @doc {heading: 'Regularizers', namespace: 'regularizers'}
 */
export function l1l2(config?: L1L2Args): Regularizer {
  return new L1L2(config);
}

/**
 * Regularizer for L1 regularization.
 *
 * Adds a term to the loss to penalize large weights:
 * loss += sum(l1 * abs(x))
 * @param args l1 config.
 *
 * @doc {heading: 'Regularizers', namespace: 'regularizers'}
 */
export function l1(config?: L1Args): Regularizer {
  return regularizers.l1(config);
}

/**
 * Regularizer for L2 regularization.
 *
 * Adds a term to the loss to penalize large weights:
 * loss += sum(l2 * x^2)
 * @param args l2 config.
 *
 * @doc {heading: 'Regularizers', namespace: 'regularizers'}
 */
export function l2(config?: L2Args): Regularizer {
  return regularizers.l2(config);
}
