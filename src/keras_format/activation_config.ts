/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {stringLiteralArray} from './utils';

/**
 * List of all known activation names.
 */
export const activationOptions = stringLiteralArray([
  'elu', 'hardSigmoid', 'linear', 'relu', 'relu6', 'selu', 'sigmoid', 'softmax',
  'softplus', 'softsign', 'tanh'
]);

/**
 * A type representing the strings that are valid loss names.
 */
export type ActivationIdentifier = typeof activationOptions[number];
