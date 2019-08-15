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
  'elu', 'hard_sigmoid', 'linear', 'relu', 'relu6', 'selu', 'sigmoid',
  'softmax', 'softplus', 'softsign', 'tanh'
]);

/**
 * A type representing the strings that are valid loss names.
 */
export type ActivationSerialization = typeof activationOptions[number];

// Sad that we have to do all this just for hard_sigmoid vs. hardSigmoid.
// TODO(soergel): Move the CamelCase versions back out of keras_format
// e.g. to src/common.ts.  Maybe even duplicate *all* of these to be pedantic?
/** @docinline */
export type ActivationIdentifier = 'elu'|'hardSigmoid'|'linear'|'relu'|'relu6'|
    'selu'|'sigmoid'|'softmax'|'softplus'|'softsign'|'tanh';
