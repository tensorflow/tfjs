/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {BaseSerialization} from './types';

export type L1L2Config = {
  l1?: number;
  l2?: number;
};

export type L1L2Serialization = BaseSerialization<'L1L2', L1L2Config>;

// Update regularizerClassNames below in concert with this.
export type RegularizerSerialization = L1L2Serialization;

export type RegularizerClassName = RegularizerSerialization['class_name'];

// We can't easily extract a string[] from the string union type, but we can
// recapitulate the list, enforcing at compile time that the values are valid
// and that we have the right number of them.

/**
 * A string array of valid Regularizer class names.
 *
 * This is guaranteed to match the `RegularizerClassName` union type.
 */
export const regularizerClassNames: RegularizerClassName[] = ['L1L2'];
