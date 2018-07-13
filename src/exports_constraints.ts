/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
// tslint:disable-next-line:max-line-length
import {Constraint, MaxNorm, MaxNormConfig, MinMaxNorm, MinMaxNormConfig, NonNeg, UnitNorm, UnitNormConfig} from './constraints';

/**
 * @doc {
 *   heading: 'Constraints',
 *   namespace: 'constraints',
 *   useDocsFrom: 'MaxNorm',
 *   configParamIndices: [0]
 * }
 */
export function maxNorm(config: MaxNormConfig): Constraint {
  return new MaxNorm(config);
}

/**
 * @doc {
 *   heading: 'Constraints',
 *   namespace: 'constraints',
 *   useDocsFrom: 'UnitNorm',
 *   configParamIndices: [0]
 * }
 */
export function unitNorm(config: UnitNormConfig): Constraint {
  return new UnitNorm(config);
}

/**
 * @doc {
 *   heading: 'Constraints',
 *   namespace: 'constraints',
 *   useDocsFrom: 'NonNeg'
 * }
 */
export function nonNeg(): Constraint {
  return new NonNeg();
}

/**
 * @doc {
 *   heading: 'Constraints',
 *   namespace: 'constraints',
 *   useDocsFrom: 'MinMaxNormConfig',
 *   configParamIndices: [0]
 * }
 */
export function minMaxNorm(config: MinMaxNormConfig): Constraint {
  return new MinMaxNorm(config);
}
