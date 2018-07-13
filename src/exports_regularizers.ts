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
import {L1Config, L1L2, L1L2Config, L2Config, Regularizer} from './regularizers';

/**
 * @doc {
 *   heading: 'Regularizers',
 *   namespace: 'regularizers',
 *   useDocsFrom: 'L1L2',
 *   configParamIndices: [0]
 * }
 */
export function l1l2(config?: L1L2Config): Regularizer {
  return new L1L2(config);
}

/**
 * @doc {
 *   heading: 'Regularizers',
 *   namespace: 'regularizers',
 *   useDocsFrom: 'L1L2',
 *   configParamIndices: [0]
 * }
 */
export function l1(config?: L1Config): Regularizer {
  return regularizers.l1(config);
}

/**
 * @doc {
 *   heading: 'Regularizers',
 *   namespace: 'regularizers',
 *   useDocsFrom: 'L1L2',
 *   configParamIndices: [0]
 * }
 */
export function l2(config?: L2Config): Regularizer {
  return regularizers.l2(config);
}
