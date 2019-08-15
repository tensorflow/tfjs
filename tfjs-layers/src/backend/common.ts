/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {backend} from '@tensorflow/tfjs-core';
import {DataFormat} from '../keras_format/common';

let _epsilon: number;

/**
 * Returns the value of the fuzz factor used in numeric expressions.
 */
export function epsilon() {
  if (_epsilon == null) {
    _epsilon = backend().epsilon();
  }
  return _epsilon;
}

/**
 * Sets the value of the fuzz factor used in numeric expressions.
 * @param e New value of epsilon.
 */
export function setEpsilon(e: number) {
  _epsilon = e;
}

/**
 * Returns the default image data format convention.
 */
export function imageDataFormat(): DataFormat {
  return 'channelsLast';
}
