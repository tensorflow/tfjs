/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {DataType} from '@tensorflow/tfjs-core';

import {Shape} from './types';

/** Constructor arguments for Layer. */
export interface LayerConfig {
  inputShape?: Shape;
  batchInputShape?: Shape;
  batchSize?: number;
  dtype?: DataType;
  name?: string;
  trainable?: boolean;
  updatable?: boolean;
  inputDType?: DataType;
}
