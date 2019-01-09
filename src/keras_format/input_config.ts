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

export interface InputLayerSerialization {
  class_name: 'Input';
  config: {
    inputShape?: Shape;
    batchSize?: number;
    batchInputShape?: Shape;
    dtype?: DataType;
    sparse?: boolean;
    name?: string;
  };
}
