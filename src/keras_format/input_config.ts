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

import {Shape} from './common';
import {BaseLayerSerialization} from './topology_config';

export type InputLayerConfig = {
  name: string;
  input_shape?: Shape;
  batch_size?: number;
  batch_input_shape?: Shape;
  dtype?: DataType;
  sparse?: boolean;
};

// This really should be BaseSerialization because an input layer has no
// inbound_nodes. But, that makes type safety more difficult.
export type InputLayerSerialization =
    BaseLayerSerialization<'InputLayer', InputLayerConfig>;
