/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {BidirectionalMergeMode} from '../common';
import {LayerConfig} from '../topology_config';

import {LayerSerialization} from './layer_serialization';
import {RecurrentLayerSerialization} from './recurrent_serialization';

export interface TimeDistributedLayerSerialization {
  class_name: 'TimeDistributed';
  config: TimeDistributedLayerConfig;
}

export interface TimeDistributedLayerConfig extends LayerConfig {
  layer: LayerSerialization;
}

export interface BidirectionalLayerSerialization {
  class_name: 'Bidirectional';
  config: BidirectionalLayerConfig;
}

export interface BidirectionalLayerConfig extends LayerConfig {
  layer: RecurrentLayerSerialization;
  mergeMode?: BidirectionalMergeMode;
}
