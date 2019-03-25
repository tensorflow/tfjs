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
import {BaseLayerSerialization, LayerConfig} from '../topology_config';
import {LayerSerialization} from './layer_serialization';
import {RecurrentLayerSerialization} from './recurrent_serialization';


export type TimeDistributedLayerSerialization =
    BaseLayerSerialization<'TimeDistributed', TimeDistributedLayerConfig>;

export interface TimeDistributedLayerConfig extends LayerConfig {
  layer: LayerSerialization;
}

export type BidirectionalLayerSerialization =
    BaseLayerSerialization<'Bidirectional', BidirectionalLayerConfig>;

export interface BidirectionalLayerConfig extends LayerConfig {
  layer: RecurrentLayerSerialization;
  merge_mode?: BidirectionalMergeMode;
}

// Update wrapperLayerClassNames below in concert with this.
export type WrapperLayerSerialization =
    TimeDistributedLayerSerialization|BidirectionalLayerSerialization;

export type WrapperLayerClassName = WrapperLayerSerialization['class_name'];

// We can't easily extract a string[] from the string union type, but we can
// recapitulate the list, enforcing at compile time that the values are valid.

/**
 * A string array of valid WrapperLayer class names.
 *
 * This is guaranteed to match the `WrapperLayerClassName` union type.
 */
export const wrapperLayerClassNames: WrapperLayerClassName[] = [
  'Bidirectional',
  'TimeDistributed',
];
