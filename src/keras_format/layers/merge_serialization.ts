/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {BaseLayerSerialization, LayerConfig} from '../topology_config';

export interface ConcatenateLayerConfig extends LayerConfig {
  axis?: number;
}

export type ConcatenateLayerSerialization =
    BaseLayerSerialization<'Concatenate', ConcatenateLayerConfig>;

export interface DotLayerConfig extends LayerConfig {
  axes: number|[number, number];
  normalize?: boolean;
}

export type DotLayerSerialization =
    BaseLayerSerialization<'Dot', DotLayerConfig>;

export type MergeLayerSerialization =
    ConcatenateLayerSerialization|DotLayerSerialization;
