/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {LayerConfig} from '../topology_config';

export interface ConcatenateLayerConfig extends LayerConfig {
  axis?: number;
}

export interface ConcatenateLayerSerialization {
  class_name: 'Concatenate';
  config: ConcatenateLayerConfig;
}

export interface DotLayerConfig extends LayerConfig {
  axes: number|[number, number];
  normalize?: boolean;
}

export interface DotLayerSerialization {
  class_name: 'Dot';
  config: DotLayerConfig;
}

export type MergeLayerSerialization =
    ConcatenateLayerSerialization|DotLayerSerialization;
