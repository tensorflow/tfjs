/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {DataFormat} from '../common';
import {LayerConfig} from '../topology_config';

export interface ZeroPadding2DLayerConfig extends LayerConfig {
  padding?: number|[number, number]|[[number, number], [number, number]];
  dataFormat?: DataFormat;
}

export interface ZeroPadding2DLayerSerialization {
  class_name: 'ZeroPadding2D';
  config: ZeroPadding2DLayerConfig;
}
