/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {DataFormat, PaddingMode} from '../common';
import {LayerConfig} from '../topology_config';


export interface Pooling1DLayerConfig extends LayerConfig {
  poolSize?: number;
  strides?: number;
  padding?: PaddingMode;
}
export interface Pooling1DLayerSerialization {
  class_name: 'Pooling1D';
  config: Pooling1DLayerConfig;
}

export interface Pooling2DLayerConfig extends LayerConfig {
  poolSize?: number|[number, number];
  strides?: number|[number, number];
  padding?: PaddingMode;
  dataFormat?: DataFormat;
}

export interface Pooling2DLayerSerialization {
  class_name: 'Pooling2D';
  config: Pooling2DLayerConfig;
}

export interface GlobalPooling2DLayerConfig extends LayerConfig {
  dataFormat?: DataFormat;
}

export interface GlobalPooling2DLayerSerialization {
  class_name: 'GlobalPooling2D';
  config: GlobalPooling2DLayerConfig;
}

export type PoolingLayerSerialization = Pooling1DLayerSerialization|
    Pooling2DLayerSerialization|GlobalPooling2DLayerSerialization;
