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
import {BaseLayerSerialization, LayerConfig} from '../topology_config';


export interface Pooling1DLayerConfig extends LayerConfig {
  pool_size?: number;
  strides?: number;
  padding?: PaddingMode;
}
export type Pooling1DLayerSerialization =
    BaseLayerSerialization<'Pooling1D', Pooling1DLayerConfig>;

export interface Pooling2DLayerConfig extends LayerConfig {
  pool_size?: number|[number, number];
  strides?: number|[number, number];
  padding?: PaddingMode;
  data_format?: DataFormat;
}

export type Pooling2DLayerSerialization =
    BaseLayerSerialization<'Pooling2D', Pooling2DLayerConfig>;

export interface GlobalPooling2DLayerConfig extends LayerConfig {
  data_format?: DataFormat;
}

export type GlobalPooling2DLayerSerialization =
    BaseLayerSerialization<'GlobalPooling2D', GlobalPooling2DLayerConfig>;

export type PoolingLayerSerialization = Pooling1DLayerSerialization|
    Pooling2DLayerSerialization|GlobalPooling2DLayerSerialization;
