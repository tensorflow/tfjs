/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {DataFormatSerialization, PaddingMode} from '../common';
import {BaseLayerSerialization, LayerConfig} from '../topology_config';


export interface Pooling1DLayerConfig extends LayerConfig {
  pool_size?: [number];
  strides?: [number];
  padding?: PaddingMode;
}

export type MaxPooling1DLayerSerialization =
    BaseLayerSerialization<'MaxPooling1D', Pooling1DLayerConfig>;

export type AveragePooling1DLayerSerialization =
    BaseLayerSerialization<'AveragePooling1D', Pooling1DLayerConfig>;

export interface Pooling2DLayerConfig extends LayerConfig {
  pool_size?: number|[number, number];
  strides?: number|[number, number];
  padding?: PaddingMode;
  data_format?: DataFormatSerialization;
}

export type MaxPooling2DLayerSerialization =
    BaseLayerSerialization<'MaxPooling2D', Pooling2DLayerConfig>;

export type AveragePooling2DLayerSerialization =
    BaseLayerSerialization<'AveragePooling2D', Pooling2DLayerConfig>;

export type GlobalAveragePooling1DLayerSerialization =
    BaseLayerSerialization<'GlobalAveragePooling1D', LayerConfig>;

export type GlobalMaxPooling1DLayerSerialization =
    BaseLayerSerialization<'GlobalMaxPooling1D', LayerConfig>;

export interface GlobalPooling2DLayerConfig extends LayerConfig {
  data_format?: DataFormatSerialization;
}

export type GlobalAveragePooling2DLayerSerialization = BaseLayerSerialization<
    'GlobalAveragePooling2D', GlobalPooling2DLayerConfig>;

export type GlobalMaxPooling2DLayerSerialization =
    BaseLayerSerialization<'GlobalMaxPooling2D', GlobalPooling2DLayerConfig>;

export type PoolingLayerSerialization = MaxPooling1DLayerSerialization|
    AveragePooling1DLayerSerialization|MaxPooling2DLayerSerialization|
    AveragePooling2DLayerSerialization|GlobalAveragePooling1DLayerSerialization|
    GlobalMaxPooling1DLayerSerialization|
    GlobalAveragePooling2DLayerSerialization|
    GlobalMaxPooling2DLayerSerialization;
