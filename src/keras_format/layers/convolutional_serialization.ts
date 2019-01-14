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
import {ConstraintSerialization} from '../constraint_config';
import {InitializerSerialization} from '../initializer_config';
import {RegularizerSerialization} from '../regularizer_config';
import {BaseLayerSerialization, LayerConfig} from '../topology_config';

export interface BaseConvLayerConfig extends LayerConfig {
  kernel_size: number|number[];
  strides?: number|number[];
  padding?: PaddingMode;
  data_format?: DataFormatSerialization;
  dilation_rate?: number|[number]|[number, number];
  activation?: string;
  use_bias?: boolean;
  kernel_initializer?: InitializerSerialization;
  bias_initializer?: InitializerSerialization;
  kernel_constraint?: ConstraintSerialization;
  bias_constraint?: ConstraintSerialization;
  kernel_regularizer?: RegularizerSerialization;
  bias_regularizer?: RegularizerSerialization;
  activity_regularizer?: RegularizerSerialization;
}

export interface ConvLayerConfig extends BaseConvLayerConfig {
  filters: number;
}

export type Conv1DLayerSerialization =
    BaseLayerSerialization<'Conv1D', ConvLayerConfig>;

export type Conv2DLayerSerialization =
    BaseLayerSerialization<'Conv2D', ConvLayerConfig>;

export type Conv2DTransposeLayerSerialization =
    BaseLayerSerialization<'Conv2DTranspose', ConvLayerConfig>;

export interface SeparableConvLayerConfig extends ConvLayerConfig {
  depthMultiplier?: number;
  depthwise_initializer?: InitializerSerialization;
  pointwise_initializer?: InitializerSerialization;
  depthwise_regularizer?: RegularizerSerialization;
  pointwise_regularizer?: RegularizerSerialization;
  depthwise_constraint?: ConstraintSerialization;
  pointwise_constraint?: ConstraintSerialization;
}

export type SeparableConv2DLayerSerialization =
    BaseLayerSerialization<'SeparableConv2D', ConvLayerConfig>;


export interface Cropping2DLayerConfig extends LayerConfig {
  cropping: number|[number, number]|[[number, number], [number, number]];
  dataFormat?: DataFormatSerialization;
}

export type Cropping2DLayerSerialization =
    BaseLayerSerialization<'Cropping2D', Cropping2DLayerConfig>;

export interface UpSampling2DLayerConfig extends LayerConfig {
  size?: number[];
  dataFormat?: DataFormatSerialization;
}

export type UpSampling2DLayerSerialization =
    BaseLayerSerialization<'UpSampling2D', UpSampling2DLayerConfig>;

export type ConvolutionalSerialization =
    Conv1DLayerSerialization|Conv2DLayerSerialization|
    Conv2DTransposeLayerSerialization|SeparableConv2DLayerSerialization|
    Cropping2DLayerSerialization|UpSampling2DLayerSerialization;
