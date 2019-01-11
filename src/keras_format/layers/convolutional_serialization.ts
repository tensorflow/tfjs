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
import {ConstraintSerialization} from '../constraint_config';
import {InitializerSerialization} from '../initializer_config';
import {RegularizerSerialization} from '../regularizer_config';
import {BaseLayerSerialization, LayerConfig} from '../topology_config';

export interface BaseConvLayerConfig extends LayerConfig {
  kernel_size: number|number[];
  strides?: number|number[];
  padding?: PaddingMode;
  data_format?: DataFormat;
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

export type ConvLayerSerialization =
    BaseLayerSerialization<'Conv', ConvLayerConfig>;
