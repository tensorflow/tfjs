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
import {LayerConfig} from '../topology_config';

export interface BaseConvLayerConfig extends LayerConfig {
  kernelSize: number|number[];
  strides?: number|number[];
  padding?: PaddingMode;
  dataFormat?: DataFormat;
  dilationRate?: number|[number]|[number, number];
  activation?: string;
  useBias?: boolean;
  kernelInitializer?: InitializerSerialization;
  biasInitializer?: InitializerSerialization;
  kernelConstraint?: ConstraintSerialization;
  biasConstraint?: ConstraintSerialization;
  kernelRegularizer?: RegularizerSerialization;
  biasRegularizer?: RegularizerSerialization;
  activityRegularizer?: RegularizerSerialization;
}

export interface ConvLayerConfig extends BaseConvLayerConfig {
  filters: number;
}

export interface ConvLayerSerialization {
  class_name: 'Conv';
  config: ConvLayerConfig;
}
