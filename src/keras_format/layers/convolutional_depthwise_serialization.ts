/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {ConstraintSerialization} from '../constraint_config';
import {InitializerSerialization} from '../initializer_config';
import {RegularizerSerialization} from '../regularizer_config';
import {BaseConvLayerConfig} from './convolutional_serialization';

export interface DepthwiseConv2DLayerConfig extends BaseConvLayerConfig {
  kernelSize: number|[number, number];
  depthMultiplier?: number;
  depthwiseInitializer?: InitializerSerialization;
  depthwiseConstraint?: ConstraintSerialization;
  depthwiseRegularizer?: RegularizerSerialization;
}

export interface DepthwiseConv2DLayerSerialization {
  class_name: 'DepthwiseConv2D';
  config: DepthwiseConv2DLayerConfig;
}
