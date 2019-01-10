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
import {LayerConfig} from '../topology_config';

export interface ReLULayerConfig extends LayerConfig {
  maxValue?: number;
}

export interface ReLULayerSerialization {
  class_name: 'ReLU';
  config: ReLULayerConfig;
}

export interface LeakyReLULayerConfig extends LayerConfig {
  alpha?: number;
}

export interface LeakyReLULayerSerialization {
  class_name: 'LeakyReLU';
  config: LeakyReLULayerConfig;
}

export interface PReLULayerConfig extends LayerConfig {
  alphaInitializer?: InitializerSerialization;
  alphaRegularizer?: RegularizerSerialization;
  alphaConstraint?: ConstraintSerialization;
  sharedAxes?: number|number[];
}

export interface PReLULayerSerialization {
  class_name: 'PReLU';
  config: PReLULayerConfig;
}

export interface ELULayerConfig extends LayerConfig {
  alpha?: number;
}

export interface ELULayerSerialization {
  class_name: 'ELU';
  config: ELULayerConfig;
}

export interface ThresholdedReLULayerConfig extends LayerConfig {
  theta?: number;
}

export interface ThresholdedReLULayerSerialization {
  class_name: 'ThresholdedReLU';
  config: ThresholdedReLULayerConfig;
}

export interface SoftmaxLayerConfig extends LayerConfig {
  axis?: number;
}

export interface SoftmaxLayerSerialization {
  class_name: 'Softmax';
  config: SoftmaxLayerConfig;
}

export type AdvancedActivationLayerSerialization = ReLULayerSerialization|
    LeakyReLULayerSerialization|PReLULayerSerialization|ELULayerSerialization|
    ThresholdedReLULayerSerialization|SoftmaxLayerSerialization;
