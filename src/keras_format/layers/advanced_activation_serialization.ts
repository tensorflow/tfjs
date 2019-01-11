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
import {BaseLayerSerialization, LayerConfig} from '../topology_config';

export interface ReLULayerConfig extends LayerConfig {
  max_value?: number;
}

export type ReLULayerSerialization =
    BaseLayerSerialization<'ReLU', ReLULayerConfig>;

export interface LeakyReLULayerConfig extends LayerConfig {
  alpha?: number;
}

export type LeakyReLULayerSerialization =
    BaseLayerSerialization<'LeakyReLU', LeakyReLULayerConfig>;

export interface PReLULayerConfig extends LayerConfig {
  alpha_initializer?: InitializerSerialization;
  alpha_regularizer?: RegularizerSerialization;
  alpha_constraint?: ConstraintSerialization;
  shared_axes?: number|number[];
}

export type PReLULayerSerialization =
    BaseLayerSerialization<'PReLU', PReLULayerConfig>;

export interface ELULayerConfig extends LayerConfig {
  alpha?: number;
}

export type ELULayerSerialization =
    BaseLayerSerialization<'ELU', ELULayerConfig>;

export interface ThresholdedReLULayerConfig extends LayerConfig {
  theta?: number;
}

export type ThresholdedReLULayerSerialization =
    BaseLayerSerialization<'ThresholdedReLU', ThresholdedReLULayerConfig>;

export interface SoftmaxLayerConfig extends LayerConfig {
  axis?: number;
}

export type SoftmaxLayerSerialization =
    BaseLayerSerialization<'Softmax', SoftmaxLayerConfig>;

export type AdvancedActivationLayerSerialization = ReLULayerSerialization|
    LeakyReLULayerSerialization|PReLULayerSerialization|ELULayerSerialization|
    ThresholdedReLULayerSerialization|SoftmaxLayerSerialization;
