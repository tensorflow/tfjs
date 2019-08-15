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

// Update advancedActivationLayerClassNames below in concert with this.
export type AdvancedActivationLayerSerialization = ReLULayerSerialization|
    LeakyReLULayerSerialization|PReLULayerSerialization|ELULayerSerialization|
    ThresholdedReLULayerSerialization|SoftmaxLayerSerialization;

export type AdvancedActivationLayerClassName =
    AdvancedActivationLayerSerialization['class_name'];

// We can't easily extract a string[] from the string union type, but we can
// recapitulate the list, enforcing at compile time that the values are valid.

/**
 * A string array of valid AdvancedActivationLayer class names.
 *
 * This is guaranteed to match the `AdvancedActivationLayerClassName` union
 * type.
 */
export const advancedActivationLayerClassNames:
    AdvancedActivationLayerClassName[] = [
      'ReLU',
      'LeakyReLU',
      'PReLU',
      'ELU',
      'ThresholdedReLU',
      'Softmax',
    ];
