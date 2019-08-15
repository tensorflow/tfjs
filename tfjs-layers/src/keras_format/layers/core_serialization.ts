/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {ActivationSerialization} from '../activation_config';
import {Shape} from '../common';
import {ConstraintSerialization} from '../constraint_config';
import {InitializerSerialization} from '../initializer_config';
import {RegularizerSerialization} from '../regularizer_config';
import {BaseLayerSerialization, LayerConfig} from '../topology_config';

export interface DropoutLayerConfig extends LayerConfig {
  rate: number;
  noise_shape?: number[];
  seed?: number;
}

export type DropoutLayerSerialization =
    BaseLayerSerialization<'Dropout', DropoutLayerConfig>;

export interface DenseLayerConfig extends LayerConfig {
  units: number;
  activation?: ActivationSerialization;
  use_bias?: boolean;
  input_dim?: number;
  kernel_initializer?: InitializerSerialization;
  bias_initializer?: InitializerSerialization;
  kernel_constraint?: ConstraintSerialization;
  bias_constraint?: ConstraintSerialization;
  kernel_regularizer?: RegularizerSerialization;
  bias_regularizer?: RegularizerSerialization;
  activity_regularizer?: RegularizerSerialization;
}

export type DenseLayerSerialization =
    BaseLayerSerialization<'Dense', DenseLayerConfig>;

export type FlattenLayerSerialization =
    BaseLayerSerialization<'Flatten', LayerConfig>;

export interface ActivationLayerConfig extends LayerConfig {
  activation: ActivationSerialization;
}

export type ActivationLayerSerialization =
    BaseLayerSerialization<'Activation', ActivationLayerConfig>;

export interface RepeatVectorLayerConfig extends LayerConfig {
  n: number;
}

export type RepeatVectorLayerSerialization =
    BaseLayerSerialization<'RepeatVector', RepeatVectorLayerConfig>;

export interface ReshapeLayerConfig extends LayerConfig {
  target_shape: Shape;
}

export type ReshapeLayerSerialization =
    BaseLayerSerialization<'Reshape', ReshapeLayerConfig>;

export interface PermuteLayerConfig extends LayerConfig {
  dims: number[];
}

export type PermuteLayerSerialization =
    BaseLayerSerialization<'Permute', PermuteLayerConfig>;

export interface MaskingLayerConfig extends LayerConfig {
  maskValue: number;
}

export type MaskingLayerSerialization =
    BaseLayerSerialization<'Masking', MaskingLayerConfig>;

// Update coreLayerClassNames below in concert with this.
export type CoreLayerSerialization =
    DropoutLayerSerialization|DenseLayerSerialization|FlattenLayerSerialization|
    ActivationLayerSerialization|RepeatVectorLayerSerialization|
    ReshapeLayerSerialization|PermuteLayerSerialization|
    MaskingLayerSerialization;

export type CoreLayerClassName = CoreLayerSerialization['class_name'];

// We can't easily extract a string[] from the string union type, but we can
// recapitulate the list, enforcing at compile time that the values are valid.

/**
 * A string array of valid CoreLayer class names.
 *
 * This is guaranteed to match the `CoreLayerClassName` union type.
 */
export const coreLayerClassNames: CoreLayerClassName[] = [
  'Activation',
  'Dense',
  'Dropout',
  'Flatten',
  'Permute',
  'RepeatVector',
  'Reshape',
];
