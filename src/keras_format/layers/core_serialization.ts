/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {ActivationIdentifier} from '../activation_config';
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
  activation?: ActivationIdentifier;
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

export interface ActivationLayerConfig extends LayerConfig {
  activation: ActivationIdentifier;
}

export type ActivationLayerSerialization =
    BaseLayerSerialization<'Activation', ActivationLayerConfig>;

export interface RepeatVectorLayerConfig extends LayerConfig {
  n: number;
}

export type RepeatVectorLayerSerialization =
    BaseLayerSerialization<'RepeatVector', RepeatVectorLayerConfig>;

export interface ReshapeLayerConfig extends LayerConfig {
  targetShape: Shape;
}

export type ReshapeLayerSerialization =
    BaseLayerSerialization<'Reshape', ReshapeLayerConfig>;

export interface PermuteLayerConfig extends LayerConfig {
  dims: number[];
}

export type PermuteLayerSerialization =
    BaseLayerSerialization<'Permute', PermuteLayerConfig>;

export type CoreLayerSerialization =
    DropoutLayerSerialization|DenseLayerSerialization|
    ActivationLayerSerialization|RepeatVectorLayerSerialization|
    ReshapeLayerSerialization|PermuteLayerSerialization;
