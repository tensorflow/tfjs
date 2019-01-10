/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {ActivationIdentifier} from './activation_config';
import {ConstraintSerialization} from './constraint_config';
import {InitializerSerialization} from './initializer_config';
import {RegularizerSerialization} from './regularizer_config';
import {LayerConfig} from './topology_config';
import {Shape} from './types';

export interface DropoutLayerConfig extends LayerConfig {
  rate: number;
  noiseShape?: number[];
  seed?: number;
}

export interface DropoutLayerSerialization {
  class_name: 'Dropout';
  config: DropoutLayerConfig;
}

export interface DenseLayerConfig extends LayerConfig {
  units: number;
  activation?: ActivationIdentifier;
  useBias?: boolean;
  inputDim?: number;
  kernelInitializer?: InitializerSerialization;
  biasInitializer?: InitializerSerialization;
  kernelConstraint?: ConstraintSerialization;
  biasConstraint?: ConstraintSerialization;
  kernelRegularizer?: RegularizerSerialization;
  biasRegularizer?: RegularizerSerialization;
  activityRegularizer?: RegularizerSerialization;
}

export interface DenseLayerSerialization {
  class_name: 'Dense';
  config: DenseLayerConfig;
}

export interface ActivationLayerConfig extends LayerConfig {
  activation: ActivationIdentifier;
}

export interface ActivationLayerSerialization {
  class_name: 'Activation';
  config: ActivationLayerConfig;
}

export interface RepeatVectorLayerConfig extends LayerConfig {
  n: number;
}

export interface RepeatVectorLayerSerialization {
  class_name: 'RepeatVector';
  config: RepeatVectorLayerConfig;
}

export interface ReshapeLayerConfig extends LayerConfig {
  targetShape: Shape;
}

export interface ReshapeLayerSerialization {
  class_name: 'Reshape';
  config: ReshapeLayerConfig;
}

export interface PermuteLayerConfig extends LayerConfig {
  dims: number[];
}

export interface PermuteLayerSerialization {
  class_name: 'Permute';
  config: PermuteLayerConfig;
}
