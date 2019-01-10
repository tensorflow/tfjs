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
import {ConstraintSerialization} from '../constraint_config';
import {InitializerSerialization} from '../initializer_config';
import {RegularizerSerialization} from '../regularizer_config';
import {LayerConfig} from '../topology_config';

export interface BaseRNNLayerConfig extends LayerConfig {
  cell?: RNNCellSerialization|RNNCellSerialization[];
  returnSequences?: boolean;
  returnState?: boolean;
  goBackwards?: boolean;
  stateful?: boolean;
  unroll?: boolean;
  inputDim?: number;
  inputLength?: number;
}

export interface SimpleRNNCellConfig extends LayerConfig {
  units: number;
  activation?: ActivationIdentifier;
  useBias?: boolean;
  kernelInitializer?: InitializerSerialization;
  recurrentInitializer?: InitializerSerialization;
  biasInitializer?: InitializerSerialization;
  kernelRegularizer?: RegularizerSerialization;
  recurrentRegularizer?: RegularizerSerialization;
  biasRegularizer?: RegularizerSerialization;
  kernelConstraint?: ConstraintSerialization;
  recurrentConstraint?: ConstraintSerialization;
  biasConstraint?: ConstraintSerialization;
  dropout?: number;
  recurrentDropout?: number;
}

export interface SimpleRNNCellSerialization {
  class_name: 'SimpleRNNCell';
  config: SimpleRNNCellConfig;
}

export interface SimpleRNNLayerConfig extends BaseRNNLayerConfig {
  units: number;
  activation?: ActivationIdentifier;
  useBias?: boolean;
  kernelInitializer?: InitializerSerialization;
  recurrentInitializer?: InitializerSerialization;
  biasInitializer?: InitializerSerialization;
  kernelRegularizer?: RegularizerSerialization;
  recurrentRegularizer?: RegularizerSerialization;
  biasRegularizer?: RegularizerSerialization;
  kernelConstraint?: ConstraintSerialization;
  recurrentConstraint?: ConstraintSerialization;
  biasConstraint?: ConstraintSerialization;
  dropout?: number;
  recurrentDropout?: number;
}

export interface SimpleRNNLayerSerialization {
  class_name: 'SimpleRNN';
  config: SimpleRNNLayerConfig;
}

export interface GRUCellConfig extends SimpleRNNCellConfig {
  recurrentActivation?: string;
  implementation?: number;
}

export interface GRUCellSerialization {
  class_name: 'GRUCell';
  config: GRUCellConfig;
}

export interface GRULayerConfig extends SimpleRNNLayerConfig {
  recurrentActivation?: string;
  implementation?: number;
}

export interface GRULayerSerialization {
  class_name: 'GRU';
  config: GRULayerConfig;
}

export interface LSTMCellConfig extends SimpleRNNCellConfig {
  recurrentActivation?: ActivationIdentifier;
  unitForgetBias?: boolean;
  implementation?: number;
}

export interface LSTMCellSerialization {
  class_name: 'LSTMCell';
  config: LSTMCellConfig;
}

export interface LSTMLayerConfig extends SimpleRNNLayerConfig {
  recurrentActivation?: string;
  unitForgetBias?: boolean;
  implementation?: number;
}
export interface LSTMLayerSerialization {
  class_name: 'LSTM';
  config: LSTMLayerConfig;
}

export interface StackedRNNCellsConfig extends LayerConfig {
  // TODO(soergel): consider whether we can avoid improperly mixing
  // Simple / LSTM / GRU cells here and in the above Layer serializations.
  cells: RNNCellSerialization[];
}

export interface StackedRNNCellsSerialization {
  class_name: 'StackedRNNCells';
  config: StackedRNNCellsConfig;
}

export type RNNCellSerialization = SimpleRNNCellSerialization|
    GRUCellSerialization|LSTMCellSerialization|StackedRNNCellsSerialization;

export type RecurrentLayerSerialization =
    SimpleRNNLayerSerialization|LSTMLayerSerialization|GRULayerSerialization;
