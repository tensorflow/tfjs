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

export interface BatchNormalizationLayerConfig extends LayerConfig {
  axis?: number;
  momentum?: number;
  epsilon?: number;
  center?: boolean;
  scale?: boolean;
  betaInitializer?: InitializerSerialization;
  gammaInitializer?: InitializerSerialization;
  movingMeanInitializer?: InitializerSerialization;
  movingVarianceInitializer?: InitializerSerialization;
  betaConstraint?: ConstraintSerialization;
  gammaConstraint?: ConstraintSerialization;
  betaRegularizer?: RegularizerSerialization;
  gammaRegularizer?: RegularizerSerialization;
}

export interface BatchNormalizationLayerSerialization {
  class_name: 'BatchNormalization';
  config: BatchNormalizationLayerConfig;
}
