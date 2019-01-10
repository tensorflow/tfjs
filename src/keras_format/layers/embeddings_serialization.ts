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

export interface EmbeddingLayerConfig extends LayerConfig {
  inputDim: number;
  outputDim: number;
  embeddingsInitializer?: InitializerSerialization;
  embeddingsRegularizer?: RegularizerSerialization;
  activityRegularizer?: RegularizerSerialization;
  embeddingsConstraint?: ConstraintSerialization;
  maskZero?: boolean;
  inputLength?: number|number[];
}

export interface EmbeddingLayerSerialization {
  class_name: 'Embedding';
  config: EmbeddingLayerConfig;
}
