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

export interface EmbeddingLayerConfig extends LayerConfig {
  input_dim: number;
  output_dim: number;
  embeddings_initializer?: InitializerSerialization;
  embeddings_regularizer?: RegularizerSerialization;
  activity_regularizer?: RegularizerSerialization;
  embeddings_constraint?: ConstraintSerialization;
  mask_zero?: boolean;
  input_length?: number|number[];
}

export type EmbeddingLayerSerialization =
    BaseLayerSerialization<'Embedding', EmbeddingLayerConfig>;
