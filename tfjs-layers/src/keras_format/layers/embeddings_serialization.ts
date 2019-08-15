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

// Update embeddingLayerClassNames below in concert with this.
export type EmbeddingLayerSerialization =
    BaseLayerSerialization<'Embedding', EmbeddingLayerConfig>;

export type EmbeddingLayerClassName = EmbeddingLayerSerialization['class_name'];

// We can't easily extract a string[] from the string union type, but we can
// recapitulate the list, enforcing at compile time that the values are valid.

/**
 * A string array of valid EmbeddingLayer class names.
 *
 * This is guaranteed to match the `EmbeddingLayerClassName` union type.
 */
export const embeddingLayerClassNames: EmbeddingLayerClassName[] = [
  'Embedding',
];
