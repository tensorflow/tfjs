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

export interface BatchNormalizationLayerConfig extends LayerConfig {
  axis?: number;
  momentum?: number;
  epsilon?: number;
  center?: boolean;
  scale?: boolean;
  beta_initializer?: InitializerSerialization;
  gamma_initializer?: InitializerSerialization;
  moving_mean_initializer?: InitializerSerialization;
  moving_variance_initializer?: InitializerSerialization;
  beta_constraint?: ConstraintSerialization;
  gamma_constraint?: ConstraintSerialization;
  beta_regularizer?: RegularizerSerialization;
  gamma_regularizer?: RegularizerSerialization;
}

// Update batchNormalizationLayerClassNames below in concert with this.
export type BatchNormalizationLayerSerialization =
    BaseLayerSerialization<'BatchNormalization', BatchNormalizationLayerConfig>;

export type NormalizationLayerSerialization =
    BatchNormalizationLayerSerialization;

export type NormalizationLayerClassName =
    NormalizationLayerSerialization['class_name'];

// We can't easily extract a string[] from the string union type, but we can
// recapitulate the list, enforcing at compile time that the values are valid.

/**
 * A string array of valid NormalizationLayer class names.
 *
 * This is guaranteed to match the `NormalizationLayerClassName` union
 * type.
 */
export const normalizationLayerClassNames: NormalizationLayerClassName[] = [
  'BatchNormalization',
];
