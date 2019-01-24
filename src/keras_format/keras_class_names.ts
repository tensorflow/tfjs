/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {ConstraintClassName, ConstraintSerialization} from './constraint_config';
import {InitializerClassName, InitializerSerialization} from './initializer_config';
import {LayerClassName, LayerSerialization} from './layers/layer_serialization';
import {OptimizerClassName, OptimizerSerialization} from './optimizer_config';
import {RegularizerClassName, RegularizerSerialization} from './regularizer_config';

/**
 * A type representing all valid values of `class_name` in a Keras JSON file
 * (regardless of context, which will naturally further restrict the valid
 * values).
 */
export type KerasClassName = LayerClassName|ConstraintClassName|
    InitializerClassName|RegularizerClassName|OptimizerClassName;

/**
 * A type representing all possible Serializations of Keras objects, including
 * Layers, Constraints, Optimizers, etc.
 */
export type KerasSerialization = LayerSerialization|ConstraintSerialization|
    InitializerSerialization|RegularizerSerialization|OptimizerSerialization;
