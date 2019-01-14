/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {ConstraintClassName} from './constraint_config';
import {InitializerClassName} from './initializer_config';
import {LayerClassName} from './layers/layer_serialization';
import {RegularizerClassName} from './regularizer_config';

/**
 * A type representing all valid values of `class_name` in a Keras JSON file
 * (regardless of context, which will naturally further restrict the valid
 * values).
 */
export type KerasClassName = LayerClassName|ConstraintClassName|
    InitializerClassName|RegularizerClassName;
