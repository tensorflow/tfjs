/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {BaseSerialization} from './types';

// Because of the limitations in the current Keras spec, there is no clear
// definition of what may or may not be the configuration of an optimizer.
//
// For now we'll represent the ones available in TF.js--but it will take more
// thought to get this right in a cross-platform way.
//
// See internal issue: b/121033602

// TODO(soergel): This is a stopgap that needs further thought.
// Does it belong here?
// Does it belong in tfjs-core?
// See also the dormant https://github.com/tensorflow/tfjs-core/pull/1404

export type AdadeltaOptimizerConfig = {
  learning_rate: number; rho: number; epsilon: number;
};

export type AdadeltaSerialization =
    BaseSerialization<'Adadelta', AdadeltaOptimizerConfig>;

export type AdagradOptimizerConfig = {
  learning_rate: number;
  initial_accumulator_value?: number;
};

export type AdagradSerialization =
    BaseSerialization<'Adagrad', AdagradOptimizerConfig>;

export type AdamOptimizerConfig = {
  learning_rate: number; beta1: number; beta2: number;
  epsilon?: number;
};

export type AdamSerialization = BaseSerialization<'Adam', AdamOptimizerConfig>;

export type AdamaxOptimizerConfig = {
  learning_rate: number; beta1: number; beta2: number;
  epsilon?: number;
  decay?: number;
};

export type AdamaxSerialization =
    BaseSerialization<'Adamax', AdamaxOptimizerConfig>;

export type MomentumOptimizerConfig = {
  // extends SGDOptimizerConfig {
  learning_rate: number; momentum: number;
  use_nesterov?: boolean;
};

export type MomentumSerialization =
    BaseSerialization<'Momentum', MomentumOptimizerConfig>;

export type RMSPropOptimizerConfig = {
  learning_rate: number;
  decay?: number;
  momentum?: number;
  epsilon?: number;
  centered?: boolean;
};

export type RMSPropSerialization =
    BaseSerialization<'RMSProp', RMSPropOptimizerConfig>;

export type SGDOptimizerConfig = {
  learning_rate: number;
};

export type SGDSerialization = BaseSerialization<'SGD', SGDOptimizerConfig>;

// Update optimizerClassNames below in concert with this.
export type OptimizerSerialization = AdadeltaSerialization|AdagradSerialization|
    AdamSerialization|AdamaxSerialization|MomentumSerialization|
    RMSPropSerialization|SGDSerialization;

export type OptimizerClassName = OptimizerSerialization['class_name'];

// We can't easily extract a string[] from the string union type, but we can
// recapitulate the list, enforcing at compile time that the values are valid.

/**
 * A string array of valid Optimizer class names.
 *
 * This is guaranteed to match the `OptimizerClassName` union type.
 */
export const optimizerClassNames: OptimizerClassName[] =
    ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Momentum', 'RMSProp', 'SGD'];
