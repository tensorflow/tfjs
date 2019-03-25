/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {DataType} from '@tensorflow/tfjs-core';
import {Shape} from './common';
import {BaseLayerSerialization} from './topology_config';


export type InputLayerConfig = {
  name?: string;
  input_shape?: Shape;
  batch_size?: number;
  batch_input_shape?: Shape;
  dtype?: DataType;
  sparse?: boolean;
};

// This really should be BaseSerialization because an input layer has no
// inbound_nodes. But, that makes type safety more difficult.

// Update inputLayerClassNames below in concert with this.
export type InputLayerSerialization =
    BaseLayerSerialization<'InputLayer', InputLayerConfig>;

export type InputLayerClassName = InputLayerSerialization['class_name'];

// We can't easily extract a string[] from the string union type, but we can
// recapitulate the list, enforcing at compile time that the values are valid.

/**
 * A string array of valid InputLayer class names.
 *
 * This is guaranteed to match the `InputLayerClassName` union type.
 */
export const inputLayerClassNames: InputLayerClassName[] = [
  'InputLayer',
];
