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
import {NodeConfig} from './node_config';
import {BaseSerialization, PyJsonDict} from './types';

/** Constructor arguments for Layer. */
export interface LayerConfig extends PyJsonDict {
  input_shape?: Shape;
  batch_input_shape?: Shape;
  batch_size?: number;
  dtype?: DataType;
  name?: string;
  trainable?: boolean;
  updatable?: boolean;
  input_dtype?: DataType;
}

/**
 * A Keras JSON entry representing a layer.
 *
 * The Keras JSON convention is to provide the `class_name` (i.e., the layer
 * type) at the top level, and then to place the layer-specific configuration in
 * a `config` subtree.  These layer-specific configurations are provided by
 * subtypes of `LayerConfig`.  Thus, this `*Serialization` has a type parameter
 * giving the specific type of the wrapped `LayerConfig`.
 */
export interface BaseLayerSerialization<N extends string, T extends LayerConfig>
    extends BaseSerialization<N, T> {
  name: string;
  inbound_nodes?: NodeConfig[];
}
