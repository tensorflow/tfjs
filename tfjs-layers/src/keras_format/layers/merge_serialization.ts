/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {BaseLayerSerialization, LayerConfig} from '../topology_config';



export type AddLayerSerialization = BaseLayerSerialization<'Add', LayerConfig>;

export type MultiplyLayerSerialization =
    BaseLayerSerialization<'Multiply', LayerConfig>;

export type AverageLayerSerialization =
    BaseLayerSerialization<'Average', LayerConfig>;

export type MaximumLayerSerialization =
    BaseLayerSerialization<'Maximum', LayerConfig>;

export type MinimumLayerSerialization =
    BaseLayerSerialization<'Minimum', LayerConfig>;

export interface ConcatenateLayerConfig extends LayerConfig {
  axis?: number;
}

export type ConcatenateLayerSerialization =
    BaseLayerSerialization<'Concatenate', ConcatenateLayerConfig>;

export interface DotLayerConfig extends LayerConfig {
  axes: number|[number, number];
  normalize?: boolean;
}

export type DotLayerSerialization =
    BaseLayerSerialization<'Dot', DotLayerConfig>;

// Update mergeLayerClassNames below in concert with this.
export type MergeLayerSerialization =
    AddLayerSerialization|MultiplyLayerSerialization|AverageLayerSerialization|
    MaximumLayerSerialization|MinimumLayerSerialization|
    ConcatenateLayerSerialization|DotLayerSerialization;

export type MergeLayerClassName = MergeLayerSerialization['class_name'];

// We can't easily extract a string[] from the string union type, but we can
// recapitulate the list, enforcing at compile time that the values are valid.

/**
 * A string array of valid MergeLayer class names.
 *
 * This is guaranteed to match the `MergeLayerClassName` union type.
 */
export const mergeLayerClassNames: MergeLayerClassName[] = [
  'Add',
  'Average',
  'Concatenate',
  'Dot',
  'Maximum',
  'Minimum',
  'Multiply',
];
