/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {DataFormatSerialization} from '../common';
import {BaseLayerSerialization, LayerConfig} from '../topology_config';

export interface ZeroPadding2DLayerConfig extends LayerConfig {
  padding?: number|[number, number]|[[number, number], [number, number]];
  data_format?: DataFormatSerialization;
}

// Update paddingLayerClassNames below in concert with this.
export type ZeroPadding2DLayerSerialization =
    BaseLayerSerialization<'ZeroPadding2D', ZeroPadding2DLayerConfig>;

export type PaddingLayerSerialization = ZeroPadding2DLayerSerialization;

export type PaddingLayerClassName = PaddingLayerSerialization['class_name'];

// We can't easily extract a string[] from the string union type, but we can
// recapitulate the list, enforcing at compile time that the values are valid.

/**
 * A string array of valid PaddingLayer class names.
 *
 * This is guaranteed to match the `PaddingLayerClassName` union type.
 */
export const paddingLayerClassNames: PaddingLayerClassName[] = [
  'ZeroPadding2D',
];
