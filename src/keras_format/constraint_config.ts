/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

export interface MaxNormSerialization {
  class_name: 'MaxNorm';
  config: {maxValue?: number; axis?: number;};
}

export interface UnitNormSerialization {
  class_name: 'UnitNorm';
  config: {axis?: number;};
}

export interface NonNegSerialization {
  class_name: 'NonNeg';
}

export interface MinMaxNormSerialization {
  class_name: 'MinMaxNorm';
  config: {minValue?: number; maxValue?: number; axis?: number; rate?: number;};
}

export type ConstraintSerialization = MaxNormSerialization|NonNegSerialization|
    UnitNormSerialization|MinMaxNormSerialization;
