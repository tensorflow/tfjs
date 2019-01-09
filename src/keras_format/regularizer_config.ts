/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

export interface L1L2Serialization {
  class_name: 'l1_l2';
  config: {l1?: number; l2?: number;};
}

export interface L1Serialization {
  class_name: 'l1';
  config: {l1: number;};
}

export interface L2Serialization {
  class_name: 'l2';
  config: {l2: number;};
}

export type RegularizerSerialization =
    L1L2Serialization|L1Serialization|L2Serialization;
