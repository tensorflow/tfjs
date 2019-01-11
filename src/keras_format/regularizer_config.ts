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

export type L1L2Config = {
  l1?: number;
  l2?: number;
};

export type L1L2Serialization = BaseSerialization<'l1_l2', L1L2Config>;

export type L1Config = {
  l1?: number;
};

export type L1Serialization = BaseSerialization<'l1', L1Config>;

export type L2Config = {
  l2?: number;
};

export type L2Serialization = BaseSerialization<'l2', L2Config>;

export type RegularizerSerialization =
    L1L2Serialization|L1Serialization|L2Serialization;
