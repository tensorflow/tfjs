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

/** @docinline */
export type FanMode = 'fanIn'|'fanOut'|'fanAvg';
export const VALID_FAN_MODE_VALUES = ['fanIn', 'fanOut', 'fanAvg'];

/** @docinline */
export type Distribution = 'normal'|'uniform';
export const VALID_DISTRIBUTION_VALUES = ['normal', 'uniform'];

export type ZerosSerialization = BaseSerialization<'Zeros', null>;

export type OnesSerialization = BaseSerialization<'Ones', null>;

export type ConstantConfig = {
  value: number;
};

export type ConstantSerialization =
    BaseSerialization<'Constant', ConstantConfig>;

export type RandomNormalConfig = {
  mean?: number;
  stddev?: number;
  seed?: number;
};

export type RandomNormalSerialization =
    BaseSerialization<'RandomNormal', RandomNormalConfig>;

export type RandomUniformConfig = {
  minval?: number;
  maxval?: number;
  seed?: number;
};

export type RandomUniformSerialization =
    BaseSerialization<'RandomUniform', RandomUniformConfig>;

export type TruncatedNormalConfig = {
  mean?: number;
  stddev?: number;
  seed?: number;
};

export type TruncatedNormalSerialization =
    BaseSerialization<'TruncatedNormal', TruncatedNormalConfig>;

export type VarianceScalingConfig = {
  scale: number;

  mode: FanMode;
  distribution: Distribution;
  seed?: number;
};

export type VarianceScalingSerialization =
    BaseSerialization<'VarianceScaling', VarianceScalingConfig>;

export type OrthogonalConfig = {
  seed?: number;
  gain?: number;
};

export type OrthogonalSerialization =
    BaseSerialization<'Orthogonal', OrthogonalConfig>;

export type IdentityConfig = {
  gain?: number;
};

export type IdentitySerialization =
    BaseSerialization<'Identity', IdentityConfig>;

export type InitializerSerialization =
    ConstantSerialization|RandomUniformSerialization|RandomNormalSerialization|
    TruncatedNormalSerialization|IdentitySerialization|
    VarianceScalingSerialization|OrthogonalSerialization;
