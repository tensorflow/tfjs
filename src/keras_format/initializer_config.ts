/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */


/** @docinline */
export type FanMode = 'fanIn'|'fanOut'|'fanAvg';
export const VALID_FAN_MODE_VALUES = ['fanIn', 'fanOut', 'fanAvg'];

/** @docinline */
export type Distribution = 'normal'|'uniform';
export const VALID_DISTRIBUTION_VALUES = ['normal', 'uniform'];

export interface ZerosSerialization {
  class_name: 'Zeros';
}

export interface OnesSerialization {
  class_name: 'Ones';
}

export interface ConstantSerialization {
  class_name: 'Constant';
  config: {

    value: number;
  };
}

export interface RandomNormalSerialization {
  class_name: 'RandomNormal';
  config: {

    mean?: number;
    stddev?: number;
    seed?: number;
  };
}

export interface RandomUniformSerialization {
  class_name: 'RandomUniform';
  config: {

    minval?: number;
    maxval?: number;
    seed?: number;
  };
}

export interface TruncatedNormalSerialization {
  class_name: 'TruncatedNormal';
  config: {

    mean?: number;
    stddev?: number;
    seed?: number;
  };
}

export interface VarianceScalingSerialization {
  class_name: 'VarianceScaling';
  config: {

    scale: number;

    mode: FanMode;
    distribution: Distribution;
    seed?: number;
  };
}

export interface OrthogonalSerialization {
  class_name: 'Orthogonal';
  config: {

    seed?: number;
    gain?: number;
  };
}

export interface IdentitySerialization {
  class_name: 'Identity';
  config: {

    gain?: number;
  };
}

export type InitializerSerialization =
    ConstantSerialization|RandomUniformSerialization|RandomNormalSerialization|
    TruncatedNormalSerialization|IdentitySerialization|
    VarianceScalingSerialization|OrthogonalSerialization;
