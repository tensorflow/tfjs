/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

// tslint:disable:max-line-length
import {doc, scalar, Scalar, Tensor, Tensor2D} from '@tensorflow/tfjs-core';

import * as K from './backend/tfjs_backend';
import {checkDataFormat, DataFormat} from './common';
import {NotImplementedError, ValueError} from './errors';
import {DType, Shape} from './types';
import {ConfigDict, ConfigDictValue} from './types';
import {ClassNameMap, Constructor, deserializeKerasObject, SerializableEnumRegistry, serializeKerasObject} from './utils/generic_utils';
import {arrayProd} from './utils/math_utils';

// tslint:enable:max-line-length

/** @docinline */
export type FanMode = 'fanIn'|'fanOut'|'fanAvg';
SerializableEnumRegistry.register(
    'mode', {'fan_in': 'fanIn', 'fan_out': 'fanOut', 'fan_avg': 'fanAvg'});
export const VALID_FAN_MODE_VALUES =
    ['fanIn', 'fanOut', 'fanAvg', undefined, null];
export function checkFanMode(value?: string): void {
  if (value == null) {
    return;
  }
  if (VALID_FAN_MODE_VALUES.indexOf(value) < 0) {
    throw new ValueError(`${value} is not a valid FanMode.  Valid values as ${
        VALID_FAN_MODE_VALUES}`);
  }
}

/** @docinline */
export type Distribution = 'normal'|'uniform';
SerializableEnumRegistry.register(
    'distribution', {'normal': 'normal', 'uniform': 'uniform'});
export const VALID_DISTRIBUTION_VALUES = ['normal', 'uniform', undefined, null];
export function checkDistribution(value?: string): void {
  if (value == null) {
    return;
  }
  if (VALID_DISTRIBUTION_VALUES.indexOf(value) < 0) {
    throw new ValueError(
        `${value} is not a valid Distribution.  Valid values as ${
            VALID_DISTRIBUTION_VALUES}`);
  }
}

/**
 * Initializer base class.
 */
@doc(
    {heading: 'Initializers', subheading: 'Classes', namespace: 'initializers'})
export abstract class Initializer {
  static fromConfig<T>(cls: Constructor<T>, config: ConfigDict): T {
    return new cls(config);
  }

  public fromConfigUsesCustomObjects(): boolean {
    return false;
  }
  /**
   * Generate an initial value.
   * @param shape
   * @param dtype
   * @return The init value.
   */
  abstract apply(shape: Shape, dtype?: DType): Tensor;

  getConfig(): ConfigDict {
    return {};
  }
}

/**
 * Initializer that generates tensors initialized to 0.
 */
export class Zeros extends Initializer {
  apply(shape: Shape, dtype?: DType): Tensor {
    return K.zeros(shape, dtype);
  }
}
ClassNameMap.register('Zeros', Zeros);

/**
 * Initializer that generates tensors initialized to 1.
 */
export class Ones extends Initializer {
  apply(shape: Shape, dtype?: DType): Tensor {
    return K.ones(shape, dtype);
  }
}
ClassNameMap.register('Ones', Ones);

export interface ConstantConfig {
  /** The value for each element in the variable. */
  value: number;
}

/**
 * Initializer that generates values initialized to some constant.
 */
export class Constant extends Initializer {
  private value: number;

  constructor(config: ConstantConfig) {
    super();
    this.value = config.value;
  }

  apply(shape: Shape, dtype?: DType): Tensor {
    return K.scalarTimesArray(scalar(this.value), K.ones(shape, dtype));
  }

  getConfig(): ConfigDict {
    return {
      value: this.value,
    };
  }
}
ClassNameMap.register('Constant', Constant);

export interface RandomUniformConfig {
  /** Lower bound of the range of random values to generate. */
  minval?: number;
  /** Upper bound of the range of random values to generate. */
  maxval?: number;
  /** Used to seed the random generator. */
  seed?: number;
}

/**
 * Initializer that generates random values initialized to a uniform
 * distribution.
 *
 * Values will be distributed uniformly between the configured minval and
 * maxval.
 */
export class RandomUniform extends Initializer {
  readonly DEFAULT_MINVAL = -0.05;
  readonly DEFAULT_MAXVAL = 0.05;
  private minval: number;
  private maxval: number;
  private seed: number;

  constructor(config: RandomUniformConfig) {
    super();
    this.minval = config.minval || this.DEFAULT_MINVAL;
    this.maxval = config.maxval || this.DEFAULT_MAXVAL;
    this.seed = config.seed;
  }

  apply(shape: Shape, dtype?: DType): Tensor {
    return K.randomUniform(shape, this.minval, this.maxval, dtype, this.seed);
  }

  getConfig(): ConfigDict {
    return {minval: this.minval, maxval: this.maxval, seed: this.seed};
  }
}
ClassNameMap.register('RandomUniform', RandomUniform);

export interface RandomNormalConfig {
  /** Mean of the random values to generate. */
  mean?: number;
  /** Standard deviation of the random values to generate. */
  stddev?: number;
  /** Used to seed the random generator. */
  seed?: number;
}

/**
 * Initializer that generates random values initialized to a normal
 * distribution.
 */
export class RandomNormal extends Initializer {
  readonly DEFAULT_MEAN = 0.;
  readonly DEFAULT_STDDEV = 0.05;
  private mean: number;
  private stddev: number;
  private seed: number;

  constructor(config: RandomNormalConfig) {
    super();
    this.mean = config.mean || this.DEFAULT_MEAN;
    this.stddev = config.stddev || this.DEFAULT_STDDEV;
    this.seed = config.seed;
  }

  apply(shape: Shape, dtype?: DType): Tensor {
    return K.randomNormal(shape, this.mean, this.stddev, dtype, this.seed);
  }

  getConfig(): ConfigDict {
    return {mean: this.mean, stddev: this.stddev, seed: this.seed};
  }
}
ClassNameMap.register('RandomNormal', RandomNormal);

export interface TruncatedNormalConfig {
  /** Mean of the random values to generate. */
  mean?: number;
  /** Standard deviation of the random values to generate. */
  stddev?: number;
  /** Used to seed the random generator. */
  seed?: number;
}

/**
 * Initializer that generates random values initialized to a truncated normal.
 * distribution.
 *
 * These values are similar to values from a `RandomNormal` except that values
 * more than two standard deviations from the mean are discarded and re-drawn.
 * This is the recommended initializer for neural network weights and filters.
 */
export class TruncatedNormal extends Initializer {
  readonly DEFAULT_MEAN = 0.;
  readonly DEFAULT_STDDEV = 0.05;
  private mean: number;
  private stddev: number;
  private seed: number;

  constructor(config: TruncatedNormalConfig) {
    super();
    this.mean = config.mean || this.DEFAULT_MEAN;
    this.stddev = config.stddev || this.DEFAULT_STDDEV;
    this.seed = config.seed;
  }

  apply(shape: Shape, dtype?: DType): Tensor {
    return K.truncatedNormal(shape, this.mean, this.stddev, dtype, this.seed);
  }

  getConfig(): ConfigDict {
    return {mean: this.mean, stddev: this.stddev, seed: this.seed};
  }
}
ClassNameMap.register('TruncatedNormal', TruncatedNormal);

export interface IdentityConfig {
  /**
   * Multiplicative factor to apply to the identity matrix.
   */
  gain?: number;
}

/**
 * Initializer that generates the identity matrix.
 * Only use for square 2D matrices.
 */
export class Identity extends Initializer {
  private gain: Scalar;
  constructor(config: IdentityConfig) {
    super();
    this.gain = config.gain != null ? scalar(config.gain) : K.getScalar(1.0);
  }
  apply(shape: Shape, dtype?: DType): Tensor {
    if (shape.length !== 2 || shape[0] !== shape[1]) {
      throw new ValueError(
          'Identity matrix initializer can only be used for' +
          ' 2D square matrices.');
    } else {
      return K.scalarTimesArray(this.gain, K.eye(shape[0]));
    }
  }
  getConfig(): ConfigDict {
    return {gain: this.gain.get()};
  }
}
ClassNameMap.register('Identity', Identity);

/**
 * Computes the number of input and output units for a weight shape.
 * @param shape Shape of weight.
 * @param dataFormat data format to use for convolution kernels.
 *   Note that all kernels in Keras are standardized on the
 *   CHANNEL_LAST ordering (even when inputs are set to CHANNEL_FIRST).
 * @return An length-2 array: fanIn, fanOut.
 */
function computeFans(
    shape: Shape, dataFormat: DataFormat = 'channelsLast'): number[] {
  let fanIn: number;
  let fanOut: number;
  checkDataFormat(dataFormat);
  if (shape.length === 2) {
    fanIn = shape[0];
    fanOut = shape[1];
  } else if ([3, 4, 5].indexOf(shape.length) !== -1) {
    if (dataFormat === 'channelsFirst') {
      const receptiveFieldSize = arrayProd(shape, 2);
      fanIn = shape[1] * receptiveFieldSize;
      fanOut = shape[0] * receptiveFieldSize;
    } else if (dataFormat === 'channelsLast') {
      const receptiveFieldSize = arrayProd(shape, 0, shape.length - 2);
      fanIn = shape[shape.length - 2] * receptiveFieldSize;
      fanOut = shape[shape.length - 1] * receptiveFieldSize;
    }
  } else {
    const shapeProd = arrayProd(shape);
    fanIn = Math.sqrt(shapeProd);
    fanOut = Math.sqrt(shapeProd);
  }

  return [fanIn, fanOut];
}

export interface VarianceScalingConfig {
  /** Scaling factor (positive float). */
  scale: number;

  /** Fanning mode for inputs and outputs. */
  mode: FanMode;

  /** Probabilistic distribution of the values. */
  distribution: Distribution;

  /** Random number generator seed. */
  seed?: number;
}


/**
 * Initializer capable of adapting its scale to the shape of weights.
 * With distribution=NORMAL, samples are drawn from a truncated normal
 * distribution centered on zero, with `stddev = sqrt(scale / n)` where n is:
 *   - number of input units in the weight tensor, if mode = FAN_IN.
 *   - number of output units, if mode = FAN_OUT.
 *   - average of the numbers of input and output units, if mode = FAN_AVG.
 * With distribution=UNIFORM,
 * samples are drawn from a uniform distribution
 * within [-limit, limit], with `limit = sqrt(3 * scale / n)`.
 */
export class VarianceScaling extends Initializer {
  private scale: number;
  private mode: FanMode;
  private distribution: Distribution;
  private seed: number;

  /**
   * Constructor of VarianceScaling.
   * @throws ValueError for invalid value in scale.
   */
  constructor(config: VarianceScalingConfig) {
    super();
    if (config.scale < 0.0) {
      throw new ValueError(
          `scale must be a positive float. Got: ${config.scale}`);
    }
    this.scale = config.scale == null ? 1.0 : config.scale;
    this.mode = config.mode;
    checkFanMode(this.mode);
    this.distribution = config.distribution;
    checkDistribution(this.distribution);
    this.seed = config.seed;
  }

  apply(shape: Shape, dtype?: DType): Tensor {
    const fans = computeFans(shape);
    const fanIn = fans[0];
    const fanOut = fans[1];

    let scale = this.scale;
    if (this.mode === 'fanIn') {
      scale /= Math.max(1, fanIn);
    } else if (this.mode === 'fanOut') {
      scale /= Math.max(1, fanOut);
    } else {
      scale /= Math.max(1, (fanIn + fanOut) / 2);
    }

    if (this.distribution === 'normal') {
      const stddev = Math.sqrt(scale);
      return K.truncatedNormal(shape, 0, stddev, dtype, this.seed);
    } else {
      const limit = Math.sqrt(3 * scale);
      return K.randomUniform(shape, -limit, limit, dtype, this.seed);
    }
  }

  getConfig(): ConfigDict {
    return {
      scale: this.scale,
      mode: this.mode,
      distribution: this.distribution,
      seed: this.seed
    };
  }
}
ClassNameMap.register('VarianceScaling', VarianceScaling);

export interface SeedOnlyInitializerConfig {
  /** Random number generator seed. */
  seed?: number;
}

/**
 * Glorot uniform initializer, also called Xavier uniform initializer.
 * It draws samples from a uniform distribution within [-limit, limit]
 * where `limit` is `sqrt(6 / (fan_in + fan_out))`
 * where `fan_in` is the number of input units in the weight tensor
 * and `fan_out` is the number of output units in the weight tensor
 *
 * Reference:
 *   Glorot & Bengio, AISTATS 2010
 *       http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf.
 */
export class GlorotUniform extends VarianceScaling {
  /**
   * Constructor of GlorotUniform
   * @param scale
   * @param mode
   * @param distribution
   * @param seed
   */
  constructor(config?: SeedOnlyInitializerConfig) {
    super({
      scale: 1.0,
      mode: 'fanAvg',
      distribution: 'uniform',
      seed: config.seed
    });
  }
}
ClassNameMap.register('GlorotUniform', GlorotUniform);

/**
 * Glorot normal initializer, also called Xavier normal initializer.
 * It draws samples from a truncated normal distribution centered on 0
 * with `stddev = sqrt(2 / (fan_in + fan_out))`
 * where `fan_in` is the number of input units in the weight tensor
 * and `fan_out` is the number of output units in the weight tensor.
 *
 * Reference:
 *   Glorot & Bengio, AISTATS 2010
 *       http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
 */
export class GlorotNormal extends VarianceScaling {
  /**
   * Constructor of GlorotNormal.
   * @param scale
   * @param mode
   * @param distribution
   * @param seed
   */
  constructor(config?: SeedOnlyInitializerConfig) {
    super({
      scale: 1.0,
      mode: 'fanAvg',
      distribution: 'normal',
      seed: config.seed
    });
  }
}
ClassNameMap.register('GlorotNormal', GlorotNormal);

/**
 * He normal initializer.
 *
 * It draws samples from a truncated normal distribution centered on 0
 * with `stddev = sqrt(2 / fanIn)`
 * where `fanIn` is the number of input units in the weight tensor.
 *
 * Reference:
 *     He et al., http://arxiv.org/abs/1502.01852
 */
export class HeNormal extends VarianceScaling {
  constructor(config?: SeedOnlyInitializerConfig) {
    super(
        {scale: 2.0, mode: 'fanIn', distribution: 'normal', seed: config.seed});
  }
}
ClassNameMap.register('HeNormal', HeNormal);

/**
 * LeCun normal initializer.
 *
 * It draws samples from a truncated normal distribution centered on 0
 * with `stddev = sqrt(1 / fanIn)`
 * where `fanIn` is the number of input units in the weight tensor.
 *
 * References:
 *   [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
 *   [Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
 */
export class LeCunNormal extends VarianceScaling {
  constructor(config?: SeedOnlyInitializerConfig) {
    super(
        {scale: 1.0, mode: 'fanIn', distribution: 'normal', seed: config.seed});
  }
}
ClassNameMap.register('LeCunNormal', LeCunNormal);

export interface OrthogonalConfig extends SeedOnlyInitializerConfig {
  /**
   * Multiplicative factor to apply to the orthogonal matrix. Defaults to 1.
   */
  gain?: number;
}

/**
 * Initializer that generates a random orthogonal matrix.
 *
 * Reference:
 * [Saxe et al., http://arxiv.org/abs/1312.6120](http://arxiv.org/abs/1312.6120)
 */
export class Orthogonal extends Initializer {
  readonly DEFAULT_GAIN = 1;
  protected readonly gain: number;
  protected readonly seed: number;

  constructor(config?: OrthogonalConfig) {
    super();
    this.gain = config.gain == null ? this.DEFAULT_GAIN : config.gain;
    this.seed = config.seed;

    if (this.seed != null) {
      throw new NotImplementedError(
          'Random seed is not implemented for Orthogonal Initializer yet.');
    }
  }

  apply(shape: Shape, dtype?: DType): Tensor {
    if (shape.length !== 2) {
      throw new NotImplementedError(
          'The Orthogonal Initializer does not support non-2D shapes yet.');
    }
    const normalizedShape = shape[0] >= shape[1] ? shape : [shape[1], shape[0]];
    // TODO(cais): Add seed support.
    const a = K.randomNormal(normalizedShape, 0, 1, DType.float32);
    let q = K.qr(a as Tensor2D)[0];
    if (q.shape[1] > normalizedShape[1]) {
      q = q.slice([0, 0], normalizedShape);
    }
    if (shape[0] < shape[1]) {
      q = q.transpose();
    }
    return K.scalarTimesArray(K.getScalar(this.gain), q);
  }

  getConfig(): ConfigDict {
    return {
      gain: this.gain,
      seed: this.seed,
    };
  }
}
ClassNameMap.register('Orthogonal', Orthogonal);

/** @docinline */
export type InitializerIdentifier = 'constant'|'glorotNormal'|'glorotUniform'|
    'heNormal'|'identity'|'leCunNormal'|'ones'|'orthogonal'|'randomNormal'|
    'randomUniform'|'truncatedNormal'|'varianceScaling'|'zeros'|string;

// Maps the JavaScript-like identifier keys to the corresponding registry
// symbols.
export const INITIALIZER_IDENTIFIER_REGISTRY_SYMBOL_MAP:
    {[identifier in InitializerIdentifier]: string} = {
      'constant': 'Constant',
      'glorotNormal': 'GlorotNormal',
      'glorotUniform': 'GlorotUniform',
      'heNormal': 'HeNormal',
      'identity': 'Identity',
      'leCunNormal': 'LeCunNormal',
      'ones': 'Ones',
      'orthogonal': 'Orthogonal',
      'randomNormal': 'RandomNormal',
      'randomUniform': 'RandomUniform',
      'truncatedNormal': 'TruncatedNormal',
      'varianceScaling': 'VarianceScaling',
      'zeros': 'Zeros'
    };

function deserializeInitializer(
    config: ConfigDict, customObjects: ConfigDict = {}): Initializer {
  return deserializeKerasObject(
      config, ClassNameMap.getMap().pythonClassNameMap, customObjects,
      'initializer');
}

export function serializeInitializer(initializer: Initializer):
    ConfigDictValue {
  return serializeKerasObject(initializer);
}

export function getInitializer(identifier: InitializerIdentifier|Initializer|
                               ConfigDict): Initializer {
  if (typeof identifier === 'string') {
    const className = identifier in INITIALIZER_IDENTIFIER_REGISTRY_SYMBOL_MAP ?
        INITIALIZER_IDENTIFIER_REGISTRY_SYMBOL_MAP[identifier] :
        identifier;
    const config = {className, config: {}};
    return deserializeInitializer(config);
  } else if (identifier instanceof Initializer) {
    return identifier;
  } else {
    return deserializeInitializer(identifier);
  }
}
