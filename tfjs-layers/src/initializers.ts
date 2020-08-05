/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {DataType, eye, linalg, mul, ones, randomUniform, scalar, serialization, Tensor, Tensor2D, tidy, truncatedNormal, zeros} from '@tensorflow/tfjs-core';

import * as K from './backend/tfjs_backend';
import {checkDataFormat} from './common';
import {NotImplementedError, ValueError} from './errors';
import {DataFormat, Shape} from './keras_format/common';
import {Distribution, FanMode, VALID_DISTRIBUTION_VALUES, VALID_FAN_MODE_VALUES} from './keras_format/initializer_config';
import {checkStringTypeUnionValue, deserializeKerasObject, serializeKerasObject} from './utils/generic_utils';
import {arrayProd} from './utils/math_utils';

export function checkFanMode(value?: string): void {
  checkStringTypeUnionValue(VALID_FAN_MODE_VALUES, 'FanMode', value);
}

export function checkDistribution(value?: string): void {
  checkStringTypeUnionValue(VALID_DISTRIBUTION_VALUES, 'Distribution', value);
}

/**
 * Initializer base class.
 *
 * @doc {
 *   heading: 'Initializers', subheading: 'Classes', namespace: 'initializers'}
 */
export abstract class Initializer extends serialization.Serializable {
  public fromConfigUsesCustomObjects(): boolean {
    return false;
  }
  /**
   * Generate an initial value.
   * @param shape
   * @param dtype
   * @return The init value.
   */
  abstract apply(shape: Shape, dtype?: DataType): Tensor;

  getConfig(): serialization.ConfigDict {
    return {};
  }
}

export class Zeros extends Initializer {
  /** @nocollapse */
  static className = 'Zeros';

  apply(shape: Shape, dtype?: DataType): Tensor {
    return zeros(shape, dtype);
  }
}
serialization.registerClass(Zeros);

export class Ones extends Initializer {
  /** @nocollapse */
  static className = 'Ones';

  apply(shape: Shape, dtype?: DataType): Tensor {
    return ones(shape, dtype);
  }
}
serialization.registerClass(Ones);

export interface ConstantArgs {
  /** The value for each element in the variable. */
  value: number;
}

export class Constant extends Initializer {
  /** @nocollapse */
  static className = 'Constant';
  private value: number;
  constructor(args: ConstantArgs) {
    super();
    if (typeof args !== 'object') {
      throw new ValueError(
          `Expected argument of type ConstantConfig but got ${args}`);
    }
    if (args.value === undefined) {
      throw new ValueError(`config must have value set but got ${args}`);
    }
    this.value = args.value;
  }

  apply(shape: Shape, dtype?: DataType): Tensor {
    return tidy(() => mul(scalar(this.value), ones(shape, dtype)));
  }

  getConfig(): serialization.ConfigDict {
    return {
      value: this.value,
    };
  }
}
serialization.registerClass(Constant);

export interface RandomUniformArgs {
  /** Lower bound of the range of random values to generate. */
  minval?: number;
  /** Upper bound of the range of random values to generate. */
  maxval?: number;
  /** Used to seed the random generator. */
  seed?: number;
}

export class RandomUniform extends Initializer {
  /** @nocollapse */
  static className = 'RandomUniform';
  readonly DEFAULT_MINVAL = -0.05;
  readonly DEFAULT_MAXVAL = 0.05;
  private minval: number;
  private maxval: number;
  private seed: number;

  constructor(args: RandomUniformArgs) {
    super();
    this.minval = args.minval || this.DEFAULT_MINVAL;
    this.maxval = args.maxval || this.DEFAULT_MAXVAL;
    this.seed = args.seed;
  }

  apply(shape: Shape, dtype?: DataType): Tensor {
    return randomUniform(shape, this.minval, this.maxval, dtype);
  }

  getConfig(): serialization.ConfigDict {
    return {minval: this.minval, maxval: this.maxval, seed: this.seed};
  }
}
serialization.registerClass(RandomUniform);

export interface RandomNormalArgs {
  /** Mean of the random values to generate. */
  mean?: number;
  /** Standard deviation of the random values to generate. */
  stddev?: number;
  /** Used to seed the random generator. */
  seed?: number;
}

export class RandomNormal extends Initializer {
  /** @nocollapse */
  static className = 'RandomNormal';
  readonly DEFAULT_MEAN = 0.;
  readonly DEFAULT_STDDEV = 0.05;
  private mean: number;
  private stddev: number;
  private seed: number;

  constructor(args: RandomNormalArgs) {
    super();
    this.mean = args.mean || this.DEFAULT_MEAN;
    this.stddev = args.stddev || this.DEFAULT_STDDEV;
    this.seed = args.seed;
  }

  apply(shape: Shape, dtype?: DataType): Tensor {
    dtype = dtype || 'float32';
    if (dtype !== 'float32' && dtype !== 'int32') {
      throw new NotImplementedError(
          `randomNormal does not support dType ${dtype}.`);
    }

    return K.randomNormal(shape, this.mean, this.stddev, dtype, this.seed);
  }

  getConfig(): serialization.ConfigDict {
    return {mean: this.mean, stddev: this.stddev, seed: this.seed};
  }
}
serialization.registerClass(RandomNormal);

export interface TruncatedNormalArgs {
  /** Mean of the random values to generate. */
  mean?: number;
  /** Standard deviation of the random values to generate. */
  stddev?: number;
  /** Used to seed the random generator. */
  seed?: number;
}

export class TruncatedNormal extends Initializer {
  /** @nocollapse */
  static className = 'TruncatedNormal';

  readonly DEFAULT_MEAN = 0.;
  readonly DEFAULT_STDDEV = 0.05;
  private mean: number;
  private stddev: number;
  private seed: number;

  constructor(args: TruncatedNormalArgs) {
    super();
    this.mean = args.mean || this.DEFAULT_MEAN;
    this.stddev = args.stddev || this.DEFAULT_STDDEV;
    this.seed = args.seed;
  }

  apply(shape: Shape, dtype?: DataType): Tensor {
    dtype = dtype || 'float32';
    if (dtype !== 'float32' && dtype !== 'int32') {
      throw new NotImplementedError(
          `truncatedNormal does not support dType ${dtype}.`);
    }
    return truncatedNormal(shape, this.mean, this.stddev, dtype, this.seed);
  }

  getConfig(): serialization.ConfigDict {
    return {mean: this.mean, stddev: this.stddev, seed: this.seed};
  }
}
serialization.registerClass(TruncatedNormal);

export interface IdentityArgs {
  /**
   * Multiplicative factor to apply to the identity matrix.
   */
  gain?: number;
}

export class Identity extends Initializer {
  /** @nocollapse */
  static className = 'Identity';
  private gain: number;
  constructor(args: IdentityArgs) {
    super();
    this.gain = args.gain != null ? args.gain : 1.0;
  }

  apply(shape: Shape, dtype?: DataType): Tensor {
    return tidy(() => {
      if (shape.length !== 2 || shape[0] !== shape[1]) {
        throw new ValueError(
            'Identity matrix initializer can only be used for' +
            ' 2D square matrices.');
      } else {
        return mul(this.gain, eye(shape[0]));
      }
    });
  }

  getConfig(): serialization.ConfigDict {
    return {gain: this.gain};
  }
}
serialization.registerClass(Identity);

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

export interface VarianceScalingArgs {
  /** Scaling factor (positive float). */
  scale?: number;

  /** Fanning mode for inputs and outputs. */
  mode?: FanMode;

  /** Probabilistic distribution of the values. */
  distribution?: Distribution;

  /** Random number generator seed. */
  seed?: number;
}

export class VarianceScaling extends Initializer {
  /** @nocollapse */
  static className = 'VarianceScaling';
  private scale: number;
  private mode: FanMode;
  private distribution: Distribution;
  private seed: number;

  /**
   * Constructor of VarianceScaling.
   * @throws ValueError for invalid value in scale.
   */
  constructor(args: VarianceScalingArgs) {
    super();
    if (args.scale < 0.0) {
      throw new ValueError(
          `scale must be a positive float. Got: ${args.scale}`);
    }
    this.scale = args.scale == null ? 1.0 : args.scale;
    this.mode = args.mode == null ? 'fanIn' : args.mode;
    checkFanMode(this.mode);
    this.distribution =
        args.distribution == null ? 'normal' : args.distribution;
    checkDistribution(this.distribution);
    this.seed = args.seed;
  }

  apply(shape: Shape, dtype?: DataType): Tensor {
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
      dtype = dtype || 'float32';
      if (dtype !== 'float32' && dtype !== 'int32') {
        throw new NotImplementedError(
            `${this.getClassName()} does not support dType ${dtype}.`);
      }
      return truncatedNormal(shape, 0, stddev, dtype, this.seed);
    } else {
      const limit = Math.sqrt(3 * scale);
      return randomUniform(shape, -limit, limit, dtype);
    }
  }

  getConfig(): serialization.ConfigDict {
    return {
      scale: this.scale,
      mode: this.mode,
      distribution: this.distribution,
      seed: this.seed
    };
  }
}
serialization.registerClass(VarianceScaling);

export interface SeedOnlyInitializerArgs {
  /** Random number generator seed. */
  seed?: number;
}

export class GlorotUniform extends VarianceScaling {
  /** @nocollapse */
  static className = 'GlorotUniform';

  /**
   * Constructor of GlorotUniform
   * @param scale
   * @param mode
   * @param distribution
   * @param seed
   */
  constructor(args?: SeedOnlyInitializerArgs) {
    super({
      scale: 1.0,
      mode: 'fanAvg',
      distribution: 'uniform',
      seed: args == null ? null : args.seed
    });
  }

  getClassName(): string {
    // In Python Keras, GlorotUniform is not a class, but a helper method
    // that creates a VarianceScaling object. Use 'VarianceScaling' as
    // class name to be compatible with that.
    return VarianceScaling.className;
  }
}
serialization.registerClass(GlorotUniform);

export class GlorotNormal extends VarianceScaling {
  /** @nocollapse */
  static className = 'GlorotNormal';

  /**
   * Constructor of GlorotNormal.
   * @param scale
   * @param mode
   * @param distribution
   * @param seed
   */
  constructor(args?: SeedOnlyInitializerArgs) {
    super({
      scale: 1.0,
      mode: 'fanAvg',
      distribution: 'normal',
      seed: args == null ? null : args.seed
    });
  }

  getClassName(): string {
    // In Python Keras, GlorotNormal is not a class, but a helper method
    // that creates a VarianceScaling object. Use 'VarianceScaling' as
    // class name to be compatible with that.
    return VarianceScaling.className;
  }
}
serialization.registerClass(GlorotNormal);

export class HeNormal extends VarianceScaling {
  /** @nocollapse */
  static className = 'HeNormal';

  constructor(args?: SeedOnlyInitializerArgs) {
    super({
      scale: 2.0,
      mode: 'fanIn',
      distribution: 'normal',
      seed: args == null ? null : args.seed
    });
  }

  getClassName(): string {
    // In Python Keras, HeNormal is not a class, but a helper method
    // that creates a VarianceScaling object. Use 'VarianceScaling' as
    // class name to be compatible with that.
    return VarianceScaling.className;
  }
}
serialization.registerClass(HeNormal);

export class HeUniform extends VarianceScaling {
  /** @nocollapse */
  static className = 'HeUniform';

  constructor(args?: SeedOnlyInitializerArgs) {
    super({
      scale: 2.0,
      mode: 'fanIn',
      distribution: 'uniform',
      seed: args == null ? null : args.seed
    });
  }

  getClassName(): string {
    // In Python Keras, HeUniform is not a class, but a helper method
    // that creates a VarianceScaling object. Use 'VarianceScaling' as
    // class name to be compatible with that.
    return VarianceScaling.className;
  }
}
serialization.registerClass(HeUniform);

export class LeCunNormal extends VarianceScaling {
  /** @nocollapse */
  static className = 'LeCunNormal';

  constructor(args?: SeedOnlyInitializerArgs) {
    super({
      scale: 1.0,
      mode: 'fanIn',
      distribution: 'normal',
      seed: args == null ? null : args.seed
    });
  }

  getClassName(): string {
    // In Python Keras, LeCunNormal is not a class, but a helper method
    // that creates a VarianceScaling object. Use 'VarianceScaling' as
    // class name to be compatible with that.
    return VarianceScaling.className;
  }
}
serialization.registerClass(LeCunNormal);

export class LeCunUniform extends VarianceScaling {
  /** @nocollapse */
  static className = 'LeCunNormal';

  constructor(args?: SeedOnlyInitializerArgs) {
    super({
      scale: 1.0,
      mode: 'fanIn',
      distribution: 'uniform',
      seed: args == null ? null : args.seed
    });
  }

  getClassName(): string {
    // In Python Keras, LeCunUniform is not a class, but a helper method
    // that creates a VarianceScaling object. Use 'VarianceScaling' as
    // class name to be compatible with that.
    return VarianceScaling.className;
  }
}
serialization.registerClass(LeCunUniform);

export interface OrthogonalArgs extends SeedOnlyInitializerArgs {
  /**
   * Multiplicative factor to apply to the orthogonal matrix. Defaults to 1.
   */
  gain?: number;
}

export class Orthogonal extends Initializer {
  /** @nocollapse */
  static className = 'Orthogonal';
  readonly DEFAULT_GAIN = 1;
  protected readonly gain: number;
  protected readonly seed: number;

  constructor(args?: OrthogonalArgs) {
    super();
    this.gain = args.gain == null ? this.DEFAULT_GAIN : args.gain;
    this.seed = args.seed;

    if (this.seed != null) {
      throw new NotImplementedError(
          'Random seed is not implemented for Orthogonal Initializer yet.');
    }
  }

  apply(shape: Shape, dtype?: DataType): Tensor {
    return tidy(() => {
      if (shape.length < 2) {
        throw new NotImplementedError('Shape must be at least 2D.');
      }
      if (shape[0] * shape[1] > 2000) {
        console.warn(
            `Orthogonal initializer is being called on a matrix with more ` +
            `than 2000 (${shape[0] * shape[1]}) elements: ` +
            `Slowness may result.`);
      }

      // TODO(cais): Add seed support.
      const normalizedShape =
          shape[0] > shape[1] ? [shape[1], shape[0]] : shape;
      const a = K.randomNormal(normalizedShape, 0, 1, 'float32') as Tensor2D;
      let q = linalg.gramSchmidt(a) as Tensor2D;
      if (shape[0] > shape[1]) {
        q = q.transpose();
      }
      return mul(this.gain, q);
    });
  }

  getConfig(): serialization.ConfigDict {
    return {
      gain: this.gain,
      seed: this.seed,
    };
  }
}
serialization.registerClass(Orthogonal);

/** @docinline */
export type InitializerIdentifier =
    'constant'|'glorotNormal'|'glorotUniform'|'heNormal'|'heUniform'|'identity'|
    'leCunNormal'|'leCunUniform'|'ones'|'orthogonal'|'randomNormal'|
    'randomUniform'|'truncatedNormal'|'varianceScaling'|'zeros'|string;

// Maps the JavaScript-like identifier keys to the corresponding registry
// symbols.
export const INITIALIZER_IDENTIFIER_REGISTRY_SYMBOL_MAP:
    {[identifier in InitializerIdentifier]: string} = {
      'constant': 'Constant',
      'glorotNormal': 'GlorotNormal',
      'glorotUniform': 'GlorotUniform',
      'heNormal': 'HeNormal',
      'heUniform': 'HeUniform',
      'identity': 'Identity',
      'leCunNormal': 'LeCunNormal',
      'leCunUniform': 'LeCunUniform',
      'ones': 'Ones',
      'orthogonal': 'Orthogonal',
      'randomNormal': 'RandomNormal',
      'randomUniform': 'RandomUniform',
      'truncatedNormal': 'TruncatedNormal',
      'varianceScaling': 'VarianceScaling',
      'zeros': 'Zeros'
    };

function deserializeInitializer(
    config: serialization.ConfigDict,
    customObjects: serialization.ConfigDict = {}): Initializer {
  return deserializeKerasObject(
      config, serialization.SerializationMap.getMap().classNameMap,
      customObjects, 'initializer');
}

export function serializeInitializer(initializer: Initializer):
    serialization.ConfigDictValue {
  return serializeKerasObject(initializer);
}

export function getInitializer(identifier: InitializerIdentifier|Initializer|
                               serialization.ConfigDict): Initializer {
  if (typeof identifier === 'string') {
    const className = identifier in INITIALIZER_IDENTIFIER_REGISTRY_SYMBOL_MAP ?
        INITIALIZER_IDENTIFIER_REGISTRY_SYMBOL_MAP[identifier] :
        identifier;
    /* We have four 'helper' classes for common initializers that
    all get serialized as 'VarianceScaling' and shouldn't go through
    the deserializeInitializer pathway. */
    if (className === 'GlorotNormal') {
      return new GlorotNormal();
    } else if (className === 'GlorotUniform') {
      return new GlorotUniform();
    } else if (className === 'HeNormal') {
      return new HeNormal();
    } else if (className === 'HeUniform') {
      return new HeUniform();
    } else if (className === 'LeCunNormal') {
      return new LeCunNormal();
    } else if (className === 'LeCunUniform') {
      return new LeCunUniform();
    } else {
      const config: serialization.ConfigDict = {};
      config['className'] = className;
      config['config'] = {};
      return deserializeInitializer(config);
    }
  } else if (identifier instanceof Initializer) {
    return identifier;
  } else {
    return deserializeInitializer(identifier);
  }
}
