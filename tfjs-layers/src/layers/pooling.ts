/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * TensorFlow.js Layers: Pooling Layers.
 */

import * as tfc from '@tensorflow/tfjs-core';
import {serialization, Tensor, Tensor3D, Tensor4D, Tensor5D, tidy} from '@tensorflow/tfjs-core';

import {imageDataFormat} from '../backend/common';
import * as K from '../backend/tfjs_backend';
import {checkDataFormat, checkPaddingMode, checkPoolMode} from '../common';
import {InputSpec} from '../engine/topology';
import {Layer, LayerArgs} from '../engine/topology';
import {NotImplementedError, ValueError} from '../errors';
import {DataFormat, PaddingMode, PoolMode, Shape} from '../keras_format/common';
import {Kwargs} from '../types';
import {convOutputLength} from '../utils/conv_utils';
import {assertPositiveInteger} from '../utils/generic_utils';
import {getExactlyOneShape, getExactlyOneTensor} from '../utils/types_utils';

import {preprocessConv2DInput, preprocessConv3DInput} from './convolutional';

/**
 * 2D pooling.
 * @param x
 * @param poolSize
 * @param stridesdes strides. Defaults to [1, 1].
 * @param padding padding. Defaults to 'valid'.
 * @param dataFormat data format. Defaults to 'channelsLast'.
 * @param poolMode Mode of pooling. Defaults to 'max'.
 * @returns Result of the 2D pooling.
 */
export function pool2d(
    x: Tensor, poolSize: [number, number], strides?: [number, number],
    padding?: PaddingMode, dataFormat?: DataFormat,
    poolMode?: PoolMode): Tensor {
  return tidy(() => {
    checkDataFormat(dataFormat);
    checkPoolMode(poolMode);
    checkPaddingMode(padding);
    if (strides == null) {
      strides = [1, 1];
    }
    if (padding == null) {
      padding = 'valid';
    }
    if (dataFormat == null) {
      dataFormat = imageDataFormat();
    }
    if (poolMode == null) {
      poolMode = 'max';
    }

    // TODO(cais): Remove the preprocessing step once deeplearn.js supports
    // dataFormat as an input argument.
    x = preprocessConv2DInput(x, dataFormat);  // x is NHWC after preprocessing.
    let y: Tensor;
    const paddingString = (padding === 'same') ? 'same' : 'valid';
    if (poolMode === 'max') {
      // TODO(cais): Rank check?
      y = tfc.maxPool(x as Tensor4D, poolSize, strides, paddingString);
    } else {  // 'avg'
      // TODO(cais): Check the dtype and rank of x and give clear error message
      //   if those are incorrect.
      y = tfc.avgPool(
          // TODO(cais): Rank check?
          x as Tensor3D | Tensor4D, poolSize, strides, paddingString);
    }
    if (dataFormat === 'channelsFirst') {
      y = tfc.transpose(y, [0, 3, 1, 2]);  // NHWC -> NCHW.
    }
    return y;
  });
}

/**
 * 3D pooling.
 * @param x
 * @param poolSize. Default to [1, 1, 1].
 * @param strides strides. Defaults to [1, 1, 1].
 * @param padding padding. Defaults to 'valid'.
 * @param dataFormat data format. Defaults to 'channelsLast'.
 * @param poolMode Mode of pooling. Defaults to 'max'.
 * @returns Result of the 3D pooling.
 */
export function pool3d(
    x: Tensor5D, poolSize: [number, number, number],
    strides?: [number, number, number], padding?: PaddingMode,
    dataFormat?: DataFormat, poolMode?: PoolMode): Tensor {
  return tidy(() => {
    checkDataFormat(dataFormat);
    checkPoolMode(poolMode);
    checkPaddingMode(padding);
    if (strides == null) {
      strides = [1, 1, 1];
    }
    if (padding == null) {
      padding = 'valid';
    }
    if (dataFormat == null) {
      dataFormat = imageDataFormat();
    }
    if (poolMode == null) {
      poolMode = 'max';
    }

    // x is NDHWC after preprocessing.
    x = preprocessConv3DInput(x as Tensor, dataFormat) as Tensor5D;
    let y: Tensor;
    const paddingString = (padding === 'same') ? 'same' : 'valid';
    if (poolMode === 'max') {
      y = tfc.maxPool3d(x, poolSize, strides, paddingString);
    } else {  // 'avg'
      y = tfc.avgPool3d(x, poolSize, strides, paddingString);
    }
    if (dataFormat === 'channelsFirst') {
      y = tfc.transpose(y, [0, 4, 1, 2, 3]);  // NDHWC -> NCDHW.
    }
    return y;
  });
}

export declare interface Pooling1DLayerArgs extends LayerArgs {
  /**
   * Size of the window to pool over, should be an integer.
   */
  poolSize?: number|[number];
  /**
   * Period at which to sample the pooled values.
   *
   * If `null`, defaults to `poolSize`.
   */
  strides?: number|[number];
  /** How to fill in data that's not an integer multiple of poolSize. */
  padding?: PaddingMode;
}

/**
 * Abstract class for different pooling 1D layers.
 */
export abstract class Pooling1D extends Layer {
  protected readonly poolSize: [number];
  protected readonly strides: [number];
  protected readonly padding: PaddingMode;

  /**
   *
   * @param args Parameters for the Pooling layer.
   *
   * config.poolSize defaults to 2.
   */
  constructor(args: Pooling1DLayerArgs) {
    if (args.poolSize == null) {
      args.poolSize = 2;
    }
    super(args);
    if (typeof args.poolSize === 'number') {
      this.poolSize = [args.poolSize];
    } else if (
        Array.isArray(args.poolSize) &&
        (args.poolSize as number[]).length === 1 &&
        typeof (args.poolSize as number[])[0] === 'number') {
      this.poolSize = args.poolSize;
    } else {
      throw new ValueError(
          `poolSize for 1D convolutional layer must be a number or an ` +
          `Array of a single number, but received ` +
          `${JSON.stringify(args.poolSize)}`);
    }
    assertPositiveInteger(this.poolSize, 'poolSize');
    if (args.strides == null) {
      this.strides = this.poolSize;
    } else {
      if (typeof args.strides === 'number') {
        this.strides = [args.strides];
      } else if (
          Array.isArray(args.strides) &&
          (args.strides as number[]).length === 1 &&
          typeof (args.strides as number[])[0] === 'number') {
        this.strides = args.strides;
      } else {
        throw new ValueError(
            `strides for 1D convolutional layer must be a number or an ` +
            `Array of a single number, but received ` +
            `${JSON.stringify(args.strides)}`);
      }
    }
    assertPositiveInteger(this.strides, 'strides');

    this.padding = args.padding == null ? 'valid' : args.padding;
    checkPaddingMode(this.padding);
    this.inputSpec = [new InputSpec({ndim: 3})];
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = getExactlyOneShape(inputShape);
    const length = convOutputLength(
        inputShape[1], this.poolSize[0], this.padding, this.strides[0]);
    return [inputShape[0], length, inputShape[2]];
  }

  protected abstract poolingFunction(
      inputs: Tensor, poolSize: [number, number], strides: [number, number],
      padding: PaddingMode, dataFormat: DataFormat): Tensor;

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      this.invokeCallHook(inputs, kwargs);
      // Add dummy last dimension.
      inputs = K.expandDims(getExactlyOneTensor(inputs), 2);
      const output = this.poolingFunction(
          getExactlyOneTensor(inputs), [this.poolSize[0], 1],
          [this.strides[0], 1], this.padding, 'channelsLast');
      // Remove dummy last dimension.
      return tfc.squeeze(output, [2]);
    });
  }

  getConfig(): serialization.ConfigDict {
    const config = {
      poolSize: this.poolSize,
      padding: this.padding,
      strides: this.strides,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}

export class MaxPooling1D extends Pooling1D {
  /** @nocollapse */
  static className = 'MaxPooling1D';
  constructor(args: Pooling1DLayerArgs) {
    super(args);
  }

  protected poolingFunction(
      inputs: Tensor, poolSize: [number, number], strides: [number, number],
      padding: PaddingMode, dataFormat: DataFormat): Tensor {
    checkDataFormat(dataFormat);
    checkPaddingMode(padding);
    return pool2d(inputs, poolSize, strides, padding, dataFormat, 'max');
  }
}
serialization.registerClass(MaxPooling1D);

export class AveragePooling1D extends Pooling1D {
  /** @nocollapse */
  static className = 'AveragePooling1D';
  constructor(args: Pooling1DLayerArgs) {
    super(args);
  }

  protected poolingFunction(
      inputs: Tensor, poolSize: [number, number], strides: [number, number],
      padding: PaddingMode, dataFormat: DataFormat): Tensor {
    checkDataFormat(dataFormat);
    checkPaddingMode(padding);
    return pool2d(inputs, poolSize, strides, padding, dataFormat, 'avg');
  }
}
serialization.registerClass(AveragePooling1D);

export declare interface Pooling2DLayerArgs extends LayerArgs {
  /**
   * Factors by which to downscale in each dimension [vertical, horizontal].
   * Expects an integer or an array of 2 integers.
   *
   * For example, `[2, 2]` will halve the input in both spatial dimension.
   * If only one integer is specified, the same window length
   * will be used for both dimensions.
   */
  poolSize?: number|[number, number];

  /**
   * The size of the stride in each dimension of the pooling window. Expects
   * an integer or an array of 2 integers. Integer, tuple of 2 integers, or
   * None.
   *
   * If `null`, defaults to `poolSize`.
   */
  strides?: number|[number, number];

  /** The padding type to use for the pooling layer. */
  padding?: PaddingMode;
  /** The data format to use for the pooling layer. */
  dataFormat?: DataFormat;
}

/**
 * Abstract class for different pooling 2D layers.
 */
export abstract class Pooling2D extends Layer {
  protected readonly poolSize: [number, number];
  protected readonly strides: [number, number];
  protected readonly padding: PaddingMode;
  protected readonly dataFormat: DataFormat;

  constructor(args: Pooling2DLayerArgs) {
    if (args.poolSize == null) {
      args.poolSize = [2, 2];
    }
    super(args);
    this.poolSize = Array.isArray(args.poolSize) ?
        args.poolSize :
        [args.poolSize, args.poolSize];
    if (args.strides == null) {
      this.strides = this.poolSize;
    } else if (Array.isArray(args.strides)) {
      if (args.strides.length !== 2) {
        throw new ValueError(
            `If the strides property of a 2D pooling layer is an Array, ` +
            `it is expected to have a length of 2, but received length ` +
            `${args.strides.length}.`);
      }
      this.strides = args.strides;
    } else {
      // `config.strides` is a number.
      this.strides = [args.strides, args.strides];
    }
    assertPositiveInteger(this.poolSize, 'poolSize');
    assertPositiveInteger(this.strides, 'strides');
    this.padding = args.padding == null ? 'valid' : args.padding;
    this.dataFormat =
        args.dataFormat == null ? 'channelsLast' : args.dataFormat;
    checkDataFormat(this.dataFormat);
    checkPaddingMode(this.padding);

    this.inputSpec = [new InputSpec({ndim: 4})];
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = getExactlyOneShape(inputShape);
    let rows =
        this.dataFormat === 'channelsFirst' ? inputShape[2] : inputShape[1];
    let cols =
        this.dataFormat === 'channelsFirst' ? inputShape[3] : inputShape[2];
    rows =
        convOutputLength(rows, this.poolSize[0], this.padding, this.strides[0]);
    cols =
        convOutputLength(cols, this.poolSize[1], this.padding, this.strides[1]);
    if (this.dataFormat === 'channelsFirst') {
      return [inputShape[0], inputShape[1], rows, cols];
    } else {
      return [inputShape[0], rows, cols, inputShape[3]];
    }
  }

  protected abstract poolingFunction(
      inputs: Tensor, poolSize: [number, number], strides: [number, number],
      padding: PaddingMode, dataFormat: DataFormat): Tensor;

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      this.invokeCallHook(inputs, kwargs);
      return this.poolingFunction(
          getExactlyOneTensor(inputs), this.poolSize, this.strides,
          this.padding, this.dataFormat);
    });
  }

  getConfig(): serialization.ConfigDict {
    const config = {
      poolSize: this.poolSize,
      padding: this.padding,
      strides: this.strides,
      dataFormat: this.dataFormat
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}

export class MaxPooling2D extends Pooling2D {
  /** @nocollapse */
  static className = 'MaxPooling2D';
  constructor(args: Pooling2DLayerArgs) {
    super(args);
  }

  protected poolingFunction(
      inputs: Tensor, poolSize: [number, number], strides: [number, number],
      padding: PaddingMode, dataFormat: DataFormat): Tensor {
    checkDataFormat(dataFormat);
    checkPaddingMode(padding);
    return pool2d(inputs, poolSize, strides, padding, dataFormat, 'max');
  }
}
serialization.registerClass(MaxPooling2D);

export class AveragePooling2D extends Pooling2D {
  /** @nocollapse */
  static className = 'AveragePooling2D';
  constructor(args: Pooling2DLayerArgs) {
    super(args);
  }

  protected poolingFunction(
      inputs: Tensor, poolSize: [number, number], strides: [number, number],
      padding: PaddingMode, dataFormat: DataFormat): Tensor {
    checkDataFormat(dataFormat);
    checkPaddingMode(padding);
    return pool2d(inputs, poolSize, strides, padding, dataFormat, 'avg');
  }
}
serialization.registerClass(AveragePooling2D);

export declare interface Pooling3DLayerArgs extends LayerArgs {
  /**
   * Factors by which to downscale in each dimension [depth, height, width].
   * Expects an integer or an array of 3 integers.
   *
   * For example, `[2, 2, 2]` will halve the input in three dimensions.
   * If only one integer is specified, the same window length
   * will be used for all dimensions.
   */
  poolSize?: number|[number, number, number];

  /**
   * The size of the stride in each dimension of the pooling window. Expects
   * an integer or an array of 3 integers. Integer, tuple of 3 integers, or
   * None.
   *
   * If `null`, defaults to `poolSize`.
   */
  strides?: number|[number, number, number];

  /** The padding type to use for the pooling layer. */
  padding?: PaddingMode;
  /** The data format to use for the pooling layer. */
  dataFormat?: DataFormat;
}

/**
 * Abstract class for different pooling 3D layers.
 */
export abstract class Pooling3D extends Layer {
  protected readonly poolSize: [number, number, number];
  protected readonly strides: [number, number, number];
  protected readonly padding: PaddingMode;
  protected readonly dataFormat: DataFormat;

  constructor(args: Pooling3DLayerArgs) {
    if (args.poolSize == null) {
      args.poolSize = [2, 2, 2];
    }
    super(args);
    this.poolSize = Array.isArray(args.poolSize) ?
        args.poolSize :
        [args.poolSize, args.poolSize, args.poolSize];
    if (args.strides == null) {
      this.strides = this.poolSize;
    } else if (Array.isArray(args.strides)) {
      if (args.strides.length !== 3) {
        throw new ValueError(
            `If the strides property of a 3D pooling layer is an Array, ` +
            `it is expected to have a length of 3, but received length ` +
            `${args.strides.length}.`);
      }
      this.strides = args.strides;
    } else {
      // `config.strides` is a number.
      this.strides = [args.strides, args.strides, args.strides];
    }
    assertPositiveInteger(this.poolSize, 'poolSize');
    assertPositiveInteger(this.strides, 'strides');
    this.padding = args.padding == null ? 'valid' : args.padding;
    this.dataFormat =
        args.dataFormat == null ? 'channelsLast' : args.dataFormat;
    checkDataFormat(this.dataFormat);
    checkPaddingMode(this.padding);

    this.inputSpec = [new InputSpec({ndim: 5})];
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = getExactlyOneShape(inputShape);
    let depths =
        this.dataFormat === 'channelsFirst' ? inputShape[2] : inputShape[1];
    let rows =
        this.dataFormat === 'channelsFirst' ? inputShape[3] : inputShape[2];
    let cols =
        this.dataFormat === 'channelsFirst' ? inputShape[4] : inputShape[3];
    depths = convOutputLength(
        depths, this.poolSize[0], this.padding, this.strides[0]);
    rows =
        convOutputLength(rows, this.poolSize[1], this.padding, this.strides[1]);
    cols =
        convOutputLength(cols, this.poolSize[2], this.padding, this.strides[2]);
    if (this.dataFormat === 'channelsFirst') {
      return [inputShape[0], inputShape[1], depths, rows, cols];
    } else {
      return [inputShape[0], depths, rows, cols, inputShape[4]];
    }
  }

  protected abstract poolingFunction(
      inputs: Tensor, poolSize: [number, number, number],
      strides: [number, number, number], padding: PaddingMode,
      dataFormat: DataFormat): Tensor;

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      this.invokeCallHook(inputs, kwargs);
      return this.poolingFunction(
          getExactlyOneTensor(inputs), this.poolSize, this.strides,
          this.padding, this.dataFormat);
    });
  }

  getConfig(): serialization.ConfigDict {
    const config = {
      poolSize: this.poolSize,
      padding: this.padding,
      strides: this.strides,
      dataFormat: this.dataFormat
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}

export class MaxPooling3D extends Pooling3D {
  /** @nocollapse */
  static className = 'MaxPooling3D';
  constructor(args: Pooling3DLayerArgs) {
    super(args);
  }

  protected poolingFunction(
      inputs: Tensor, poolSize: [number, number, number],
      strides: [number, number, number], padding: PaddingMode,
      dataFormat: DataFormat): Tensor {
    checkDataFormat(dataFormat);
    checkPaddingMode(padding);
    return pool3d(
        inputs as Tensor5D, poolSize, strides, padding, dataFormat, 'max');
  }
}
serialization.registerClass(MaxPooling3D);

export class AveragePooling3D extends Pooling3D {
  /** @nocollapse */
  static className = 'AveragePooling3D';
  constructor(args: Pooling3DLayerArgs) {
    super(args);
  }

  protected poolingFunction(
      inputs: Tensor, poolSize: [number, number, number],
      strides: [number, number, number], padding: PaddingMode,
      dataFormat: DataFormat): Tensor {
    checkDataFormat(dataFormat);
    checkPaddingMode(padding);
    return pool3d(
        inputs as Tensor5D, poolSize, strides, padding, dataFormat, 'avg');
  }
}
serialization.registerClass(AveragePooling3D);

/**
 * Abstract class for different global pooling 1D layers.
 */
export abstract class GlobalPooling1D extends Layer {
  constructor(args: LayerArgs) {
    super(args);
    this.inputSpec = [new InputSpec({ndim: 3})];
  }

  computeOutputShape(inputShape: Shape): Shape {
    return [inputShape[0], inputShape[2]];
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    throw new NotImplementedError();
  }
}

export class GlobalAveragePooling1D extends GlobalPooling1D {
  /** @nocollapse */
  static className = 'GlobalAveragePooling1D';
  constructor(args?: LayerArgs) {
    super(args || {});
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      const input = getExactlyOneTensor(inputs);
      return tfc.mean(input, 1);
    });
  }
}
serialization.registerClass(GlobalAveragePooling1D);

export class GlobalMaxPooling1D extends GlobalPooling1D {
  /** @nocollapse */
  static className = 'GlobalMaxPooling1D';
  constructor(args: LayerArgs) {
    super(args || {});
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      const input = getExactlyOneTensor(inputs);
      return tfc.max(input, 1);
    });
  }
}
serialization.registerClass(GlobalMaxPooling1D);

export declare interface GlobalPooling2DLayerArgs extends LayerArgs {
  /**
   * One of `CHANNEL_LAST` (default) or `CHANNEL_FIRST`.
   *
   * The ordering of the dimensions in the inputs. `CHANNEL_LAST` corresponds
   * to inputs with shape `[batch, height, width, channels[` while
   * `CHANNEL_FIRST` corresponds to inputs with shape
   * `[batch, channels, height, width]`.
   */
  dataFormat?: DataFormat;
}

/**
 * Abstract class for different global pooling 2D layers.
 */
export abstract class GlobalPooling2D extends Layer {
  protected dataFormat: DataFormat;
  constructor(args: GlobalPooling2DLayerArgs) {
    super(args);
    this.dataFormat =
        args.dataFormat == null ? 'channelsLast' : args.dataFormat;
    checkDataFormat(this.dataFormat);
    this.inputSpec = [new InputSpec({ndim: 4})];
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = inputShape as Shape;
    if (this.dataFormat === 'channelsLast') {
      return [inputShape[0], inputShape[3]];
    } else {
      return [inputShape[0], inputShape[1]];
    }
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    throw new NotImplementedError();
  }

  getConfig(): serialization.ConfigDict {
    const config = {dataFormat: this.dataFormat};
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}

export class GlobalAveragePooling2D extends GlobalPooling2D {
  /** @nocollapse */
  static className = 'GlobalAveragePooling2D';

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      const input = getExactlyOneTensor(inputs);
      if (this.dataFormat === 'channelsLast') {
        return tfc.mean(input, [1, 2]);
      } else {
        return tfc.mean(input, [2, 3]);
      }
    });
  }
}
serialization.registerClass(GlobalAveragePooling2D);

export class GlobalMaxPooling2D extends GlobalPooling2D {
  /** @nocollapse */
  static className = 'GlobalMaxPooling2D';

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      const input = getExactlyOneTensor(inputs);
      if (this.dataFormat === 'channelsLast') {
        return tfc.max(input, [1, 2]);
      } else {
        return tfc.max(input, [2, 3]);
      }
    });
  }
}
serialization.registerClass(GlobalMaxPooling2D);
