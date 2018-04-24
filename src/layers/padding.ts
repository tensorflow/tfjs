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
 * Padding Layers.
 */

// Porting Note: In Python Keras, the padding layers are in convolutional.py,
//   but we decided to put them in a separate file (padding.ts) for clarity.

// tslint:disable:max-line-length
import {Tensor} from '@tensorflow/tfjs-core';

import {imageDataFormat} from '../backend/common';
import * as K from '../backend/tfjs_backend';
import {DataFormat} from '../common';
import {InputSpec, Layer, LayerConfig} from '../engine/topology';
import {ValueError} from '../errors';
import {ConfigDict, Shape} from '../types';
import {ClassNameMap, getExactlyOneShape, getExactlyOneTensor} from '../utils/generic_utils';
// tslint:enable:max-line-length

export interface ZeroPadding2DLayerConfig extends LayerConfig {
  /**
   * Integer, or `Array` of 2 integers, or `Array` of 2 `Array`s, each of
   * which is an `Array` of 2 integers.
   * - If integer, the same symmetric padding is applied to width and height.
   * - If Array` of 2 integers, interpreted as two different symmetric values
   *   for height and width:
   *   `[symmetricHeightPad, symmetricWidthPad]`.
   * - If `Array` of 2 `Array`s, interpreted as:
   *   `[[topPad, bottomPad], [leftPad, rightPad]]`.
   */
  padding?: number|[number, number]|[[number, number], [number, number]];

  /**
   * One of `'channelsLast'` (default) and `'channelsFirst'`.
   *
   * The ordering of the dimensions in the inputs.
   * `channelsLast` corresponds to inputs with shape
   * `[batch, height, width, channels]` while `channelsFirst`
   * corresponds to inputs with shape
   * `[batch, channels, height, width]`.
   */
  dataFormat?: DataFormat;
}

/**
 * Zero-padding layer for 2D input (e.g., image).
 *
 * This layer can add rows and columns of zeros
 * at the top, bottom, left and right side of an image tensor.
 *
 * Input shape:
 *   4D tensor with shape:
 *   - If `dataFormat` is `"channelsLast"`:
 *     `[batch, rows, cols, channels]`
 *   - If `data_format` is `"channels_first"`:
 *     `[batch, channels, rows, cols]`.
 *
 * Output shape:
 *   4D with shape:
 *   - If `dataFormat` is `"channelsLast"`:
 *     `[batch, paddedRows, paddedCols, channels]`
 *    - If `dataFormat` is `"channelsFirst"`:
 *     `[batch, channels, paddedRows, paddedCols]`.
 */
export class ZeroPadding2D extends Layer {
  static className = 'ZeroPadding2D';
  readonly dataFormat: DataFormat;
  readonly padding: [[number, number], [number, number]];

  constructor(config?: ZeroPadding2DLayerConfig) {
    if (config == null) {
      config = {};
    }
    super(config);

    this.dataFormat =
        config.dataFormat == null ? imageDataFormat() : config.dataFormat;
    // TODO(cais): Maybe refactor the following logic surrounding `padding`
    //   into a helper method.
    if (config.padding == null) {
      this.padding = [[1, 1], [1, 1]];
    } else if (typeof config.padding === 'number') {
      this.padding =
          [[config.padding, config.padding], [config.padding, config.padding]];
    } else {
      config.padding = config.padding as [number, number] |
          [[number, number], [number, number]];
      if (config.padding.length !== 2) {
        throw new ValueError(
            `ZeroPadding2D expects padding to be a length-2 array, but ` +
            `received a length-${config.padding.length} array.`);
      }

      let heightPadding: [number, number];
      let widthPadding: [number, number];
      if (typeof config.padding[0] === 'number') {
        heightPadding =
            [config.padding[0] as number, config.padding[0] as number];
        widthPadding =
            [config.padding[1] as number, config.padding[1] as number];
      } else {
        config.padding = config.padding as [[number, number], [number, number]];

        if (config.padding[0].length !== 2) {
          throw new ValueError(
              `ZeroPadding2D expects height padding to be a length-2 array, ` +
              `but received a length-${config.padding[0].length} array.`);
        }
        heightPadding = config.padding[0] as [number, number];

        if (config.padding[1].length !== 2) {
          throw new ValueError(
              `ZeroPadding2D expects width padding to be a length-2 array, ` +
              `but received a length-${config.padding[1].length} array.`);
        }
        widthPadding = config.padding[1] as [number, number];
      }
      this.padding = [heightPadding, widthPadding];
    }
    this.inputSpec = [new InputSpec({ndim: 4})];
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = getExactlyOneShape(inputShape);

    let rows: number;
    let cols: number;
    if (this.dataFormat === 'channelsFirst') {
      if (inputShape[2] != null && inputShape[2] >= 0) {
        rows = inputShape[2] + this.padding[0][0] + this.padding[0][1];
      } else {
        rows = null;
      }
      if (inputShape[3] != null && inputShape[3] >= 0) {
        cols = inputShape[3] + this.padding[1][0] + this.padding[1][1];
      } else {
        cols = null;
      }
      return [inputShape[0], inputShape[1], rows, cols];
    } else {
      if (inputShape[1] != null && inputShape[1] >= 0) {
        rows = inputShape[1] + this.padding[0][0] + this.padding[0][1];
      } else {
        rows = null;
      }
      if (inputShape[2] != null && inputShape[2] >= 0) {
        cols = inputShape[2] + this.padding[1][0] + this.padding[1][1];
      } else {
        cols = null;
      }
      return [inputShape[0], rows, cols, inputShape[3]];
    }
  }

  // tslint:disable-next-line:no-any
  call(inputs: Tensor|Tensor[], kwargs: any): Tensor|Tensor[] {
    return K.spatial2dPadding(
        getExactlyOneTensor(inputs), this.padding, this.dataFormat);
  }

  getClassName(): string {
    return ZeroPadding2D.className;
  }

  getConfig(): ConfigDict {
    const config: ConfigDict = {
      padding: this.padding,
      dataFormat: this.dataFormat,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
ClassNameMap.register(ZeroPadding2D);
