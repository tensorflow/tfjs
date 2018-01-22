/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {ENV} from '../environment';
import * as util from '../util';
import * as conv_util from './conv_util';
import {operation} from './decorators';
import {Array4D, DataType, NDArray, Rank, RankMap} from './ndarray';

export class Ops {
  /**
   * Computes the 2D max pooling of an image.
   *
   * @param x The input ndarray, of rank 4 or rank 3 of shape
   *     [batch, height, width, inChannels]. If rank 3, batch of 1 is assumed.
   * @param filterSize The filter size, a tuple [filterHeight, filterWidth].
   * @param strides The strides of the pooling: [strideHeight, strideWidth].
   * @param pad A string from: 'same', 'valid'. The type of padding algorithm.
   *    - 'same' pad and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *    - 'valid' pad: output will be smaller than input if filter is larger
   *       than 1x1.
   *   - For more info, see this guide:
   *     https://www.tensorflow.org/api_guides/python/nn#Convolution
   * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. The
   *     rounding mode used when computing output dimensions if pad is a
   *     number. If none is provided, it will not round and error if the output
   *     is of fractional size.
   */
  @operation
  static maxPool<T extends NDArray>(
      x: T, filterSize: [number, number]|number,
      strides: [number, number]|number, pad: 'valid'|'same'|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    let x4D = x as Array4D;
    let reshapedTo4D = false;
    if (x.rank === 3) {
      reshapedTo4D = true;
      x4D = x.as4D(1, x.shape[0], x.shape[1], x.shape[2]);
    }
    util.assert(
        x4D.rank === 4,
        `Error in maxPool: input must be rank 4 but got rank ${x4D.rank}.`);
    if (dimRoundingMode != null) {
      util.assert(
          util.isInt(pad as number),
          `Error in maxPool: pad must be an integer when using, ` +
              `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
    }
    const convInfo = conv_util.computePool2DInfo(
        x4D.shape, filterSize, strides, pad, dimRoundingMode);

    const gradients = (dy: Array4D<'float32'>, y: Array4D) => {
      return {x: () => Ops.maxPoolBackprop(dy, x4D, filterSize, strides, pad)};
    };
    const res = ENV.engine.executeKernel(
        'MaxPool', {inputs: {x: x4D}, args: {convInfo}}, gradients);
    if (reshapedTo4D) {
      return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as T;
    }
    return res as T;
  }

  /**
   * Computes the backprop of a max pool.
   *
   * @param dy The dy error, of rank 4 or rank 3 of shape
   *     [batchSize, height, width, channels]. If rank 3, batch of 1 is
   * assumed.
   * @param input The input image, of rank 4 or rank 3 of shape
   *     [batchSize, height, width, channels]. If rank 3, batch of 1 is
   * assumed.
   * @param filterSize The filter size, a tuple [filterHeight, filterWidth].
   * @param strides The strides of the pooling: [strideHeight, strideWidth].
   * @param pad A string from: 'same', 'valid'. The type of padding algorithm
   *     used in the forward prop of the op.
   * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. The
   *     rounding mode used when computing output dimensions if pad is a
   *     number. If none is provided, it will not round and error if the output
   *     is of fractional size.
   */
  @operation
  static maxPoolBackprop<
      D extends DataType, R extends Rank, T extends RankMap<'float32'>[R]>(
      dy: NDArray<'float32', R>, input: NDArray<D, R>,
      filterSize: [number, number]|number, strides: [number, number]|number,
      pad: 'valid'|'same'|number, dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    util.assert(
        input.rank === dy.rank,
        `Rank of input (${input.rank}) does not match rank of dy (${dy.rank})`);

    let input4D = input as Array4D;
    let dy4D = dy as Array4D;
    let reshapedTo4D = false;
    if (input.rank === 3) {
      reshapedTo4D = true;
      input4D = input.as4D(1, input.shape[0], input.shape[1], input.shape[2]);
      dy4D = dy.as4D(1, dy.shape[0], dy.shape[1], dy.shape[2]);
    }

    util.assert(
        dy4D.rank === 4,
        `Error in maxPoolBackprop: dy must be rank 4 but got rank ` +
            `${dy4D.rank}.`);
    util.assert(
        input4D.rank === 4,
        `Error in maxPoolBackprop: input must be rank 4 but got rank ` +
            `${input4D.rank}.`);
    if (dimRoundingMode != null) {
      util.assert(
          util.isInt(pad as number),
          `Error in maxPoolBackprop: pad must be an integer when using, ` +
              `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
    }

    const convInfo = conv_util.computePool2DInfo(
        input4D.shape, filterSize, strides, pad, dimRoundingMode);
    const res = ENV.engine.executeKernel(
        'MaxPoolBackprop', {inputs: {dy: dy4D, x: input4D}, args: {convInfo}});
    if (reshapedTo4D) {
      return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as T;
    }
    return res as T;
  }

  /**
   * Computes the 2D min pooling of an image.
   *
   * @param input The input ndarray, of rank 4 or rank 3 of shape
   *     [batch, height, width, inChannels]. If rank 3, batch of 1 is assumed.
   * @param filterSize The filter size, a tuple [filterHeight, filterWidth].
   * @param strides The strides of the pooling: [strideHeight, strideWidth].
   * @param pad A string from: 'same', 'valid'. The type of padding algorithm.
   *    - 'same' pad and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *    - 'valid' pad: output will be smaller than input if filter is larger
   *       than 1x1.
   *   - For more info, see this guide:
   *     https://www.tensorflow.org/api_guides/python/nn#Convolution
   * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. The
   *     rounding mode used when computing output dimensions if pad is a
   *     number. If none is provided, it will not round and error if the output
   *     is of fractional size.
   */
  @operation
  static minPool<T extends NDArray>(
      input: T, filterSize: [number, number]|number,
      strides: [number, number]|number, pad: 'valid'|'same'|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    let input4D = input as Array4D;
    let reshapedTo4D = false;
    if (input.rank === 3) {
      reshapedTo4D = true;
      input4D = input.as4D(1, input.shape[0], input.shape[1], input.shape[2]);
    }
    util.assert(
        input4D.rank === 4,
        `Error in minPool: x must be rank 4 but got rank ${input4D.rank}.`);
    if (dimRoundingMode != null) {
      util.assert(
          util.isInt(pad as number),
          `Error in minPool: pad must be an integer when using, ` +
              `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
    }
    const convInfo = conv_util.computePool2DInfo(
        input4D.shape, filterSize, strides, pad, dimRoundingMode);
    const res = ENV.engine.executeKernel(
        'MinPool', {inputs: {x: input4D}, args: {convInfo}});
    if (reshapedTo4D) {
      return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as T;
    }
    return res as T;
  }

  /**
   * Computes the 2D average pooling of an image.
   *
   * @param x The input ndarray, of rank 4 or rank 3 of shape
   *     [batch, height, width, inChannels]. If rank 3, batch of 1 is assumed.
   * @param filterSize The filter size, a tuple [filterHeight, filterWidth].
   * @param strides The strides of the pooling: [strideHeight, strideWidth].
   * @param pad A string from: 'same', 'valid'. The type of padding algorithm.
   *    - 'same' pad and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *    - 'valid' pad: output will be smaller than input if filter is larger
   *       than 1x1.
   *   - For more info, see this guide:
   *     https://www.tensorflow.org/api_guides/python/nn#Convolution
   * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. The
   *     rounding mode used when computing output dimensions if pad is a
   *     number. If none is provided, it will not round and error if the output
   *     is of fractional size.
   */
  @operation
  static avgPool<R extends '3'|'4'>(
      x: NDArray<'int32'|'float32', R>, filterSize: [number, number]|number,
      strides: [number, number]|number, pad: 'valid'|'same'|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): RankMap<'float32'>[R] {
    let x4D = x as Array4D;
    let reshapedTo4D = false;
    if (x.rank === 3) {
      reshapedTo4D = true;
      x4D = x.as4D(1, x.shape[0], x.shape[1], x.shape[2]);
    }
    util.assert(
        x4D.rank === 4,
        `Error in avgPool: x must be rank 4 but got rank ${x4D.rank}.`);
    if (dimRoundingMode != null) {
      util.assert(
          util.isInt(pad as number),
          `Error in avgPool: pad must be an integer when using, ` +
              `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
    }

    const convInfo =
        conv_util.computePool2DInfo(x4D.shape, filterSize, strides, pad);

    const gradients = (dy: Array4D<'float32'>, y: Array4D) => {
      return {x: () => Ops.avgPoolBackprop(dy, x4D, filterSize, strides, pad)};
    };
    const res = ENV.engine.executeKernel(
        'AvgPool', {inputs: {x: x4D}, args: {convInfo}}, gradients);
    if (reshapedTo4D) {
      return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as
          RankMap<'float32'>[R];
    }
    return res as RankMap<'float32'>[R];
  }

  /**
   * Computes the backprop of an avg pool.
   *
   * @param dy The dy error, of rank 4 or rank 3 of shape
   *     [batchSize, height, width, channels]. If rank 3, batch of 1 is
   * assumed.
   * @param input The input image, of rank 4 or rank 3 of shape
   *     [batchSize, height, width, channels]. If rank 3, batch of 1 is
   * assumed.
   * @param filterSize The filter size, a tuple [filterHeight, filterWidth].
   * @param strides The strides of the pooling: [strideHeight, strideWidth].
   * @param pad A string from: 'same', 'valid'. The type of padding algorithm
   *     used in the forward prop of the op.
   */
  @operation
  static avgPoolBackprop<D extends DataType, R extends
                         '3'|'4', T extends RankMap<'float32'>[R]>(
      dy: NDArray<'float32', R>, input: NDArray<D, R>,
      filterSize: [number, number]|number, strides: [number, number]|number,
      pad: 'valid'|'same'|number): T {
    util.assert(
        input.rank === dy.rank,
        `Rank of input (${input.rank}) does not match rank of dy (${dy.rank})`);

    let input4D = input as Array4D;
    let dy4D = dy as Array4D;
    let reshapedTo4D = false;
    if (input.rank === 3) {
      reshapedTo4D = true;
      input4D = input.as4D(1, input.shape[0], input.shape[1], input.shape[2]);
      dy4D = dy.as4D(1, dy.shape[0], dy.shape[1], dy.shape[2]);
    }

    util.assert(
        dy4D.rank === 4,
        `Error in avgPoolBackprop: dy must be rank 4 but got rank ` +
            `${dy4D.rank}.`);
    util.assert(
        input4D.rank === 4,
        `Error in avgPoolBackprop: input must be rank 4 but got rank ` +
            `${input4D.rank}.`);

    const convInfo =
        conv_util.computePool2DInfo(input4D.shape, filterSize, strides, pad);
    const res = ENV.engine.executeKernel(
        'AvgPoolBackprop', {inputs: {dy: dy4D, x: input4D}, args: {convInfo}});
    if (reshapedTo4D) {
      return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as T;
    }
    return res as T;
  }
}
