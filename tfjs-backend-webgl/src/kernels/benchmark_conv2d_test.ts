/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

import * as tf from '@tensorflow/tfjs-core';
import {Tensor4D} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {ALL_ENVS, describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

async function benchmarkConv2d(
    opName: string, parameter: string, heightOrWidth: number,
    filterSize: number, inputChannel: number, outputChannel: number,
    strides = 1, dilations = 1) {
  let sum = 0;
  const round = 100;
  const opFunc = opName === 'depthwiseConv2d' ? tf.depthwiseConv2d : tf.conv2d;

  // Ramp up.
  let x =
      tf.randomUniform(
          [1, heightOrWidth, heightOrWidth, inputChannel], 0, 100) as Tensor4D;
  let w = tf.randomUniform(
              [filterSize, filterSize, inputChannel, outputChannel], 0, 100) as
      Tensor4D;
  let res = opFunc(x, w, strides, 'same', 'NHWC', dilations);
  tf.dispose(x);
  tf.dispose(w);
  tf.dispose(res);

  for (let i = 0; i < round; i++) {
    const x = tf.randomUniform(
                  [1, heightOrWidth, heightOrWidth, inputChannel], 0, 100) as
        Tensor4D;
    const w = tf.randomUniform(
                  [filterSize, filterSize, inputChannel, outputChannel], 0,
                  100) as Tensor4D;

    // Upload and pack the inputs.
    let res = opFunc(x, w, strides, 'same', 'NHWC', dilations);
    tf.dispose(res);

    const profile = await tf.profile(() => {
      res = opFunc(x, w, strides, 'same', 'NHWC', dilations);
    });

    expect(profile.kernels[0].name)
        .toBe(
            opName === 'depthwiseConv2d' ? 'DepthwiseConv2dNative' : 'Conv2D');
    sum += profile.kernels[0].kernelTimeMs as number;

    tf.dispose(x);
    tf.dispose(w);
    tf.dispose(res);
  }

  // Log the benchmark result.
  console.log(
      `Time (ms) for ${opName}-${parameter} with ${heightOrWidth}-${
          filterSize}-${inputChannel}-${outputChannel}, Di${dilations}-St${
          strides}: `
          .padEnd(70) +
      `${sum / round}`);
}


describeWithFlags('Benchmark dense', ALL_ENVS, () => {
  describeWithFlags('Benchmark general conv2d with dense', ALL_ENVS, () => {
    const defaultHeightOrWidth = 196;
    const defaultFilterSize = 3;
    const defaultInputChannel = 32;
    const defaultOutputChannel = 32;
    const defaultStrides = 1;

    it('benchmark input channel', async () => {
      for (let inputChannel = 4; inputChannel <= 256; inputChannel *= 2) {
        await benchmarkConv2d(
            'conv2d', 'inputChannel', defaultHeightOrWidth, defaultFilterSize,
            inputChannel, defaultOutputChannel);
      }
    }, 100000000);

    it('benchmark output channel', async () => {
      for (let outputChannel = 4; outputChannel <= 256; outputChannel *= 2) {
        await benchmarkConv2d(
            'conv2d', 'outputChannel', defaultHeightOrWidth, defaultFilterSize,
            defaultInputChannel, outputChannel);
      }
    }, 100000000);

    it('benchmark image size', async () => {
      for (let imageSize = 16; imageSize <= 1024; imageSize *= 4) {
        await benchmarkConv2d(
            'conv2d', 'imageSize', imageSize, defaultFilterSize,
            defaultInputChannel, defaultOutputChannel);
      }
    }, 100000000);

    it('benchmark filter size', async () => {
      for (let filterSize = 1; filterSize <= 9; filterSize += 1) {
        await benchmarkConv2d(
            'conv2d', 'filterSize', defaultHeightOrWidth, filterSize,
            defaultInputChannel, defaultOutputChannel);
      }
    }, 100000000);

    it('benchmark strides', async () => {
      for (let strides = 1; strides <= 4; strides += 1) {
        await benchmarkConv2d(
            'conv2d', 'strides', defaultHeightOrWidth, defaultFilterSize,
            defaultInputChannel, defaultOutputChannel, strides);
      }
    }, 100000000);

    it('benchmark dilations', async () => {
      for (let dilations = 1; dilations <= 4; dilations += 1) {
        await benchmarkConv2d(
            'conv2d', 'dilations', defaultHeightOrWidth, defaultFilterSize,
            defaultInputChannel, defaultOutputChannel, defaultStrides,
            dilations);
      }
    }, 100000000);
  });

  describeWithFlags('Benchmark pointwise conv2d with dense', ALL_ENVS, () => {
    const defaultHeightOrWidth = 196;
    const defaultFilterSize = 1;
    const defaultInputChannel = 32;
    const defaultOutputChannel = 32;
    const defaultStrides = 1;

    it('benchmark input channel', async () => {
      for (let inputChannel = 4; inputChannel <= 256; inputChannel *= 2) {
        await benchmarkConv2d(
            'pointwiseConv2d', 'inputChannel', defaultHeightOrWidth,
            defaultFilterSize, inputChannel, defaultOutputChannel);
      }
    }, 100000000);

    it('benchmark output channel', async () => {
      for (let outputChannel = 4; outputChannel <= 256; outputChannel *= 2) {
        await benchmarkConv2d(
            'pointwiseConv2d', 'outputChannel', defaultHeightOrWidth,
            defaultFilterSize, defaultInputChannel, outputChannel);
      }
    }, 100000000);

    it('benchmark image size', async () => {
      for (let imageSize = 16; imageSize <= 1024; imageSize *= 4) {
        await benchmarkConv2d(
            'pointwiseConv2d', 'imageSize', imageSize, defaultFilterSize,
            defaultInputChannel, defaultOutputChannel);
      }
    }, 100000000);

    it('benchmark strides', async () => {
      for (let strides = 1; strides <= 4; strides += 1) {
        await benchmarkConv2d(
            'pointwiseConv2d', 'strides', defaultHeightOrWidth,
            defaultFilterSize, defaultInputChannel, defaultOutputChannel,
            strides);
      }
    }, 100000000);

    it('benchmark dilations', async () => {
      for (let dilations = 1; dilations <= 4; dilations += 1) {
        await benchmarkConv2d(
            'pointwiseConv2d', 'dilations', defaultHeightOrWidth,
            defaultFilterSize, defaultInputChannel, defaultOutputChannel,
            defaultStrides, dilations);
      }
    }, 100000000);
  });

  describeWithFlags('Benchmark depthwise conv2d with dense', ALL_ENVS, () => {
    const defaultHeightOrWidth = 196;
    const defaultFilterSize = 3;
    const defaultInputChannel = 32;
    const defaultMultiplier = 4;
    const defaultStrides = 1;

    it('benchmark input channel', async () => {
      for (let inputChannel = 4; inputChannel <= 256; inputChannel *= 2) {
        await benchmarkConv2d(
            'depthwiseConv2d', 'inputChannel', defaultHeightOrWidth,
            defaultFilterSize, inputChannel, defaultMultiplier);
      }
    }, 100000000);

    it('benchmark multiplier', async () => {
      for (let multiplier = 4; multiplier <= 256; multiplier *= 2) {
        await benchmarkConv2d(
            'depthwiseConv2d', 'multiplier', defaultHeightOrWidth,
            defaultFilterSize, defaultInputChannel, multiplier);
      }
    }, 100000000);

    it('benchmark image size', async () => {
      for (let imageSize = 16; imageSize <= 1024; imageSize *= 4) {
        await benchmarkConv2d(
            'depthwiseConv2d', 'imageSize', imageSize, defaultFilterSize,
            defaultInputChannel, defaultMultiplier);
      }
    }, 100000000);

    it('benchmark filter size', async () => {
      for (let filterSize = 1; filterSize <= 9; filterSize += 1) {
        await benchmarkConv2d(
            'depthwiseConv2d', 'filterSize', defaultHeightOrWidth, filterSize,
            defaultInputChannel, defaultMultiplier);
      }
    }, 100000000);

    it('benchmark strides', async () => {
      for (let strides = 1; strides <= 4; strides += 1) {
        await benchmarkConv2d(
            'depthwiseConv2d', 'strides', defaultHeightOrWidth,
            defaultFilterSize, defaultInputChannel, defaultMultiplier, strides);
      }
    }, 100000000);

    it('benchmark dilations', async () => {
      for (let dilations = 1; dilations <= 4; dilations += 1) {
        await benchmarkConv2d(
            'depthwiseConv2d', 'dilations', defaultHeightOrWidth,
            defaultFilterSize, defaultInputChannel, defaultMultiplier,
            defaultStrides, dilations);
      }
    }, 100000000);
  });
});
