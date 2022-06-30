/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
import {Tensor4D, test_util} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {ALL_ENVS, describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';
import {expectNumbersClose} from '@tensorflow/tfjs-core/dist/test_util';

const {expectArraysClose} = test_util;

describeWithFlags('Dense Conv2D WebGL Implementation', ALL_ENVS, () => {
  it('works for pointwise', async () => {
    const filter = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [1, 1, 4, 4]);
    const image = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [1, 1, 4, 4]);
    const result = tf.conv2d(image, filter, 1, 'valid');

    const resultData = tf.backend().readSync(result.dataId);
    const expected = [
      90, 100, 110, 120, 202, 228, 254, 280, 314, 356, 398, 440, 426, 484, 542,
      600
    ];
    expectArraysClose(resultData, expected);
  });

  it('works for random input values', async () => {
    let x = [
      14.53886890411377,  3.2612102031707764, 41.83704376220703,
      65.01011657714844,  74.32471466064453,  90.3241958618164,
      97.19178009033203,  72.43881225585938,  13.196929931640625,
      16.188322067260742, 37.91130065917969,  13.378096580505371,
      24.05510711669922,  51.85160446166992,  47.989566802978516,
      98.70687103271484,  38.842918395996094, 46.15006637573242,
      42.90564727783203,  62.361053466796875, 80.26527404785156,
      23.49817657470703,  8.765037536621094,  80.52574920654297,
      2.248786211013794,  94.95948791503906,  98.5092544555664,
      49.111351013183594, 62.8216667175293,   93.8804702758789,
      14.801081657409668, 46.94222640991211,  60.513309478759766,
      19.310476303100586, 89.20853424072266,  1.0774822235107422
    ];
    let w = [
      55.935569763183594, 22.930469512939453, 73.5508041381836,
      19.002504348754883, 65.49813079833984,  24.745664596557617,
      80.48580169677734,  43.38823318481445,  97.25115966796875,
      34.73556900024414,  21.946622848510742, 11.281351089477539,
      74.37937927246094,  11.655951499938965, 12.955927848815918,
      7.335716247558594,  42.38795471191406,  24.94944190979004,
      69.87342834472656,  96.22297668457031,  24.25912857055664,
      83.51610565185547,  29.456464767456055, 41.00996398925781,
      45.44337844848633,  89.85302734375,     38.99005126953125,
      44.157005310058594, 98.72321319580078,  8.593161582946777,
      16.10645294189453,  30.820934295654297, 91.03855895996094,
      86.76542663574219,  67.54839324951172,  1.7017613649368286,
      53.69328689575195,  48.975746154785156, 70.46859741210938,
      75.47567749023438,  78.28702545166016,  61.63998031616211,
      67.71986389160156,  94.63495635986328,  90.77911376953125,
      1.0160717964172363, 61.46479797363281,  38.49305725097656,
      55.37416458129883,  70.73543548583984,  80.8202133178711,
      60.7786750793457,   87.7402114868164,   6.762171268463135,
      49.950313568115234, 47.961212158203125, 20.8282527923584,
      23.973291397094727, 44.88302993774414,  84.17291259765625,
      72.31243133544922,  97.99678039550781,  38.954952239990234,
      98.84089660644531
    ];
    const expected = [
      56135.4765625, 39262.8203125, 40297.625, 45391.953125, 56455.203125,
      35638.84765625, 42706.16015625, 35691.8984375, 60515.08203125,
      34868.0390625, 43818.2421875, 46009.4609375, 49488.046875, 27358.39453125,
      40908.18359375, 37857.68359375
    ];

    const image = tf.tensor4d(x, [1, 3, 3, 4]);
    const filter = tf.tensor4d(w, [2, 2, 4, 4]);

    const result = tf.conv2d(image, filter, 1, 'valid');

    const resultData = tf.backend().readSync(result.dataId);
    expectArraysClose(resultData, expected);
  });

  it('x=[1,8,8,4] f=[3,3,4,4] s=[2,2] d=1 p=same', async () => {
    const inputDepth = 4;
    const xSize = 8;
    const inputShape: [number, number, number, number] =
        [1, xSize, xSize, inputDepth];
    const outputDepth = 4;
    const fSize = 3;
    const pad = 'same';
    const stride: [number, number] = [2, 2];

    const inputData = [];
    for (let i = 0; i < xSize * xSize * inputDepth; i++) {
      inputData.push(i % 5);
    }

    const wData = [];
    for (let i = 0; i < fSize * fSize * inputDepth * outputDepth; i++) {
      wData.push(i % 5);
    }

    const x = tf.tensor4d(inputData, inputShape);
    const w = tf.tensor4d(wData, [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.conv2d(x, w, stride, pad);
    expect(result.shape).toEqual([1, 4, 4, 4]);
    expectArraysClose(
        result.dataSync(), new Float32Array([
          140, 175, 175, 140, 140, 73,  146, 184, 175, 176, 142, 73,  46,
          94,  117, 125, 70,  144, 183, 187, 175, 142, 74,  146, 140, 175,
          175, 140, 98,  47,  96,  125, 140, 73,  146, 184, 175, 176, 142,
          73,  70,  144, 183, 187, 125, 100, 50,  100, 124, 98,  47,  96,
          96,  117, 113, 84,  98,  46,  94,  117, 84,  87,  60,  33
        ]));
  });

  it('x=[1,3,3,4] f=[1,1,4,4] s=[1, 1] d=1 p=valid', async () => {
    const filter = tf.tensor4d(
        [-1, 3, 2, 1, 3, 4, 4, -2, -1, 3, 2, 1, 3, 4, 4, -2], [1, 1, 4, 4]);
    const image = tf.tensor3d(
        [
          111, 112, 113, 114, 121, 122, 123, 124, 131, 132, 133, 134,
          211, 212, 213, 214, 221, 222, 223, 224, 231, 232, 233, 234,
          311, 312, 313, 314, 321, 322, 323, 324, 331, 332, 333, 334,

        ],
        [3, 3, 4]);

    tf.conv2d(image, filter, 1, 'valid');
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const result = tf.conv2d(image, filter, 1, 'valid');
    const resultData = result.dataSync();

    const expected = [
      454,  1576, 1352, -228, 494,  1716, 1472, -248, 534,  1856, 1592, -268,
      854,  2976, 2552, -428, 894,  3116, 2672, -448, 934,  3256, 2792, -468,
      1254, 4376, 3752, -628, 1294, 4516, 3872, -648, 1334, 4656, 3992, -668
    ];

    expectArraysClose(resultData, expected);
  });

  it('x=[1,8,8,4] f=[3,3,4,4] s=[2,2] d=1 p=same', async () => {
    const inputDepth = 4;
    const xSize = 8;
    const inputShape: [number, number, number, number] =
        [1, xSize, xSize, inputDepth];
    const outputDepth = 4;
    const fSize = 3;
    const pad = 'same';
    const stride: [number, number] = [2, 2];

    const inputData = [];
    for (let i = 0; i < xSize * xSize * inputDepth; i++) {
      inputData.push(i % 5);
    }

    const wData = [];
    for (let i = 0; i < fSize * fSize * inputDepth * outputDepth; i++) {
      wData.push(i % 5);
    }

    const x = tf.tensor4d(inputData, inputShape);
    const w = tf.tensor4d(wData, [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.conv2d(x, w, stride, pad);
    expect(result.shape).toEqual([1, 4, 4, 4]);
    expectArraysClose(
        result.dataSync(), new Float32Array([
          140, 175, 175, 140, 140, 73,  146, 184, 175, 176, 142, 73,  46,
          94,  117, 125, 70,  144, 183, 187, 175, 142, 74,  146, 140, 175,
          175, 140, 98,  47,  96,  125, 140, 73,  146, 184, 175, 176, 142,
          73,  70,  144, 183, 187, 125, 100, 50,  100, 124, 98,  47,  96,
          96,  117, 113, 84,  98,  46,  94,  117, 84,  87,  60,  33
        ]));
  });

  it('x=[1,4,4,4] f=[3,3,4,4] s=[1, 1] d=1 p=same', async () => {
    const inputDepth = 4;
    const xSize = 4;
    const inputShape: [number, number, number, number] =
        [1, xSize, xSize, inputDepth];
    const outputDepth = 4;
    const fSize = 3;
    const pad = 'same';
    const stride: [number, number] = [1, 1];
    const dataFormat = 'NHWC';
    const dilation = 2;

    const inputData = [];
    for (let i = 0; i < xSize * xSize * inputDepth; i++) {
      inputData.push(i % 5);
    }

    const wData = [];
    for (let i = 0; i < fSize * fSize * inputDepth * outputDepth; i++) {
      wData.push(i % 5);
    }

    const x = tf.tensor4d(inputData, inputShape);
    const w = tf.tensor4d(wData, [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.conv2d(x, w, stride, pad, dataFormat, dilation);
    expect(result.shape).toEqual([1, 4, 4, 4]);
    expectArraysClose(
        result.dataSync(), new Float32Array([
          39, 48, 72, 71, 77, 55, 48, 71, 60, 39, 48, 72, 79, 77, 55, 48,
          51, 76, 91, 76, 39, 48, 72, 71, 56, 51, 76, 91, 60, 39, 48, 72,
          72, 71, 60, 39, 48, 71, 79, 77, 48, 72, 71, 60, 55, 48, 71, 79,
          91, 76, 56, 51, 72, 71, 60, 39, 76, 91, 76, 56, 48, 72, 71, 60
        ]));
  });

  it('image is packed and isChannelLast.', async () => {
    const filter = tf.tensor4d(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 4, 4]);
    const image = tf.tensor3d(
        [
          11, 12, 13, 21, 22, 23, 31, 32, 33, 11, 12, 13,
          21, 22, 23, 31, 32, 33, 11, 12, 13, 21, 22, 23,
          31, 32, 33, 11, 12, 13, 21, 22, 23, 31, 32, 33
        ],
        [3, 3, 4]);

    tf.conv2d(image, filter, 1, 'valid');
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const result = tf.conv2d(image, filter, 1, 'valid');
    const resultData = result.dataSync();

    const expected = [
      57,  57,  57,  57,  108, 108, 108, 108, 69,  69,  69,  69,
      97,  97,  97,  97,  88,  88,  88,  88,  79,  79,  79,  79,
      107, 107, 107, 107, 68,  68,  68,  68,  119, 119, 119, 119
    ];

    expectArraysClose(resultData, expected);
  });

  it('works for large inputs', async () => {
    let x = [] as number[];
    let w = [] as number[];
    for (let i = 0; i < 196 * 196 * 32; i++) {
      x.push(i - 196 * 196 * 16);
    }
    for (let i = 0; i < 9 * 32 * 32; i++) {
      w.push(i - 9 * 32 * 16);
    }
    const image = tf.tensor4d(x, [1, 196, 196, 32]);
    const filter = tf.tensor4d(w, [3, 3, 32, 32]);

    const result = tf.conv2d(image, filter, 1, 'valid');

    const resultData =
        tf.backend().readSync(result.dataId) as unknown as number[];
    let sum = 0;
    for (let num of resultData) {
      sum += num;
    }

    const cpuResultSum = 4463875743285248;
    expectNumbersClose(sum, cpuResultSum, cpuResultSum * 0.000001);
  });
});

async function benchmarkConv2d(
    type: string, heightOrWidth: number, filterSize: number,
    inputChannel: number, outputChannel: number, dilation = 1, strides = 1) {
  let sum = 0;
  const round = 100;

  // Ramp up.
  let x =
      tf.randomUniform(
          [1, heightOrWidth, heightOrWidth, inputChannel], 0, 100) as Tensor4D;
  let w = tf.randomUniform(
              [filterSize, filterSize, inputChannel, outputChannel], 0, 100) as
      Tensor4D;
  let res = tf.conv2d(x, w, strides, 'same', 'NHWC', dilation);
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
    let res = tf.conv2d(x, w, strides, 'same', 'NHWC', dilation);
    tf.dispose(res);

    const profile = await tf.profile(() => {
      res = tf.conv2d(x, w, strides, 'same', 'NHWC', dilation);
    });

    expect(profile.kernels[0].name).toBe('Conv2D');
    sum += profile.kernels[0].kernelTimeMs as number;

    tf.dispose(x);
    tf.dispose(w);
    tf.dispose(res);
  }
  console.log(`Benchmark ${type} result for ${heightOrWidth}-${filterSize}-${
      inputChannel}-${outputChannel}: ${sum / round}ms`);
}

describeWithFlags(
    'Benchmark pointwise conv2d with dense and dilations', ALL_ENVS, () => {
      const defaultHeightOrWidth = 196;
      const defaultFilterSize = 1;
      const defaultInputChannel = 32;
      const defaultOutputChannel = 32;

      it('benchmark input channel', async () => {
        for (let inputChannel = 4; inputChannel <= 256; inputChannel *= 2) {
          await benchmarkConv2d(
              'inputChannel', defaultHeightOrWidth, defaultFilterSize,
              inputChannel, defaultOutputChannel, 2);
        }
      }, 100000000);

      it('benchmark output channel', async () => {
        for (let outputChannel = 4; outputChannel <= 256; outputChannel *= 2) {
          await benchmarkConv2d(
              'outputChannel', defaultHeightOrWidth, defaultFilterSize,
              defaultInputChannel, outputChannel, 2);
        }
      }, 100000000);

      it('benchmark image size', async () => {
        for (let imageSize = 16; imageSize <= 1024; imageSize *= 4) {
          await benchmarkConv2d(
              'imageSize', imageSize, defaultFilterSize, defaultInputChannel,
              defaultOutputChannel, 2);
        }
      }, 100000000);
    });

describeWithFlags(
    'Benchmark pointwise conv2d with dense and strides', ALL_ENVS, () => {
      const defaultHeightOrWidth = 196;
      const defaultFilterSize = 1;
      const defaultInputChannel = 32;
      const defaultOutputChannel = 32;

      it('benchmark input channel', async () => {
        for (let inputChannel = 4; inputChannel <= 256; inputChannel *= 2) {
          await benchmarkConv2d(
              'inputChannel', defaultHeightOrWidth, defaultFilterSize,
              inputChannel, defaultOutputChannel, 1, 2);
        }
      }, 100000000);

      it('benchmark output channel', async () => {
        for (let outputChannel = 4; outputChannel <= 256; outputChannel *= 2) {
          await benchmarkConv2d(
              'outputChannel', defaultHeightOrWidth, defaultFilterSize,
              defaultInputChannel, outputChannel, 1, 2);
        }
      }, 100000000);

      it('benchmark image size', async () => {
        for (let imageSize = 16; imageSize <= 1024; imageSize *= 4) {
          await benchmarkConv2d(
              'imageSize', imageSize, defaultFilterSize, defaultInputChannel,
              defaultOutputChannel, 1, 2);
        }
      }, 100000000);
    });
