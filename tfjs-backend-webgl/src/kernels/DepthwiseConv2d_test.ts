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

const {expectArraysClose, expectNumbersClose} = test_util;

describeWithFlags(
    'Dense depthwise Conv2D WebGL Implementation', ALL_ENVS, () => {
      it('works for depthwise', async () => {
        const image = tf.tensor4d(
            [
              1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
              13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
              25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36
            ],
            [1, 3, 3, 4]);
        const filter = tf.tensor4d(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            [1, 1, 4, 4]);
        const result = tf.depthwiseConv2d(image, filter, 1, 'valid');

        const resultData = tf.backend().readSync(result.dataId);
        const expected = [
          1,   2,   3,   4,   10,  12,  14,  16,  27,  30,  33,  36,  52,  56,
          60,  64,  5,   10,  15,  20,  30,  36,  42,  48,  63,  70,  77,  84,
          104, 112, 120, 128, 9,   18,  27,  36,  50,  60,  70,  80,  99,  110,
          121, 132, 156, 168, 180, 192, 13,  26,  39,  52,  70,  84,  98,  112,
          135, 150, 165, 180, 208, 224, 240, 256, 17,  34,  51,  68,  90,  108,
          126, 144, 171, 190, 209, 228, 260, 280, 300, 320, 21,  42,  63,  84,
          110, 132, 154, 176, 207, 230, 253, 276, 312, 336, 360, 384, 25,  50,
          75,  100, 130, 156, 182, 208, 243, 270, 297, 324, 364, 392, 420, 448,
          29,  58,  87,  116, 150, 180, 210, 240, 279, 310, 341, 372, 416, 448,
          480, 512, 33,  66,  99,  132, 170, 204, 238, 272, 315, 350, 385, 420,
          468, 504, 540, 576
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
          8304.548828125,   7022.46533203125,   11026.8447265625,
          9829.77734375,    9238.0888671875,    10475.7705078125,
          8882.232421875,   9972.638671875,     13136.0341796875,
          14172.875,        9883.2763671875,    12916.66015625,
          25456.80859375,   7591.708984375,     10505.2685546875,
          12672.87890625,   12697.611328125,    11081.37890625,
          15499.5908203125, 7626.7216796875,    10848.462890625,
          6006.25048828125, 12172.5380859375,   9193.09765625,
          14716.34765625,   9637.287109375,     6910.1611328125,
          7568.65673828125, 18192.77734375,     8913.9306640625,
          8123.8720703125,  11303.4208984375,   6675.431640625,
          6159.5390625,     9712.5234375,       8016.73291015625,
          17851.5234375,    10422.923828125,    16913.7578125,
          15812.115234375,  14637.0947265625,   11949.0869140625,
          10061.453125,     13004.240234375,    21351.029296875,
          6336.48876953125, 7130.50634765625,   9176.369140625,
          14645.0498046875, 12624.443359375,    17599.5390625,
          12246.302734375,  10327.8447265625,   7832.93701171875,
          11986.787109375,  10977.875,          7587.7275390625,
          5328.8779296875,  6289.65966796875,   9780.7138671875,
          16927.42578125,   1572.1346435546875, 5032.19677734375,
          4852.791015625
        ];

        const image = tf.tensor4d(x, [1, 3, 3, 4]);
        const filter = tf.tensor4d(w, [2, 2, 4, 4]);

        const result = tf.depthwiseConv2d(image, filter, 1, 'valid');

        const resultData = tf.backend().readSync(result.dataId);
        expectArraysClose(result.shape, [1, 2, 2, 16]);
        expectArraysClose(resultData, expected);
      });

      it('x=[1,8,8,4] f=[3,3,4,4] s=[2,2] d=1 p=same', async () => {
        const inputDepth = 4;
        const xSize = 8;
        const inputShape: [number, number, number, number] =
            [1, xSize, xSize, inputDepth];
        const channel_multiplier = 4;
        const fSize = 3;
        const pad = 'same';
        const stride: [number, number] = [2, 2];

        const inputData = [];
        for (let i = 0; i < xSize * xSize * inputDepth; i++) {
          inputData.push(i % 5);
        }

        const wData = [];
        for (let i = 0; i < fSize * fSize * inputDepth * channel_multiplier;
             i++) {
          wData.push(i % 5);
        }

        const x = tf.tensor4d(inputData, inputShape);
        const w =
            tf.tensor4d(wData, [fSize, fSize, inputDepth, channel_multiplier]);

        const result = tf.depthwiseConv2d(x, w, stride, pad);
        expect(result.shape).toEqual([1, 4, 4, 16]);
        expectArraysClose(
            result.dataSync(), new Float32Array([
              36, 50, 49, 38, 34, 42, 50, 38, 34, 41, 38, 40, 36, 42, 38, 24,
              24, 20, 36, 42, 40, 20, 40, 50, 38, 17, 36, 50, 38, 16, 34, 42,
              42, 50, 38, 16, 41, 38, 40, 17, 42, 38, 24, 20, 50, 50, 40, 20,
              13, 26, 34, 37, 10, 24, 33, 37, 10, 20, 25, 25, 13, 24, 25, 26,
              20, 40, 50, 50, 17, 36, 50, 49, 16, 34, 42, 50, 17, 34, 41, 38,
              38, 40, 17, 34, 38, 24, 20, 36, 50, 40, 20, 40, 49, 38, 17, 36,
              36, 50, 49, 38, 34, 42, 50, 38, 34, 41, 38, 40, 36, 42, 38, 24,
              22, 14, 26, 33, 20, 13, 26, 34, 36, 10, 24, 33, 20, 10, 20, 25,
              24, 20, 36, 42, 40, 20, 40, 50, 38, 17, 36, 50, 38, 16, 34, 42,
              42, 50, 38, 16, 41, 38, 40, 17, 42, 38, 24, 20, 50, 50, 40, 20,
              20, 40, 50, 50, 17, 36, 50, 49, 16, 34, 42, 50, 17, 34, 41, 38,
              26, 22, 13, 24, 25, 22, 14, 26, 37, 20, 13, 26, 37, 36, 10, 24,
              25, 22, 14, 26, 37, 20, 13, 26, 37, 36, 10, 24, 25, 20, 10, 20,
              20, 25, 25, 20, 24, 25, 26, 22, 26, 33, 25, 22, 26, 34, 37, 20,
              20, 13, 26, 34, 36, 10, 24, 33, 20, 10, 20, 25, 22, 13, 24, 25,
              17, 13, 4,  10, 25, 25, 20, 10, 25, 24, 18, 7,  17, 25, 18, 6
            ]));
      });

      it('x=[1,3,3,4] f=[1,1,4,4] s=[1, 1] d=1 p=valid', async () => {
        const filter = tf.tensor4d(
            [-1, 3, 2, 1, 3, 4, 4, -2, -1, 3, 2, 1, 3, 4, 4, -2], [1, 1, 4, 4]);
        const image = tf.tensor4d(
            [
              111, 112, 113, 114, 121, 122, 123, 124, 131, 132, 133, 134,
              211, 212, 213, 214, 221, 222, 223, 224, 231, 232, 233, 234,
              311, 312, 313, 314, 321, 322, 323, 324, 331, 332, 333, 334,

            ],
            [1, 3, 3, 4]);

        // tslint:disable-next-line: no-unnecessary-type-assertion
        const result = tf.depthwiseConv2d(image, filter, 1, 'valid');
        const resultData = result.dataSync();

        const expected = [
          -111, 333,  222,  111,  336,  448,  448,  -224, -113, 339,  226,
          113,  342,  456,  456,  -228, -121, 363,  242,  121,  366,  488,
          488,  -244, -123, 369,  246,  123,  372,  496,  496,  -248, -131,
          393,  262,  131,  396,  528,  528,  -264, -133, 399,  266,  133,
          402,  536,  536,  -268, -211, 633,  422,  211,  636,  848,  848,
          -424, -213, 639,  426,  213,  642,  856,  856,  -428, -221, 663,
          442,  221,  666,  888,  888,  -444, -223, 669,  446,  223,  672,
          896,  896,  -448, -231, 693,  462,  231,  696,  928,  928,  -464,
          -233, 699,  466,  233,  702,  936,  936,  -468, -311, 933,  622,
          311,  936,  1248, 1248, -624, -313, 939,  626,  313,  942,  1256,
          1256, -628, -321, 963,  642,  321,  966,  1288, 1288, -644, -323,
          969,  646,  323,  972,  1296, 1296, -648, -331, 993,  662,  331,
          996,  1328, 1328, -664, -333, 999,  666,  333,  1002, 1336, 1336,
          -668
        ];

        expect(result.shape).toEqual([1, 3, 3, 16]);
        expectArraysClose(resultData, expected);
      });

      it('x=[1,8,8,4] f=[3,3,4,4] s=[2,2] d=1 p=same', async () => {
        const inputDepth = 4;
        const xSize = 8;
        const inputShape: [number, number, number, number] =
            [1, xSize, xSize, inputDepth];
        const channel_multiplier = 4;
        const fSize = 3;
        const pad = 'same';
        const stride: [number, number] = [2, 2];

        const inputData = [];
        for (let i = 0; i < xSize * xSize * inputDepth; i++) {
          inputData.push(i % 5);
        }

        const wData = [];
        for (let i = 0; i < fSize * fSize * inputDepth * channel_multiplier;
             i++) {
          wData.push(i % 5);
        }

        const x = tf.tensor4d(inputData, inputShape);
        const w =
            tf.tensor4d(wData, [fSize, fSize, inputDepth, channel_multiplier]);

        const result = tf.depthwiseConv2d(x, w, stride, pad);
        expect(result.shape).toEqual([1, 4, 4, 16]);
        expectArraysClose(
            result.dataSync(), new Float32Array([
              36, 50, 49, 38, 34, 42, 50, 38, 34, 41, 38, 40, 36, 42, 38, 24,
              24, 20, 36, 42, 40, 20, 40, 50, 38, 17, 36, 50, 38, 16, 34, 42,
              42, 50, 38, 16, 41, 38, 40, 17, 42, 38, 24, 20, 50, 50, 40, 20,
              13, 26, 34, 37, 10, 24, 33, 37, 10, 20, 25, 25, 13, 24, 25, 26,
              20, 40, 50, 50, 17, 36, 50, 49, 16, 34, 42, 50, 17, 34, 41, 38,
              38, 40, 17, 34, 38, 24, 20, 36, 50, 40, 20, 40, 49, 38, 17, 36,
              36, 50, 49, 38, 34, 42, 50, 38, 34, 41, 38, 40, 36, 42, 38, 24,
              22, 14, 26, 33, 20, 13, 26, 34, 36, 10, 24, 33, 20, 10, 20, 25,
              24, 20, 36, 42, 40, 20, 40, 50, 38, 17, 36, 50, 38, 16, 34, 42,
              42, 50, 38, 16, 41, 38, 40, 17, 42, 38, 24, 20, 50, 50, 40, 20,
              20, 40, 50, 50, 17, 36, 50, 49, 16, 34, 42, 50, 17, 34, 41, 38,
              26, 22, 13, 24, 25, 22, 14, 26, 37, 20, 13, 26, 37, 36, 10, 24,
              25, 22, 14, 26, 37, 20, 13, 26, 37, 36, 10, 24, 25, 20, 10, 20,
              20, 25, 25, 20, 24, 25, 26, 22, 26, 33, 25, 22, 26, 34, 37, 20,
              20, 13, 26, 34, 36, 10, 24, 33, 20, 10, 20, 25, 22, 13, 24, 25,
              17, 13, 4,  10, 25, 25, 20, 10, 25, 24, 18, 7,  17, 25, 18, 6
            ]));
      });

      it('x=[1,4,4,4] f=[3,3,4,4] s=[1, 1] d=1 p=same', async () => {
        const inputDepth = 4;
        const xSize = 4;
        const inputShape: [number, number, number, number] =
            [1, xSize, xSize, inputDepth];
        const channel_multiplier = 4;
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
        for (let i = 0; i < fSize * fSize * inputDepth * channel_multiplier;
             i++) {
          wData.push(i % 5);
        }

        const x = tf.tensor4d(inputData, inputShape);
        const w =
            tf.tensor4d(wData, [fSize, fSize, inputDepth, channel_multiplier]);

        const result =
            tf.depthwiseConv2d(x, w, stride, pad, dataFormat, dilation);
        expect(result.shape).toEqual([1, 4, 4, 16]);
        expectArraysClose(
            result.dataSync(), new Float32Array([
              4,  9,  14, 9,  24, 13, 17, 21, 6,  14, 22, 20, 5,  12, 19, 21,
              30, 21, 12, 18, 14, 4,  9,  14, 15, 24, 13, 17, 18, 6,  14, 22,
              14, 4,  9,  14, 15, 24, 13, 17, 18, 6,  14, 22, 13, 5,  12, 19,
              29, 30, 21, 12, 9,  14, 4,  9,  21, 15, 24, 13, 20, 18, 6,  14,
              13, 17, 21, 15, 14, 22, 20, 18, 12, 19, 21, 13, 12, 18, 29, 30,
              4,  9,  14, 9,  24, 13, 17, 21, 6,  14, 22, 20, 5,  12, 19, 21,
              24, 13, 17, 21, 6,  14, 22, 20, 5,  12, 19, 21, 21, 12, 18, 29,
              14, 4,  9,  14, 15, 24, 13, 17, 18, 6,  14, 22, 13, 5,  12, 19,
              14, 9,  14, 4,  17, 21, 15, 24, 22, 20, 18, 6,  19, 21, 13, 5,
              12, 18, 29, 30, 9,  14, 9,  14, 13, 17, 21, 15, 14, 22, 20, 18,
              9,  14, 9,  14, 13, 17, 21, 15, 14, 22, 20, 18, 12, 19, 21, 13,
              21, 12, 18, 29, 4,  9,  14, 9,  24, 13, 17, 21, 6,  14, 22, 20,
              21, 15, 24, 13, 20, 18, 6,  14, 21, 13, 5,  12, 29, 30, 21, 12,
              14, 9,  14, 4,  17, 21, 15, 24, 22, 20, 18, 6,  19, 21, 13, 5,
              17, 21, 15, 24, 22, 20, 18, 6,  19, 21, 13, 5,  18, 29, 30, 21,
              9,  14, 9,  14, 13, 17, 21, 15, 14, 22, 20, 18, 12, 19, 21, 13
            ]));
      });

      it('image is packed and isChannelLast.', async () => {
        const filter = tf.tensor4d(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 4, 4]);
        const image = tf.tensor4d(
            [
              11, 12, 13, 21, 22, 23, 31, 32, 33, 11, 12, 13,
              21, 22, 23, 31, 32, 33, 11, 12, 13, 21, 22, 23,
              31, 32, 33, 11, 12, 13, 21, 22, 23, 31, 32, 33
            ],
            [1, 3, 3, 4]);

        const result = tf.depthwiseConv2d(image, filter, 1, 'valid');
        const resultData = result.dataSync();

        const expected = [
          11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 21, 21, 21, 21,
          22, 22, 22, 22, 23, 23, 23, 23, 31, 31, 31, 31, 32, 32, 32, 32,
          33, 33, 33, 33, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13,
          21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 31, 31, 31, 31,
          32, 32, 32, 32, 33, 33, 33, 33, 11, 11, 11, 11, 12, 12, 12, 12,
          13, 13, 13, 13, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23,
          31, 31, 31, 31, 32, 32, 32, 32, 33, 33, 33, 33, 11, 11, 11, 11,
          12, 12, 12, 12, 13, 13, 13, 13, 21, 21, 21, 21, 22, 22, 22, 22,
          23, 23, 23, 23, 31, 31, 31, 31, 32, 32, 32, 32, 33, 33, 33, 33
        ];

        expect(result.shape).toEqual([1, 3, 3, 16]);
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

        const result = tf.depthwiseConv2d(image, filter, 1, 'valid');

        const resultData =
            tf.backend().readSync(result.dataId) as unknown as number[];
        let sum = 0;
        for (let num of resultData) {
          sum += num;
        }

        const cpuResultSum = 4463874313941890;
        expectNumbersClose(sum, cpuResultSum, cpuResultSum * 0.0000001);
      });
    });


describeWithFlags('Benchmark dense depthwise', ALL_ENVS, () => {
  async function benchmarkDepthwise(
      type: string, heightOrWidth: number, filterSize: number,
      inputChannel: number, outputChannel: number, strides = 1, dilation = 1) {
    let sum = 0;
    const round = 100;

    // Ramp up.
    let x = tf.randomUniform(
                [1, heightOrWidth, heightOrWidth, inputChannel], 0, 100) as
        Tensor4D;
    let w = tf.randomUniform(
                [filterSize, filterSize, inputChannel, outputChannel], 0,
                100) as Tensor4D;
    let res = tf.depthwiseConv2d(x, w, strides, 'same');
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
      let res;
      const profile = await tf.profile(() => {
        res = tf.depthwiseConv2d(x, w, strides, 'same');
      });
      sum += profile.kernels[0].kernelTimeMs as number;
      tf.dispose(x);
      tf.dispose(w);
      tf.dispose(res);
    }
    console.log(`Benchmark ${type} result for ${heightOrWidth}-${filterSize}-${
        inputChannel}-${outputChannel}: ${sum / round}ms`);
  }

  const defaultHeightOrWidth = 196;
  const defaultFilterSize = 3;
  const defaultInputChannel = 32;
  const defaultOutputChannel = 32;

  it('benchmark input channel', async () => {
    for (let inputChannel = 4; inputChannel <= 256; inputChannel *= 2) {
      await benchmarkDepthwise(
          'inputChannel', defaultHeightOrWidth, defaultFilterSize, inputChannel,
          defaultOutputChannel);
    }
  }, 100000000);

  it('benchmark output channel', async () => {
    for (let outputChannel = 256; outputChannel <= 256; outputChannel *= 2) {
      await benchmarkDepthwise(
          'outputChannel', defaultHeightOrWidth, defaultFilterSize,
          defaultInputChannel, outputChannel);
    }
  }, 100000000);

  it('benchmark image size', async () => {
    for (let imageSize = 16; imageSize <= 1024; imageSize *= 4) {
      await benchmarkDepthwise(
          'imageSize', imageSize, defaultFilterSize, defaultInputChannel,
          defaultOutputChannel);
    }
  }, 100000000);

  it('benchmark filter size', async () => {
    for (let filterSize = 1; filterSize <= 9; filterSize += 1) {
      await benchmarkDepthwise(
          'filterSize', defaultHeightOrWidth, filterSize, defaultInputChannel,
          defaultOutputChannel);
    }
  }, 100000000);
});
