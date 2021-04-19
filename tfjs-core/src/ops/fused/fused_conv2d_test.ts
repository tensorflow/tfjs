/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import * as tf from '../../index';
import {ALL_ENVS, describeWithFlags} from '../../jasmine_util';
import {expectArraysClose} from '../../test_util';

function generateCaseInputs(totalSizeTensor: number, totalSizeFilter: number) {
  const inp = new Array(totalSizeTensor);
  const filt = new Array(totalSizeFilter);

  for (let i = 0; i < totalSizeTensor; i++) {
    inp[i] = i * 0.001 - totalSizeTensor * 0.001 / 2;
  }
  for (let i = 0; i < totalSizeFilter; i++) {
    const sign = i % 2 === 0 ? -1 : 1;
    filt[i] = i * 0.001 * sign;
  }

  return {input: inp, filter: filt};
}

describeWithFlags('fused conv2d', ALL_ENVS, () => {
  it('basic', async () => {
    const inputDepth = 2;
    const inShape: [number, number, number, number] = [2, 2, 2, inputDepth];
    const outputDepth = 2;
    const fSize = 1;
    const pad = 0;
    const stride = 1;

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], inShape);
    const w =
        tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.fused.conv2d({x, filter: w, strides: stride, pad});
    expect(result.shape).toEqual([2, 2, 2, 2]);
    const expected =
        [-5, 2, -11, 5, -17, 8, -23, 11, -29, 14, -35, 17, -41, 20, -47, 23];

    expectArraysClose(await result.data(), expected);
  });

  it('basic with relu', async () => {
    const inputDepth = 2;
    const inShape: [number, number, number, number] = [2, 2, 2, inputDepth];
    const outputDepth = 2;
    const fSize = 1;
    const pad = 0;
    const stride = 1;

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], inShape);
    const w =
        tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.fused.conv2d({
      x,
      filter: w,
      strides: stride,
      pad,
      dataFormat: 'NHWC',
      dilations: [1, 1],
      activation: 'relu'
    });
    expect(result.shape).toEqual([2, 2, 2, 2]);
    const expected = [0, 2, 0, 5, 0, 8, 0, 11, 0, 14, 0, 17, 0, 20, 0, 23];

    expectArraysClose(await result.data(), expected);
  });

  it('relu with stride 2 x=[1,8,8,16] f=[3,3,16,1] s=[2,2] d=1 p=same',
     async () => {
       const inputDepth = 16;
       const xSize = 8;
       const inputShape: [number, number, number, number] =
           [1, xSize, xSize, inputDepth];
       const outputDepth = 1;
       const fSize = 3;
       const pad = 'same';
       const stride: [number, number] = [2, 2];

       // TODO(annxingyuan): Make this test work with large inputs
       // https://github.com/tensorflow/tfjs/issues/3143
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

       const result = tf.fused.conv2d({
         x,
         filter: w,
         strides: stride,
         pad,
         dataFormat: 'NHWC',
         dilations: [1, 1],
         activation: 'relu'
       });
       expect(result.shape).toEqual([1, 4, 4, 1]);
       expectArraysClose(await result.data(), new Float32Array([
                           854, 431, 568, 382, 580, 427, 854, 288, 431, 568,
                           580, 289, 285, 570, 285, 258
                         ]));
     });

  it('relu bias stride 2 x=[1,8,8,16] f=[3,3,16,1] s=[2,2] d=8 p=same',
     async () => {
       const inputDepth = 16;
       const xSize = 8;
       const inputShape: [number, number, number, number] =
           [1, xSize, xSize, inputDepth];
       const outputDepth = 8;
       const fSize = 3;
       const pad = 'same';
       const stride: [number, number] = [2, 2];

       const inputs = generateCaseInputs(
           1 * xSize * xSize * inputDepth,
           fSize * fSize * inputDepth * outputDepth);
       const x = tf.tensor4d(inputs.input, inputShape);
       const w =
           tf.tensor4d(inputs.filter, [fSize, fSize, inputDepth, outputDepth]);
       const bias = tf.tensor1d([1, 4, 2, 3, 9, 6, 5, 8]);
       const result = tf.fused.conv2d({
         x,
         filter: w,
         strides: stride,
         pad,
         dataFormat: 'NHWC',
         dilations: [1, 1],
         activation: 'relu',
         bias
       });
       expect(result.shape).toEqual([1, 4, 4, 8]);
       expectArraysClose(await result.data(), new Float32Array([
                           25.75398063659668,
                           0,
                           26.857805252075195,
                           0,
                           33.961631774902344,
                           0,
                           30.065458297729492,
                           0,
                           23.118206024169922,
                           0,
                           24.212820053100586,
                           0,
                           31.307422637939453,
                           0,
                           27.402034759521484,
                           0,
                           20.482431411743164,
                           0,
                           21.567821502685547,
                           0,
                           28.653217315673828,
                           0,
                           24.73861312866211,
                           0,
                           11.078080177307129,
                           0,
                           12.130399703979492,
                           0,
                           19.182720184326172,
                           0,
                           15.235037803649902,
                           0,
                           4.6677775382995605,
                           0.31717729568481445,
                           5.697869777679443,
                           0,
                           12.727968215942383,
                           2.2569849491119385,
                           8.758066177368164,
                           4.226885795593262,
                           2.0319995880126953,
                           2.9575586318969727,
                           3.052880048751831,
                           1.9366796016693115,
                           10.073760032653809,
                           4.915799617767334,
                           6.094639778137207,
                           6.89492130279541,
                           0,
                           5.5979437828063965,
                           0.4078875780105591,
                           4.586280822753906,
                           7.419551849365234,
                           7.5746169090271,
                           3.43121600151062,
                           9.562952041625977,
                           0,
                           6.404943943023682,
                           0,
                           5.401776313781738,
                           6.5998077392578125,
                           8.398608207702637,
                           2.602976083755493,
                           10.395440101623535,
                           0,
                           21.440250396728516,
                           0,
                           20.483882904052734,
                           0,
                           23.527509689331055,
                           0,
                           25.571144104003906,
                           0,
                           24.080629348754883,
                           0,
                           23.133480072021484,
                           0,
                           26.186328887939453,
                           0,
                           28.239177703857422,
                           0,
                           26.721012115478516,
                           0,
                           25.783079147338867,
                           0,
                           28.84514808654785,
                           0,
                           30.907209396362305,
                           0,
                           18.914127349853516,
                           0,
                           17.960111618041992,
                           0,
                           21.006093978881836,
                           0,
                           23.052082061767578,
                           0,
                           17.89089584350586,
                           0,
                           16.95684814453125,
                           0,
                           20.022798538208008,
                           0,
                           22.088754653930664,
                           0,
                           19.06132698059082,
                           0,
                           18.133424758911133,
                           0,
                           21.205520629882812,
                           0,
                           23.27761459350586,
                           0,
                           20.23175811767578,
                           0,
                           19.309999465942383,
                           0,
                           22.388240814208984,
                           0,
                           24.46647834777832,
                           0,
                           13.584352493286133,
                           0,
                           12.6395845413208,
                           0,
                           15.694815635681152,
                           0,
                           17.750045776367188
                         ]));
     });

  it('prelu bias stride 2 x=[1,8,8,16] f=[3,3,16,1] s=[2,2] d=8 p=same',
     async () => {
       const inputDepth = 16;
       const xSize = 8;
       const inputShape: [number, number, number, number] =
           [1, xSize, xSize, inputDepth];
       const outputDepth = 8;
       const fSize = 3;
       const pad = 'same';
       const stride: [number, number] = [2, 2];

       const inputs = generateCaseInputs(
           1 * xSize * xSize * inputDepth,
           fSize * fSize * inputDepth * outputDepth);
       const x = tf.tensor4d(inputs.input, inputShape);
       const w =
           tf.tensor4d(inputs.filter, [fSize, fSize, inputDepth, outputDepth]);
       const bias = tf.tensor1d([1, 4, 2, 3, 9, 6, 5, 8]);
       const preluActivationWeights = tf.tensor1d([1, 2, 3, 4, 5, 6, 7, 8]);

       const result = tf.fused.conv2d({
         x,
         filter: w,
         strides: stride,
         pad,
         dataFormat: 'NHWC',
         dilations: [1, 1],
         activation: 'prelu',
         preluActivationWeights,
         bias
       });
       expect(result.shape).toEqual([1, 4, 4, 8]);
       expectArraysClose(
           await result.data(), new Float32Array([
             25.75398063659668,   -41.61178970336914,  26.857805252075195,
             -87.63885498046875,  33.961631774902344,  -114.0812759399414,
             30.065458297729492,  -136.93893432617188, 23.118206024169922,
             -36.33102035522461,  24.212820053100586,  -77.04048156738281,
             31.307422637939453,  -98.12835693359375,  27.402034759521484,
             -115.5947265625,     20.482431411743164,  -31.050262451171875,
             21.567821502685547,  -66.44209289550781,  28.653217315673828,
             -82.17544555664062,  24.73861312866211,   -94.25041198730469,
             11.078080177307129,  -12.208478927612305, 12.130399703979492,
             -28.626232147216797, 19.182720184326172,  -25.253299713134766,
             15.235037803649902,  -18.08960723876953,  4.6677775382995605,
             0.31717729568481445, 5.697869777679443,   -2.8516759872436523,
             12.727968215942383,  2.2569849491119385,  8.758066177368164,
             4.226885795593262,   2.0319995880126953,  2.9575586318969727,
             3.052880048751831,   1.9366796016693115,  10.073760032653809,
             4.915799617767334,   6.094639778137207,   6.89492130279541,
             -0.6037763357162476, 5.5979437828063965,  0.4078875780105591,
             4.586280822753906,   7.419551849365234,   7.5746169090271,
             3.43121600151062,    9.562952041625977,   -1.4065279960632324,
             6.404943943023682,   -1.2100803852081299, 5.401776313781738,
             6.5998077392578125,  8.398608207702637,   2.602976083755493,
             10.395440101623535,  -16.418434143066406, 21.440250396728516,
             -46.38618850708008,  20.483882904052734,  -42.52848815917969,
             23.527509689331055,  -87.84530639648438,  25.571144104003906,
             -19.054208755493164, 24.080629348754883,  -54.32115936279297,
             23.133480072021484,  -55.79951477050781,  26.186328887939453,
             -106.48924255371094, 28.239177703857422,  -21.689987182617188,
             26.721012115478516,  -62.25614929199219,  25.783079147338867,
             -69.070556640625,    28.84514808654785,   -125.13325500488281,
             30.907209396362305,  -13.891133308410645, 18.914127349853516,
             -38.81135940551758,  17.960111618041992,  -29.915504455566406,
             21.006093978881836,  -70.20361328125,     23.052082061767578,
             -12.857919692993164, 17.89089584350586,   -35.771610260009766,
             16.95684814453125,   -24.949115753173828, 20.022798538208008,
             -63.39042282104492,  22.088754653930664,  -14.02528190612793,
             19.06132698059082,   -39.2921257019043,   18.133424758911133,
             -30.847349166870117, 21.205520629882812,  -71.69097137451172,
             23.27761459350586,   -15.192638397216797, 20.23175811767578,
             -42.8126335144043,   19.309999465942383,  -36.74560546875,
             22.388240814208984,  -79.99152374267578,  24.46647834777832,
             -8.556736946105957,  13.584352493286133,  -22.835901260375977,
             12.6395845413208,    -3.336000442504883,  15.694815635681152,
             -33.0570182800293,   17.750045776367188
           ]));
     });

  it('relu6 bias stride 2 x=[1,8,8,16] f=[3,3,16,8] s=[2,2] d=8 p=same',
     async () => {
       const inputDepth = 16;
       const xSize = 8;
       const inputShape: [number, number, number, number] =
           [1, xSize, xSize, inputDepth];
       const outputDepth = 8;
       const fSize = 3;
       const pad = 'same';
       const stride: [number, number] = [2, 2];

       const inputs = generateCaseInputs(
           1 * xSize * xSize * inputDepth,
           fSize * fSize * inputDepth * outputDepth);
       const x = tf.tensor4d(inputs.input, inputShape);
       const w =
           tf.tensor4d(inputs.filter, [fSize, fSize, inputDepth, outputDepth]);
       const bias = tf.tensor1d([1, 4, 2, 3, 9, 6, 5, 8]);

       const result = tf.fused.conv2d({
         x,
         filter: w,
         strides: stride,
         pad,
         dataFormat: 'NHWC',
         dilations: [1, 1],
         activation: 'relu6',
         bias
       });
       expect(result.shape).toEqual([1, 4, 4, 8]);
       const resultData = await result.data();
       expectArraysClose(resultData, new Float32Array([
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           4.6677775382995605,
                           0.31717729568481445,
                           5.697869777679443,
                           0,
                           6,
                           2.2569849491119385,
                           6,
                           4.226885795593262,
                           2.0319995880126953,
                           2.9575586318969727,
                           3.052880048751831,
                           1.9366796016693115,
                           6,
                           4.915799617767334,
                           6,
                           6,
                           0,
                           5.5979437828063965,
                           0.4078875780105591,
                           4.586280822753906,
                           6,
                           6,
                           3.43121600151062,
                           6,
                           0,
                           6,
                           0,
                           5.401776313781738,
                           6,
                           6,
                           2.602976083755493,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6,
                           0,
                           6
                         ]));
     });

  it('leakyrelu bias stride 2 x=[1,8,8,16] f=[3,3,16,1] s=[2,2] d=8 p=same',
     async () => {
       const inputDepth = 16;
       const xSize = 8;
       const inputShape: [number, number, number, number] =
           [1, xSize, xSize, inputDepth];
       const outputDepth = 8;
       const fSize = 3;
       const pad = 'same';
       const stride: [number, number] = [2, 2];

       const inputs = generateCaseInputs(
           1 * xSize * xSize * inputDepth,
           fSize * fSize * inputDepth * outputDepth);
       const x = tf.tensor4d(inputs.input, inputShape);
       const w =
           tf.tensor4d(inputs.filter, [fSize, fSize, inputDepth, outputDepth]);
       const bias = tf.tensor1d([1, 4, 2, 3, 9, 6, 5, 8]);
       const leakyreluAlpha = 0.3;

       const result = tf.fused.conv2d({
         x,
         filter: w,
         strides: stride,
         pad,
         dataFormat: 'NHWC',
         dilations: [1, 1],
         activation: 'leakyrelu',
         leakyreluAlpha,
         bias
       });
       expect(result.shape).toEqual([1, 4, 4, 8]);
       expectArraysClose(
           await result.data(), new Float32Array([
             25.75398063659668,    -6.241768836975098,   26.857805252075195,
             -6.5729146003723145,  33.961631774902344,   -5.704063892364502,
             30.065458297729492,   -5.135210037231445,   23.118206024169922,
             -5.449653148651123,   24.212820053100586,   -5.778036117553711,
             31.307422637939453,   -4.906418323516846,   27.402034759521484,
             -4.334802627563477,   20.482431411743164,   -4.657539367675781,
             21.567821502685547,   -4.983157157897949,   28.653217315673828,
             -4.108772277832031,   24.73861312866211,    -3.534390687942505,
             11.078080177307129,   -1.8312718868255615,  12.130399703979492,
             -2.1469674110412598,  19.182720184326172,   -1.262665033340454,
             15.235037803649902,   -0.6783602833747864,  4.6677775382995605,
             0.31717729568481445,  5.697869777679443,    -0.21387571096420288,
             12.727968215942383,   2.2569849491119385,   8.758066177368164,
             4.226885795593262,    2.0319995880126953,   2.9575586318969727,
             3.052880048751831,    1.9366796016693115,   10.073760032653809,
             4.915799617767334,    6.094639778137207,    6.89492130279541,
             -0.18113291263580322, 5.5979437828063965,   0.4078875780105591,
             4.586280822753906,    7.419551849365234,    7.5746169090271,
             3.43121600151062,     9.562952041625977,    -0.42195841670036316,
             6.404943943023682,    -0.12100804597139359, 5.401776313781738,
             6.5998077392578125,   8.398608207702637,    2.602976083755493,
             10.395440101623535,   -4.925530433654785,   21.440250396728516,
             -4.6386189460754395,  20.483882904052734,   -2.5517091751098633,
             23.527509689331055,   -3.764799118041992,   25.571144104003906,
             -5.7162628173828125,  24.080629348754883,   -5.432116508483887,
             23.133480072021484,   -3.347970962524414,   26.186328887939453,
             -4.5638251304626465,  28.239177703857422,   -6.5069966316223145,
             26.721012115478516,   -6.225615501403809,   25.783079147338867,
             -4.144233703613281,   28.84514808654785,    -5.36285400390625,
             30.907209396362305,   -4.167340278625488,   18.914127349853516,
             -3.881135940551758,   17.960111618041992,   -1.794930338859558,
             21.006093978881836,   -3.0087265968322754,  23.052082061767578,
             -3.8573760986328125,  17.89089584350586,    -3.5771610736846924,
             16.95684814453125,    -1.4969470500946045,  20.022798538208008,
             -2.7167325019836426,  22.088754653930664,   -4.207584857940674,
             19.06132698059082,    -3.9292125701904297,  18.133424758911133,
             -1.8508410453796387,  21.205520629882812,   -3.0724704265594482,
             23.27761459350586,    -4.557791709899902,   20.23175811767578,
             -4.28126335144043,    19.309999465942383,   -2.2047364711761475,
             22.388240814208984,   -3.428208351135254,   24.46647834777832,
             -2.567021131515503,   13.584352493286133,   -2.283590316772461,
             12.6395845413208,     -0.20016004145145416, 15.694815635681152,
             -1.41672945022583,    17.750045776367188
           ]));
     });

  it('basic with bias', async () => {
    const inputDepth = 2;
    const inShape: [number, number, number, number] = [2, 2, 2, inputDepth];
    const outputDepth = 2;
    const fSize = 1;
    const pad = 0;
    const stride = 1;

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], inShape);
    const w =
        tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.fused.conv2d({
      x,
      filter: w,
      strides: stride,
      pad,
      dataFormat: 'NHWC',
      dilations: [1, 1],
      bias: tf.tensor1d([5, 6])
    });
    expect(result.shape).toEqual([2, 2, 2, 2]);
    const expected =
        [0, 8, -6, 11, -12, 14, -18, 17, -24, 20, -30, 23, -36, 26, -42, 29];

    expectArraysClose(await result.data(), expected);
  });

  it('basic with explicit padding', async () => {
    const inputDepth = 1;
    const outputDepth = 1;
    const pad =
        [[0, 0], [1, 2], [0, 1], [0, 0]] as tf.backend_util.ExplicitPadding;
    const stride = 1;
    const dataFormat = 'NHWC';
    const dilation = 1;

    const x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [4, 2, inputDepth]);
    const w =
        tf.tensor4d([3, 1, 5, 0, 2, 7, 8, 9], [4, 2, inputDepth, outputDepth]);

    const result = tf.fused.conv2d(
        {x, filter: w, strides: stride, pad, dataFormat, dilations: dilation});

    const resultData = await result.data();
    expect(result.shape).toEqual([4, 2, 1]);
    expectArraysClose(resultData, [133, 66, 200, 102, 108, 58, 56, 58]);
  });

  it('basic with elu', async () => {
    const inputDepth = 2;
    const inShape: [number, number, number, number] = [2, 2, 2, inputDepth];
    const outputDepth = 2;
    const fSize = 1;
    const pad = 0;
    const stride = 1;

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], inShape);
    const w =
        tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.fused.conv2d({
      x,
      filter: w,
      strides: stride,
      pad,
      dataFormat: 'NHWC',
      dilations: [1, 1],
      activation: 'elu'
    });
    expect(result.shape).toEqual([2, 2, 2, 2]);
    const expected =
        [-0.99326, 2, -1, 5, -1, 8, -1, 11, -1, 14, -1, 17, -1, 20, -1, 23];

    expectArraysClose(await result.data(), expected);
  });

  it('basic with prelu', async () => {
    const inputDepth = 2;
    const inShape: [number, number, number, number] = [2, 2, 2, inputDepth];
    const outputDepth = 2;
    const fSize = 1;
    const pad = 0;
    const stride = 1;

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], inShape);
    const alpha = tf.tensor3d([0.25, 0.75], [1, 1, 2]);
    const w =
        tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.fused.conv2d({
      x,
      filter: w,
      strides: stride,
      pad,
      dataFormat: 'NHWC',
      dilations: [1, 1],
      activation: 'prelu',
      preluActivationWeights: alpha
    });
    expect(result.shape).toEqual([2, 2, 2, 2]);
    const expected = [
      -1.25, 2, -2.75, 5, -4.25, 8, -5.75, 11, -7.25, 14, -8.75, 17, -10.25, 20,
      -11.75, 23
    ];

    expectArraysClose(await result.data(), expected);
  });

  it('basic with leakyrelu', async () => {
    const inputDepth = 2;
    const inShape: [number, number, number, number] = [2, 2, 2, inputDepth];
    const outputDepth = 2;
    const fSize = 1;
    const pad = 0;
    const stride = 1;

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], inShape);
    const alpha = 0.3;
    const w =
        tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.fused.conv2d({
      x,
      filter: w,
      strides: stride,
      pad,
      dataFormat: 'NHWC',
      dilations: [1, 1],
      activation: 'leakyrelu',
      leakyreluAlpha: alpha
    });
    expect(result.shape).toEqual([2, 2, 2, 2]);
    const expected = [
      -1.5, 2, -3.3000001907348633, 5, -5.100000381469727, 8,
      -6.900000095367432, 11, -8.700000762939453, 14, -10.5, 17,
      -12.300000190734863, 20, -14.100000381469727, 23
    ];

    expectArraysClose(await result.data(), expected);
  });

  it('basic with sigmoid', async () => {
    const inputDepth = 2;
    const inShape: [number, number, number, number] = [2, 2, 2, inputDepth];
    const outputDepth = 2;
    const fSize = 1;
    const pad = 0;
    const stride = 1;

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], inShape);
    const alpha = 0.3;
    const w = tf.tensor4d(
        [-0.1, 0.1, -0.2, 0.05], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.fused.conv2d({
      x,
      filter: w,
      strides: stride,
      pad,
      dataFormat: 'NHWC',
      dilations: [1, 1],
      activation: 'sigmoid',
      leakyreluAlpha: alpha
    });
    expect(result.shape).toEqual([2, 2, 2, 2]);
    const expected = [
      0.3775407, 0.549834, 0.24973989, 0.6224593, 0.15446526, 0.6899744,
      0.09112296, 0.7502601, 0.0521535, 0.80218387, 0.02931219, 0.84553474,
      0.0163025, 0.8807971, 0.0090133, 0.908877
    ];

    expectArraysClose(await result.data(), expected);
  });

  it('basic with broadcasted bias and relu', async () => {
    const inputDepth = 2;
    const inShape: [number, number, number, number] = [2, 2, 2, inputDepth];
    const outputDepth = 2;
    const fSize = 1;
    const pad = 0;
    const stride = 1;

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], inShape);
    const w =
        tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.fused.conv2d({
      x,
      filter: w,
      strides: stride,
      pad,
      dataFormat: 'NHWC',
      dilations: [1, 1],
      bias: tf.scalar(5),
      activation: 'relu'
    });
    expect(result.shape).toEqual([2, 2, 2, 2]);
    const expected = [0, 7, 0, 10, 0, 13, 0, 16, 0, 19, 0, 22, 0, 25, 0, 28];

    expectArraysClose(await result.data(), expected);
  });

  it('im2row', async () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [4, 4, inputDepth];
    const outputDepth = 3;
    const fSize = 1;
    const pad = 'same';
    const strides: [number, number] = [2, 2];

    const x = tf.tensor3d(
        [
          10, 30, 50, 70, 20, 40, 60, 80, -10, -30, -50, -70, -20, -40, -60, -80
        ],
        inputShape);
    const w = tf.tensor4d([1, 0.5, 1], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.fused.conv2d({x, filter: w, strides, pad});

    expectArraysClose(
        await result.data(),
        [10, 5, 10, 50, 25, 50, -10, -5, -10, -50, -25, -50]);
  });

  it('im2row with relu', async () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [4, 4, inputDepth];
    const outputDepth = 3;
    const fSize = 1;
    const pad = 'same';
    const strides: [number, number] = [2, 2];

    const x = tf.tensor3d(
        [
          10, 30, 50, 70, 20, 40, 60, 80, -10, -30, -50, -70, -20, -40, -60, -80
        ],
        inputShape);
    const w = tf.tensor4d([1, 0.5, 1], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.fused.conv2d({
      x,
      filter: w,
      strides,
      pad,
      dataFormat: 'NHWC',
      dilations: [1, 1],
      activation: 'relu'
    });

    expectArraysClose(
        await result.data(), [10, 5, 10, 50, 25, 50, 0, 0, 0, 0, 0, 0]);
  });

  it('im2row with prelu', async () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [4, 4, inputDepth];
    const outputDepth = 3;
    const fSize = 1;
    const pad = 'same';
    const strides: [number, number] = [2, 2];

    const x = tf.tensor3d(
        [
          10, 30, 50, 70, 20, 40, 60, 80, -10, -30, -50, -70, -20, -40, -60, -80
        ],
        inputShape);
    const w = tf.tensor4d([1, 0.5, 1], [fSize, fSize, inputDepth, outputDepth]);
    const alpha = tf.tensor3d([0.5], [1, 1, inputDepth]);

    const result = tf.fused.conv2d({
      x,
      filter: w,
      strides,
      pad,
      dataFormat: 'NHWC',
      dilations: [1, 1],
      activation: 'prelu',
      preluActivationWeights: alpha
    });

    expectArraysClose(
        await result.data(),
        [10, 5, 10, 50, 25, 50, -5, -2.5, -5, -25, -12.5, -25]);
  });

  it('im2row with leakyrelu', async () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [4, 4, inputDepth];
    const outputDepth = 3;
    const fSize = 1;
    const pad = 'same';
    const strides: [number, number] = [2, 2];

    const x = tf.tensor3d(
        [
          10, 30, 50, 70, 20, 40, 60, 80, -10, -30, -50, -70, -20, -40, -60, -80
        ],
        inputShape);
    const w = tf.tensor4d([1, 0.5, 1], [fSize, fSize, inputDepth, outputDepth]);
    const alpha = 0.3;

    const result = tf.fused.conv2d({
      x,
      filter: w,
      strides,
      pad,
      dataFormat: 'NHWC',
      dilations: [1, 1],
      activation: 'leakyrelu',
      leakyreluAlpha: alpha
    });

    expectArraysClose(await result.data(), [
      10, 5, 10, 50, 25, 50, -3, -1.5, -3, -15.000000953674316,
      -7.500000476837158, -15.000000953674316
    ]);
  });

  it('pointwise with prelu', async () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [4, 4, inputDepth];
    const outputDepth = 3;
    const fSize = 1;
    const pad = 'same';
    const strides: [number, number] = [1, 1];

    const x = tf.tensor3d(
        [
          10, 30, 50, 70, 20, 40, 60, 80, -10, -30, -50, -70, -20, -40, -60, -80
        ],
        inputShape);
    const w = tf.tensor4d([1, 0.5, 1], [fSize, fSize, inputDepth, outputDepth]);
    const alpha = tf.tensor3d([0.5], [1, 1, inputDepth]);

    const result = tf.fused.conv2d({
      x,
      filter: w,
      strides,
      pad,
      dataFormat: 'NHWC',
      dilations: [1, 1],
      activation: 'prelu',
      preluActivationWeights: alpha
    });

    expectArraysClose(await result.data(), [
      10,  5,    10,  30,  15,   30,  50,  25,    50,  70,  35,    70,
      20,  10,   20,  40,  20,   40,  60,  30,    60,  80,  40,    80,
      -5,  -2.5, -5,  -15, -7.5, -15, -25, -12.5, -25, -35, -17.5, -35,
      -10, -5,   -10, -20, -10,  -20, -30, -15,   -30, -40, -20,   -40
    ]);
  });

  it('pointwise with leakyrelu', async () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [4, 4, inputDepth];
    const outputDepth = 3;
    const fSize = 1;
    const pad = 'same';
    const strides: [number, number] = [1, 1];

    const x = tf.tensor3d(
        [
          10, 30, 50, 70, 20, 40, 60, 80, -10, -30, -50, -70, -20, -40, -60, -80
        ],
        inputShape);
    const w = tf.tensor4d([1, 0.5, 1], [fSize, fSize, inputDepth, outputDepth]);
    const alpha = 0.3;

    const result = tf.fused.conv2d({
      x,
      filter: w,
      strides,
      pad,
      dataFormat: 'NHWC',
      dilations: [1, 1],
      activation: 'leakyrelu',
      leakyreluAlpha: alpha
    });

    expectArraysClose(await result.data(), [
      10,
      5,
      10,
      30,
      15,
      30,
      50,
      25,
      50,
      70,
      35,
      70,
      20,
      10,
      20,
      40,
      20,
      40,
      60,
      30,
      60,
      80,
      40,
      80,
      -3,
      -1.5,
      -3,
      -9,
      -4.5,
      -9,
      -15.000000953674316,
      -7.500000476837158,
      -15.000000953674316,
      -21,
      -10.5,
      -21,
      -6,
      -3,
      -6,
      -12,
      -6,
      -12,
      -18,
      -9,
      -18,
      -24,
      -12,
      -24
    ]);
  });

  it('im2row with broadcasted bias and relu', async () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [4, 4, inputDepth];
    const outputDepth = 3;
    const fSize = 1;
    const pad = 'same';
    const strides: [number, number] = [2, 2];

    const x = tf.tensor3d(
        [
          10, 30, 50, 70, 20, 40, 60, 80, -10, -30, -50, -70, -20, -40, -60, -80
        ],
        inputShape);
    const w = tf.tensor4d([1, 0.5, 1], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.fused.conv2d({
      x,
      filter: w,
      strides,
      pad,
      dataFormat: 'NHWC',
      dilations: [1, 1],
      bias: tf.scalar(5),
      activation: 'relu'
    });

    expectArraysClose(
        await result.data(), [15, 10, 15, 55, 30, 55, 0, 0, 0, 0, 0, 0]);
  });

  it('backProp input x=[2,3,3,1] f=[2,2,1,1] s=1 p=0', async () => {
    const inputDepth = 1;
    const outputDepth = 1;
    const inputShape: [number, number, number, number] = [2, 3, 3, inputDepth];
    const filterSize = 2;
    const strides = 1;
    const pad = 0;

    const filterShape: [number, number, number, number] =
        [filterSize, filterSize, inputDepth, outputDepth];
    const filter = tf.tensor4d([-1, 1, -2, 0.5], filterShape);

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9], inputShape);
    const dy = tf.tensor4d([3, 1, 2, 0, 3, 1, 2, 0], [2, 2, 2, 1]);

    const grads = tf.grads(
        (x: tf.Tensor4D) => tf.fused.conv2d({x, filter, strides, pad}));
    const [dx] = grads([x], dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(
        await dx.data(),
        [-3, 2, 1, -8, 1.5, 0.5, -4, 1, 0, -3, 2, 1, -8, 1.5, 0.5, -4, 1, 0]);
  });

  it('gradient x=[2,3,3,1] f=[2,2,1,1] s=1 p=0', async () => {
    const inputDepth = 1;
    const outputDepth = 1;
    const inputShape: [number, number, number, number] = [2, 3, 3, inputDepth];
    const filterSize = 2;
    const strides = 1;
    const pad = 0;

    const filterShape: [number, number, number, number] =
        [filterSize, filterSize, inputDepth, outputDepth];
    const filter = tf.tensor4d([-1, 1, -2, 0.5], filterShape);

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9], inputShape);
    const dy = tf.tensor4d([3, 1, 2, 0, 3, 1, 2, 0], [2, 2, 2, 1]);

    const grads = tf.grads(
        (x: tf.Tensor4D, filter: tf.Tensor4D) =>
            tf.fused.conv2d({x, filter, strides, pad}));
    const [dx, dfilter] = grads([x, filter], dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(
        await dx.data(),
        [-3, 2, 1, -8, 1.5, 0.5, -4, 1, 0, -3, 2, 1, -8, 1.5, 0.5, -4, 1, 0]);

    expect(dfilter.shape).toEqual(filterShape);
    expectArraysClose(await dfilter.data(), [26, 38, 62, 74]);
  });

  it('gradient x=[2,3,3,1] f=[2,2,1,1] s=1 p=0 with bias', async () => {
    const inputDepth = 1;
    const outputDepth = 1;
    const inputShape: [number, number, number, number] = [2, 3, 3, inputDepth];
    const filterSize = 2;
    const strides = 1;
    const pad = 0;

    const filterShape: [number, number, number, number] =
        [filterSize, filterSize, inputDepth, outputDepth];
    const filter = tf.tensor4d([-1, 1, -2, 0.5], filterShape);
    const bias = tf.ones([2, 2, 2, 1]);

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9], inputShape);
    const dy = tf.tensor4d([3, 1, 2, 0, 3, 1, 2, 0], [2, 2, 2, 1]);

    const fusedGrads =
        tf.grads((x: tf.Tensor4D, w: tf.Tensor4D, b) => tf.fused.conv2d({
          x,
          filter: w,
          strides,
          pad,
          dataFormat: 'NHWC',
          dilations: [1, 1],
          bias: b
        }));
    const [dxFused, dfilterFused, dbiasFused] =
        fusedGrads([x, filter, bias], dy);

    const grads = tf.grads((x: tf.Tensor4D, filter: tf.Tensor4D, bias) => {
      const conv = tf.conv2d(x, filter, strides, pad);
      const sum = tf.add(conv, bias);
      return sum;
    });
    const [dx, dfilter, dbias] = grads([x, filter, bias], dy);

    expectArraysClose(await dxFused.array(), await dx.array());
    expectArraysClose(await dfilterFused.array(), await dfilter.array());
    expectArraysClose(await dbiasFused.array(), await dbias.array());
  });

  it('gradient x=[2,3,3,1] f=[2,2,1,1] s=1 p=0 with bias and relu',
     async () => {
       const inputDepth = 1;
       const outputDepth = 1;
       const inputShape: [number, number, number, number] =
           [2, 3, 3, inputDepth];
       const filterSize = 2;
       const strides = 1;
       const pad = 0;

       const filterShape: [number, number, number, number] =
           [filterSize, filterSize, inputDepth, outputDepth];
       const filter = tf.tensor4d([-1, 1, -2, 0.5], filterShape);
       const bias = tf.ones([2, 2, 2, 1]);

       const x = tf.tensor4d(
           [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9], inputShape);
       const dy = tf.tensor4d([3, 1, 2, 0, 3, 1, 2, 0], [2, 2, 2, 1]);

       const fusedGrads =
           tf.grads((x: tf.Tensor4D, w: tf.Tensor4D, b) => tf.fused.conv2d({
             x,
             filter: w,
             strides,
             pad,
             dataFormat: 'NHWC',
             dilations: [1, 1],
             bias: b,
             activation: 'relu'
           }));
       const [dxFused, dfilterFused, dbiasFused] =
           fusedGrads([x, filter, bias], dy);

       const grads = tf.grads((x: tf.Tensor4D, filter: tf.Tensor4D, bias) => {
         const conv = tf.conv2d(x, filter, strides, pad);
         const sum = tf.add(conv, bias);
         return tf.relu(sum);
       });
       const [dx, dfilter, dbias] = grads([x, filter, bias], dy);

       expectArraysClose(await dxFused.array(), await dx.array());
       expectArraysClose(await dfilterFused.array(), await dfilter.array());
       expectArraysClose(await dbiasFused.array(), await dbias.array());
     });

  it('gradient x=[2,3,3,1] f=[2,2,1,1] s=1 p=0 with bias and elu', async () => {
    const inputDepth = 1;
    const outputDepth = 1;
    const inputShape: [number, number, number, number] = [2, 3, 3, inputDepth];
    const filterSize = 2;
    const strides = 1;
    const pad = 0;

    const filterShape: [number, number, number, number] =
        [filterSize, filterSize, inputDepth, outputDepth];
    const filter = tf.tensor4d([-1, 1, -2, 0.5], filterShape);
    const bias = tf.ones([2, 2, 2, 1]);

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9], inputShape);
    const dy = tf.tensor4d([3, 1, 2, 0, 3, 1, 2, 0], [2, 2, 2, 1]);

    const fusedGrads =
        tf.grads((x: tf.Tensor4D, w: tf.Tensor4D, b) => tf.fused.conv2d({
          x,
          filter: w,
          strides,
          pad,
          dataFormat: 'NHWC',
          dilations: [1, 1],
          bias: b,
          activation: 'elu'
        }));
    const [dxFused, dfilterFused, dbiasFused] =
        fusedGrads([x, filter, bias], dy);

    const grads = tf.grads((x: tf.Tensor4D, filter: tf.Tensor4D, bias) => {
      const conv = tf.conv2d(x, filter, strides, pad);
      const sum = tf.add(conv, bias);
      return tf.elu(sum);
    });
    const [dx, dfilter, dbias] = grads([x, filter, bias], dy);

    expectArraysClose(await dxFused.array(), await dx.array());
    expectArraysClose(await dfilterFused.array(), await dfilter.array());
    expectArraysClose(await dbiasFused.array(), await dbias.array());
  });
});
