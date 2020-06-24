/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import * as tf from '../index';
import {ALL_ENVS, describeWithFlags} from '../jasmine_util';
import {Tensor5D} from '../tensor';
import {expectArraysClose} from '../test_util';
import {sizeFromShape} from '../util';

// Generates small floating point inputs to avoid overflows
function generateCaseInputs(totalSizeTensor: number, totalSizeFilter: number) {
  const inp = new Array(totalSizeTensor);
  const filt = new Array(totalSizeFilter);

  for (let i = 0; i < totalSizeTensor; i++) {
    inp[i] = (i + 1) / totalSizeTensor;
  }
  for (let i = 0; i < totalSizeFilter; i++) {
    filt[i] = (i + 1) / totalSizeFilter;
  }

  return {input: inp, filter: filt};
}

function generateGradientCaseInputs(
    totalSizeTensor: number, totalSizeFilter: number) {
  const inp = new Array(totalSizeTensor);
  const filt = new Array(totalSizeFilter);

  for (let i = 0; i < totalSizeTensor; i++) {
    inp[i] = i + 1;
  }
  for (let i = 0; i < totalSizeFilter; i++) {
    filt[i] = i + 1;
  }

  return {input: inp, filter: filt};
}

function runConv3DTestCase(
    batch: number, inDepth: number, inHeight: number, inWidth: number,
    inChannels: number, outChannels: number, fDepth: number, fHeight: number,
    fWidth: number, pad: 'valid'|'same',
    stride: [number, number, number]|number) {
  const inputShape: [number, number, number, number, number] =
      [batch, inDepth, inHeight, inWidth, inChannels];
  const filterShape: [number, number, number, number, number] =
      [fDepth, fHeight, fWidth, inChannels, outChannels];

  const totalSizeTensor = sizeFromShape(inputShape);
  const totalSizeFilter = sizeFromShape(filterShape);
  const inputs = generateCaseInputs(totalSizeTensor, totalSizeFilter);

  const x = tf.tensor5d(inputs.input, inputShape);
  const w = tf.tensor5d(inputs.filter, filterShape);

  const result = tf.conv3d(x, w, stride, pad);
  return result;
}

function runGradientConv3DTestCase(
    batch: number, inDepth: number, inHeight: number, inWidth: number,
    inChannels: number, outChannels: number, fDepth: number, fHeight: number,
    fWidth: number, pad: 'valid'|'same',
    stride: [number, number, number]|number) {
  const inputShape: [number, number, number, number, number] =
      [batch, inDepth, inHeight, inWidth, inChannels];
  const filterShape: [number, number, number, number, number] =
      [fDepth, fHeight, fWidth, inChannels, outChannels];

  const totalSizeTensor = sizeFromShape(inputShape);
  const totalSizeFilter = sizeFromShape(filterShape);
  const inputs = generateGradientCaseInputs(totalSizeTensor, totalSizeFilter);

  const x = tf.tensor5d(inputs.input, inputShape);
  const w = tf.tensor5d(inputs.filter, filterShape);

  const grads = tf.grads(
      (x: Tensor5D, filter: Tensor5D) =>
          tf.conv3d(x.clone(), filter.clone(), stride, pad).clone());
  const [dx, dfilter] = grads([x, w]);

  expect(dx.shape).toEqual(x.shape);
  expect(dfilter.shape).toEqual(w.shape);

  return [dx, dfilter];
}

describeWithFlags('conv3d', ALL_ENVS, () => {
  it('x=[1, 2, 3, 1, 3] f=[1, 1, 1, 3, 3] s=1 d=1 p=valid', async () => {
    const batch = 1;
    const inDepth = 2;
    const inHeight = 3;
    const inWidth = 1;
    const inChannels = 3;
    const outChannels = 3;
    const fSize = 1;
    const pad = 'valid';
    const stride = 1;
    const result = runConv3DTestCase(
        batch, inDepth, inHeight, inWidth, inChannels, outChannels, fSize,
        fSize, fSize, pad, stride);

    const expectedOutput = [
      0.18518519, 0.22222222, 0.25925926, 0.40740741, 0.5, 0.59259259,
      0.62962963, 0.77777778, 0.92592593, 0.85185185, 1.05555556, 1.25925926,
      1.07407407, 1.33333333, 1.59259259, 1.2962963, 1.61111111, 1.92592593
    ];

    expectArraysClose(await result.data(), expectedOutput);
  });
  it('x=[1, 2, 1, 3, 3] f=[1, 1, 1, 3, 3] s=1 d=1 p=valid', async () => {
    const batch = 1;
    const inDepth = 2;
    const inHeight = 1;
    const inWidth = 3;
    const inChannels = 3;
    const outChannels = 3;
    const fSize = 1;
    const pad = 'valid';
    const stride = 1;
    const result = runConv3DTestCase(
        batch, inDepth, inHeight, inWidth, inChannels, outChannels, fSize,
        fSize, fSize, pad, stride);

    const expectedOutput = [
      0.18518519, 0.22222222, 0.25925926, 0.40740741, 0.5, 0.59259259,
      0.62962963, 0.77777778, 0.92592593, 0.85185185, 1.05555556, 1.25925926,
      1.07407407, 1.33333333, 1.59259259, 1.2962963, 1.61111111, 1.92592593
    ];

    expectArraysClose(await result.data(), expectedOutput);
  });

  it('x=[1, 1, 2, 3, 3] f=[1, 1, 1, 3, 3] s=1 d=1 p=valid', async () => {
    const batch = 1;
    const inDepth = 1;
    const inHeight = 2;
    const inWidth = 3;
    const inChannels = 3;
    const outChannels = 3;
    const fSize = 1;
    const pad = 'valid';
    const stride = 1;
    const result = runConv3DTestCase(
        batch, inDepth, inHeight, inWidth, inChannels, outChannels, fSize,
        fSize, fSize, pad, stride);

    const expectedOutput = [
      0.18518519, 0.22222222, 0.25925926, 0.40740741, 0.5, 0.59259259,
      0.62962963, 0.77777778, 0.92592593, 0.85185185, 1.05555556, 1.25925926,
      1.07407407, 1.33333333, 1.59259259, 1.2962963, 1.61111111, 1.92592593
    ];

    expectArraysClose(await result.data(), expectedOutput);
  });

  it('x=[1, 4, 2, 3, 3] f=[2, 2, 2, 3, 3] s=1 d=1 p=valid', async () => {
    const batch = 1;
    const inDepth = 4;
    const inHeight = 2;
    const inWidth = 3;
    const inChannels = 3;
    const outChannels = 3;
    const fSize = 2;
    const pad = 'valid';
    const stride = 1;
    const result = runConv3DTestCase(
        batch, inDepth, inHeight, inWidth, inChannels, outChannels, fSize,
        fSize, fSize, pad, stride);

    const expectedOutput = [
      3.77199074, 3.85069444, 3.92939815, 4.2650463, 4.35763889, 4.45023148,
      6.73032407, 6.89236111, 7.05439815, 7.22337963, 7.39930556, 7.57523148,
      9.68865741, 9.93402778, 10.17939815, 10.18171296, 10.44097222, 10.70023148
    ];

    expectArraysClose(await result.data(), expectedOutput);
  });

  it('x=[1, 5, 8, 7, 1] f=[1, 2, 3, 1, 1] s=[2, 3, 1] d=1 p=same', async () => {
    const batch = 1;
    const inDepth = 5;
    const inHeight = 8;
    const inWidth = 7;
    const inChannels = 1;
    const outChannels = 1;
    const fDepth = 1;
    const fHeight = 2;
    const fWidth = 3;
    const pad = 'same';
    const stride: [number, number, number] = [2, 3, 1];
    const result = runConv3DTestCase(
        batch, inDepth, inHeight, inWidth, inChannels, outChannels, fDepth,
        fHeight, fWidth, pad, stride);

    const expectedOutput = [
      0.06071429, 0.08988095, 0.10238095, 0.11488095, 0.12738095, 0.13988095,
      0.08452381, 0.26071429, 0.35238095, 0.36488095, 0.37738095, 0.38988095,
      0.40238095, 0.23452381, 0.46071429, 0.61488095, 0.62738095, 0.63988095,
      0.65238095, 0.66488095, 0.38452381, 1.12738095, 1.48988095, 1.50238095,
      1.51488095, 1.52738095, 1.53988095, 0.88452381, 1.32738095, 1.75238095,
      1.76488095, 1.77738095, 1.78988095, 1.80238095, 1.03452381, 1.52738095,
      2.01488095, 2.02738095, 2.03988095, 2.05238095, 2.06488095, 1.18452381,
      2.19404762, 2.88988095, 2.90238095, 2.91488095, 2.92738095, 2.93988095,
      1.68452381, 2.39404762, 3.15238095, 3.16488095, 3.17738095, 3.18988095,
      3.20238095, 1.83452381, 2.59404762, 3.41488095, 3.42738095, 3.43988095,
      3.45238095, 3.46488095, 1.98452381
    ];

    expectArraysClose(await result.data(), expectedOutput);
  });

  it('x=[1, 4, 2, 3, 3] f=[2, 2, 2, 3, 3] s=2 d=1 p=valid', async () => {
    const batch = 1;
    const inDepth = 4;
    const inHeight = 2;
    const inWidth = 3;
    const inChannels = 3;
    const outChannels = 3;
    const fSize = 2;
    const pad = 'valid';
    const stride = 2;
    const result = runConv3DTestCase(
        batch, inDepth, inHeight, inWidth, inChannels, outChannels, fSize,
        fSize, fSize, pad, stride);

    const expectedOutput = [
      3.77199074, 3.85069444, 3.92939815, 9.68865741, 9.93402778, 10.17939815
    ];

    expectArraysClose(await result.data(), expectedOutput);
  });

  it('x=[1, 6, 7, 8, 2] f=[3, 2, 1, 2, 3] s=3 d=1 p=valid', async () => {
    const batch = 1;
    const inDepth = 6;
    const inHeight = 7;
    const inWidth = 8;
    const inChannels = 2;
    const outChannels = 3;
    const fDepth = 3;
    const fHeight = 2;
    const fWidth = 1;
    const pad = 'valid';
    const stride = 3;
    const result = runConv3DTestCase(
        batch, inDepth, inHeight, inWidth, inChannels, outChannels, fDepth,
        fHeight, fWidth, pad, stride);

    const expectedOutput = [
      1.51140873, 1.57167659, 1.63194444, 1.56349206, 1.62673611, 1.68998016,
      1.6155754,  1.68179563, 1.74801587, 1.9280754,  2.01215278, 2.09623016,
      1.98015873, 2.0672123,  2.15426587, 2.03224206, 2.12227183, 2.21230159,
      4.4280754,  4.65500992, 4.88194444, 4.48015873, 4.71006944, 4.93998016,
      4.53224206, 4.76512897, 4.99801587, 4.84474206, 5.09548611, 5.34623016,
      4.8968254,  5.15054563, 5.40426587, 4.94890873, 5.20560516, 5.46230159
    ];

    expectArraysClose(await result.data(), expectedOutput);
  });

  it('x=[1, 4, 2, 3, 3] f=[2, 2, 2, 3, 3] s=2 d=1 p=same', async () => {
    const batch = 1;
    const inDepth = 4;
    const inHeight = 2;
    const inWidth = 3;
    const inChannels = 3;
    const outChannels = 3;
    const fSize = 2;
    const pad = 'same';
    const stride = 2;
    const result = runConv3DTestCase(
        batch, inDepth, inHeight, inWidth, inChannels, outChannels, fSize,
        fSize, fSize, pad, stride);

    const expectedOutput = [
      3.77199074, 3.85069444, 3.92939815, 2.0162037, 2.06597222, 2.11574074,
      9.68865741, 9.93402778, 10.17939815, 4.59953704, 4.73263889, 4.86574074
    ];

    expectArraysClose(await result.data(), expectedOutput);
  });

  it('x=[1, 3, 3, 3, 1] f=[1, 1, 1, 1, 1] s=2 d=1 p=same', async () => {
    const batch = 1;
    const inDepth = 3;
    const inHeight = 3;
    const inWidth = 3;
    const inChannels = 1;
    const outChannels = 1;
    const fSize = 1;
    const pad = 'same';
    const stride = 2;
    const result = runConv3DTestCase(
        batch, inDepth, inHeight, inWidth, inChannels, outChannels, fSize,
        fSize, fSize, pad, stride);

    const expectedOutput = [
      0.03703704, 0.11111111, 0.25925926, 0.33333333, 0.7037037, 0.77777778,
      0.92592593, 1.
    ];

    expectArraysClose(await result.data(), expectedOutput);
  });

  it('x=[1, 3, 3, 3, 1] f=[1, 1, 1, 1, 1] s=2 d=1 p=valid', async () => {
    const batch = 1;
    const inDepth = 3;
    const inHeight = 3;
    const inWidth = 3;
    const inChannels = 1;
    const outChannels = 1;
    const fSize = 1;
    const pad = 'valid';
    const stride = 2;
    const result = runConv3DTestCase(
        batch, inDepth, inHeight, inWidth, inChannels, outChannels, fSize,
        fSize, fSize, pad, stride);

    const expectedOutput = [
      0.03703704, 0.11111111, 0.25925926, 0.33333333, 0.7037037, 0.77777778,
      0.92592593, 1.
    ];

    expectArraysClose(await result.data(), expectedOutput);
  });

  it('x=[1, 7, 7, 7, 1] f=[2, 2, 2, 1, 1] s=3 d=1 p=same', async () => {
    const batch = 1;
    const inDepth = 7;
    const inHeight = 7;
    const inWidth = 7;
    const inChannels = 1;
    const outChannels = 1;
    const fSize = 2;
    const pad = 'same';
    const stride = 3;
    const result = runConv3DTestCase(
        batch, inDepth, inHeight, inWidth, inChannels, outChannels, fSize,
        fSize, fSize, pad, stride);

    const expectedOutput = [
      0.54081633, 0.58017493, 0.28061224, 0.81632653, 0.85568513, 0.40306122,
      0.41873178, 0.4340379,  0.19642857, 2.46938776, 2.50874636, 1.1377551,
      2.74489796, 2.78425656, 1.26020408, 1.16873178, 1.1840379,  0.51785714,
      1.09511662, 1.10604956, 0.44642857, 1.17164723, 1.18258017, 0.47704082,
      0.3691691,  0.37244898, 0.125
    ];

    expectArraysClose(await result.data(), expectedOutput);
  });

  it('x=[1, 7, 7, 7, 1] f=[2, 2, 2, 1, 1] s=3 d=1 p=valid', async () => {
    const batch = 1;
    const inDepth = 7;
    const inHeight = 7;
    const inWidth = 7;
    const inChannels = 1;
    const outChannels = 1;
    const fSize = 2;
    const pad = 'valid';
    const stride = 3;
    const result = runConv3DTestCase(
        batch, inDepth, inHeight, inWidth, inChannels, outChannels, fSize,
        fSize, fSize, pad, stride);

    const expectedOutput = [
      0.540816, 0.580175, 0.816327, 0.855685, 2.469388, 2.508746, 2.744898,
      2.784257
    ];

    expectArraysClose(await result.data(), expectedOutput);
  });

  it('x=[1, 2, 1, 2, 1] f=[2, 1, 2, 1, 2] s=1 d=1 p=valid', async () => {
    const batch = 1;
    const inDepth = 2;
    const inHeight = 1;
    const inWidth = 2;
    const inChannels = 1;
    const outChannels = 2;
    const fDepth = 2;
    const fHeight = 1;
    const fWidth = 2;
    const pad = 'valid';
    const stride = 1;
    const result = runConv3DTestCase(
        batch, inDepth, inHeight, inWidth, inChannels, outChannels, fDepth,
        fHeight, fWidth, pad, stride);

    const expectedOutput = [1.5625, 1.875];

    expectArraysClose(await result.data(), expectedOutput);
  });

  it('gradient with clones, x=[1,3,6,1,1] filter=[2,2,1,1,1] s=1 d=1 p=valid',
     async () => {
       const batch = 1;
       const inDepth = 3;
       const inHeight = 6;
       const inWidth = 1;
       const inChannels = 1;
       const outChannels = 1;
       const fDepth = 2;
       const fHeight = 2;
       const fWidth = 1;
       const pad = 'valid';
       const stride = 1;
       const [dx, dfilter] = runGradientConv3DTestCase(
           batch, inDepth, inHeight, inWidth, inChannels, outChannels, fDepth,
           fHeight, fWidth, pad, stride);

       const expectedFilterOutput = [60.0, 70.0, 120.0, 130.0];
       const expectedOutput = [
         1.0, 3.0, 3.0, 3.0, 3.0, 2.0, 4.0, 10.0, 10.0, 10.0, 10.0, 6.0, 3.0,
         7.0, 7.0, 7.0, 7.0, 4.0
       ];
       expectArraysClose(await dx.data(), expectedOutput);
       expectArraysClose(await dfilter.data(), expectedFilterOutput);
     });

  it('throws when passed x as a non-tensor', () => {
    const inputDepth = 1;
    const outputDepth = 1;
    const fSize = 1;
    const pad = 'valid';
    const stride = 1;

    const w = tf.tensor5d([2], [fSize, fSize, fSize, inputDepth, outputDepth]);

    expect(() => tf.conv3d({} as tf.Tensor4D, w, stride, pad))
        .toThrowError(/Argument 'x' passed to 'conv3d' must be a Tensor/);
  });

  it('throws when passed filter as a non-tensor', () => {
    const inputDepth = 1;
    const inputShape: [number, number, number, number] = [2, 2, 1, inputDepth];
    const pad = 'valid';
    const stride = 1;

    const x = tf.tensor4d([1, 2, 3, 4], inputShape);

    expect(() => tf.conv3d(x, {} as Tensor5D, stride, pad))
        .toThrowError(/Argument 'filter' passed to 'conv3d' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const pad = 'valid';
    const stride = 1;
    const x = [[[[1], [2]], [[3], [4]]]];  // 2x2x1x1
    const w = [[[[[2]]]]];                 // 1x1x1x1x1

    const result = tf.conv3d(x, w, stride, pad);
    expectArraysClose(await result.data(), [2, 4, 6, 8]);
  });

  it('throws when data format not NDHWC', () => {
    const inputDepth = 1;
    const outputDepth = 1;
    const inputShape: [number, number, number, number] = [2, 2, 1, inputDepth];
    const pad = 'valid';
    const fSize = 1;
    const stride = 1;
    const dataFormat = 'NCDHW';

    const x = tf.tensor4d([1, 2, 3, 4], inputShape);
    const w = tf.tensor5d([2], [fSize, fSize, fSize, inputDepth, outputDepth]);

    expect(() => tf.conv3d(x, w, stride, pad, dataFormat)).toThrowError();
  });
});
